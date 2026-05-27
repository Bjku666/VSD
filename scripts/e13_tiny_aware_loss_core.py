#!/usr/bin/env python3
"""E13: E6 multi-scale fusion with tiny-aware bbox loss.

The model architecture is intentionally identical to E6. The only training-time
change is a scale-aware weighting term in the bbox regression loss for small
objects, keeping the ablation focused on loss behavior rather than capacity.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import RANK
from ultralytics.utils.loss import DFLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, make_anchors

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "") or str(Path(__file__).resolve().parent)
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from e6_feature_fusion_multiscale_core import E6DetectionTrainer, E6MultiScaleFusionModel


def _load_class_confusion_map(path: str | None) -> dict[str, set[int]]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"class-confusion map not found: {source}")
    mapping: dict[str, set[int]] = {}
    if source.suffix.lower() == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"invalid class-confusion JSON map: {source}")
        for stem, values in raw.items():
            if isinstance(values, list):
                mapping[str(stem)] = {int(v) for v in values}
        return mapping
    with source.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("taxonomy") != "class_confusion":
                continue
            if row.get("model") and row.get("model") != "E6":
                continue
            stem = row.get("stem") or Path(row.get("image", "")).stem
            nearest = row.get("nearest_gt_class_id")
            if not stem or nearest in {None, ""}:
                continue
            mapping.setdefault(stem, set()).add(int(float(nearest)))
    return mapping


class ScaleAwareBboxLoss(nn.Module):
    """Bbox loss with capped extra weight for small assigned targets."""

    def __init__(
        self,
        reg_max: int = 16,
        small_px: float = 32.0,
        alpha: float = 1.0,
        gamma: float = 0.5,
        max_gain: float = 3.0,
        use_scale: bool = True,
        center_alpha: float = 0.0,
        center_max: float = 4.0,
        loss_scope: str = "all",
        aux_weight: float = 1.0,
        tiny_px: float = 29.7,
        dark_threshold: float = 33.50320816040039,
        low_contrast_threshold: float = 0.08425217866897583,
        contrast_ring_scale: float = 1.6,
    ):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.small_px = float(small_px)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.max_gain = float(max_gain)
        self.use_scale = bool(use_scale)
        self.center_alpha = float(center_alpha)
        self.center_max = float(center_max)
        self.loss_scope = str(loss_scope)
        self.aux_weight = float(aux_weight)
        self.tiny_px = float(tiny_px)
        self.dark_threshold = float(dark_threshold)
        self.low_contrast_threshold = float(low_contrast_threshold)
        self.contrast_ring_scale = float(contrast_ring_scale)

    def _target_size_info(
        self,
        target_bboxes: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_fg = target_bboxes[fg_mask]
        if target_fg.numel() == 0:
            empty = torch.ones((0, 1), device=target_bboxes.device, dtype=target_bboxes.dtype)
            return empty, empty.bool()

        stride_fg = stride.view(1, -1, 1).expand(target_bboxes.shape[0], -1, -1)[fg_mask]
        target_px = target_fg * stride_fg
        wh = (target_px[:, 2:4] - target_px[:, 0:2]).clamp(min=1.0)
        image_area = (imgsz[0] * imgsz[1]).clamp(min=1.0)
        area_norm = (wh[:, 0:1] * wh[:, 1:2]) / image_area
        small_area = torch.as_tensor((self.small_px * self.small_px), device=target_bboxes.device, dtype=target_bboxes.dtype)
        small_mask = (wh[:, 0:1] * wh[:, 1:2]) <= small_area
        return area_norm, small_mask

    def _scale_gain(
        self,
        target_bboxes: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
        target_scope_mask: torch.Tensor | None = None,
    ):
        target_fg = target_bboxes[fg_mask]
        if target_fg.numel() == 0:
            return torch.ones((0, 1), device=target_bboxes.device, dtype=target_bboxes.dtype)
        if not self.use_scale:
            return torch.ones((target_fg.shape[0], 1), device=target_bboxes.device, dtype=target_bboxes.dtype)

        area_norm, small_mask = self._target_size_info(target_bboxes, fg_mask, imgsz, stride)
        image_area = (imgsz[0] * imgsz[1]).clamp(min=1.0)
        small_area = torch.as_tensor((self.small_px * self.small_px), device=target_bboxes.device, dtype=target_bboxes.dtype)
        small_norm = small_area / image_area

        ratio = small_norm / area_norm.clamp(min=1e-9)
        gain = 1.0 + self.alpha * (ratio.clamp(min=1.0).pow(self.gamma) - 1.0)
        gain = gain.clamp(min=1.0, max=self.max_gain)
        if self.loss_scope == "small":
            gain = torch.where(small_mask, gain, torch.ones_like(gain))
        if target_scope_mask is not None:
            scope_fg = target_scope_mask[fg_mask].unsqueeze(-1).to(dtype=torch.bool)
            gain = torch.where(scope_fg, gain, torch.ones_like(gain))
        return 1.0 + self.aux_weight * (gain - 1.0)

    def _center_penalty(
        self,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
        target_scope_mask: torch.Tensor | None = None,
    ):
        if self.center_alpha <= 0:
            return None
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]
        if pred_fg.numel() == 0:
            return torch.ones((0, 1), device=target_bboxes.device, dtype=target_bboxes.dtype)

        pred_ctr = (pred_fg[:, 0:2] + pred_fg[:, 2:4]) * 0.5
        target_ctr = (target_fg[:, 0:2] + target_fg[:, 2:4]) * 0.5
        target_wh = (target_fg[:, 2:4] - target_fg[:, 0:2]).clamp(min=1.0)
        penalty = ((pred_ctr - target_ctr) / target_wh).pow(2).sum(dim=1, keepdim=True)
        if self.loss_scope == "small":
            _, small_mask = self._target_size_info(target_bboxes, fg_mask, imgsz, stride)
            penalty = torch.where(small_mask, penalty, torch.zeros_like(penalty))
        if target_scope_mask is not None:
            scope_fg = target_scope_mask[fg_mask].unsqueeze(-1).to(dtype=torch.bool)
            penalty = torch.where(scope_fg, penalty, torch.zeros_like(penalty))
        return penalty.clamp(min=0.0, max=self.center_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
        target_scope_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        scale_gain = self._scale_gain(target_bboxes, fg_mask, imgsz, stride, target_scope_mask).to(base_weight.dtype)
        weight = base_weight * scale_gain

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        center_penalty = self._center_penalty(pred_bboxes, target_bboxes, fg_mask, imgsz, stride, target_scope_mask)
        if center_penalty is not None:
            loss_iou = loss_iou + (self.center_alpha * self.aux_weight * center_penalty * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True)
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum

        return loss_iou, loss_dfl


class ScaleAwareDetectionLoss(v8DetectionLoss):
    """YOLO detection loss with the bbox component swapped for scale-aware weighting."""

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        small_px: float = 32.0,
        alpha: float = 1.0,
        gamma: float = 0.5,
        max_gain: float = 3.0,
        use_scale: bool = True,
        center_alpha: float = 0.0,
        center_max: float = 4.0,
        loss_scope: str = "all",
        aux_weight: float = 1.0,
        tiny_px: float = 29.7,
        dark_threshold: float = 33.50320816040039,
        low_contrast_threshold: float = 0.08425217866897583,
        contrast_ring_scale: float = 1.6,
        class_confusion_map: str | None = None,
        class_confusion_cls_gain: float = 1.0,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.class_confusion_targets = _load_class_confusion_map(class_confusion_map)
        self.class_confusion_cls_gain = float(class_confusion_cls_gain)
        self.bbox_loss = ScaleAwareBboxLoss(
            self.reg_max,
            small_px=small_px,
            alpha=alpha,
            gamma=gamma,
            max_gain=max_gain,
            use_scale=use_scale,
            center_alpha=center_alpha,
            center_max=center_max,
            loss_scope=loss_scope,
            aux_weight=aux_weight,
            tiny_px=tiny_px,
            dark_threshold=dark_threshold,
            low_contrast_threshold=low_contrast_threshold,
            contrast_ring_scale=contrast_ring_scale,
        ).to(self.device)

    def _box_contrast(
        self,
        gray: torch.Tensor,
        xyxy: torch.Tensor,
        ring_scale: float,
    ) -> torch.Tensor | None:
        height, width = gray.shape[-2:]
        x1, y1, x2, y2 = xyxy.tolist()
        x1_i = int(max(0, min(width - 1, torch.floor(torch.tensor(x1)).item())))
        y1_i = int(max(0, min(height - 1, torch.floor(torch.tensor(y1)).item())))
        x2_i = int(max(x1_i + 1, min(width, torch.ceil(torch.tensor(x2)).item())))
        y2_i = int(max(y1_i + 1, min(height, torch.ceil(torch.tensor(y2)).item())))
        inner = gray[y1_i:y2_i, x1_i:x2_i]
        if inner.numel() == 0:
            return None

        box_w = max(1, x2_i - x1_i)
        box_h = max(1, y2_i - y1_i)
        exp_w = max(box_w + 2, int(round(box_w * ring_scale)))
        exp_h = max(box_h + 2, int(round(box_h * ring_scale)))
        cx = (x1_i + x2_i) / 2.0
        cy = (y1_i + y2_i) / 2.0
        ex1 = int(max(0, round(cx - exp_w / 2.0)))
        ey1 = int(max(0, round(cy - exp_h / 2.0)))
        ex2 = int(min(width, round(cx + exp_w / 2.0)))
        ey2 = int(min(height, round(cy + exp_h / 2.0)))
        expanded = gray[ey1:ey2, ex1:ex2]
        if expanded.numel() == 0:
            return None

        mask = torch.ones(expanded.shape, dtype=torch.bool, device=gray.device)
        rx1 = int(max(0, min(expanded.shape[1], x1_i - ex1)))
        ry1 = int(max(0, min(expanded.shape[0], y1_i - ey1)))
        rx2 = int(max(rx1 + 1, min(expanded.shape[1], x2_i - ex1)))
        ry2 = int(max(ry1 + 1, min(expanded.shape[0], y2_i - ey1)))
        mask[ry1:ry2, rx1:rx2] = False
        ring = expanded[mask]
        if ring.numel() == 0:
            return None
        return (inner.mean() - ring.mean()).abs() / (ring.mean() + 1e-6)

    def _target_scope_mask(
        self,
        batch: dict[str, Any],
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.bbox_loss.loss_scope != "target":
            return None
        images = batch.get("img")
        if images is None or gt_bboxes.numel() == 0:
            return torch.zeros(gt_bboxes.shape[:2], device=self.device, dtype=torch.bool)

        imgs = images.to(self.device).float()
        rgb = imgs[:, :3] if imgs.shape[1] >= 3 else imgs
        gray = rgb.mean(dim=1)
        if gray.numel() and float(gray.detach().max()) <= 2.0:
            gray = gray * 255.0
        brightness = gray.flatten(1).mean(dim=1)

        scope = torch.zeros(gt_bboxes.shape[:2], device=self.device, dtype=torch.bool)
        small_area = self.bbox_loss.small_px * self.bbox_loss.small_px
        tiny_area = self.bbox_loss.tiny_px * self.bbox_loss.tiny_px
        for b in range(gt_bboxes.shape[0]):
            valid = mask_gt[b, :, 0].bool()
            if not valid.any():
                continue
            for j in torch.where(valid)[0].tolist():
                box = gt_bboxes[b, j]
                wh = (box[2:4] - box[0:2]).clamp(min=0.0)
                area = wh[0] * wh[1]
                is_tiny = bool((area <= tiny_area).item())
                is_dark_small = bool((brightness[b] <= self.bbox_loss.dark_threshold).item()) and bool(
                    (area <= small_area).item()
                )
                contrast = self._box_contrast(gray[b], box.detach(), self.bbox_loss.contrast_ring_scale)
                is_low_contrast = contrast is not None and bool((contrast <= self.bbox_loss.low_contrast_threshold).item())
                scope[b, j] = bool(is_tiny or is_dark_small or is_low_contrast)
        return scope

    def _class_confusion_gt_mask(
        self,
        batch: dict[str, Any],
        gt_labels: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.class_confusion_cls_gain <= 1.0 or not self.class_confusion_targets:
            return None
        im_files = batch.get("im_file") or batch.get("im_files")
        if im_files is None:
            return None
        scope = torch.zeros(gt_labels.shape[:2], device=self.device, dtype=torch.bool)
        for b, im_file in enumerate(im_files):
            classes = self.class_confusion_targets.get(Path(str(im_file)).stem)
            if not classes:
                continue
            valid = mask_gt[b, :, 0].bool()
            labels = gt_labels[b, :, 0].long()
            cls_mask = torch.zeros_like(valid, dtype=torch.bool)
            for cls_id in classes:
                cls_mask |= labels.eq(int(cls_id))
            scope[b] = valid & cls_mask
        return scope

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        gt_scope = self._target_scope_mask(batch, gt_bboxes, mask_gt)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        cls_loss = self.bce(pred_scores, target_scores.to(dtype))
        class_confusion_gt = self._class_confusion_gt_mask(batch, gt_labels, mask_gt)
        if class_confusion_gt is not None:
            gather_idx = target_gt_idx.clamp(min=0, max=max(class_confusion_gt.shape[1] - 1, 0))
            class_confusion_anchor = class_confusion_gt.gather(1, gather_idx) & fg_mask
            positive_cls = target_scores > 0
            gain = torch.as_tensor(self.class_confusion_cls_gain, device=self.device, dtype=cls_loss.dtype)
            cls_weight = torch.where(class_confusion_anchor.unsqueeze(-1) & positive_cls, gain, torch.ones_like(cls_loss))
            loss[1] = (cls_loss * cls_weight).sum() / target_scores_sum
        else:
            loss[1] = cls_loss.sum() / target_scores_sum

        target_scope_mask = None
        if gt_scope is not None:
            if gt_scope.shape[1] == 0:
                target_scope_mask = torch.zeros_like(fg_mask, dtype=torch.bool, device=self.device)
            else:
                gather_idx = target_gt_idx.clamp(min=0, max=max(gt_scope.shape[1] - 1, 0))
                target_scope_mask = gt_scope.gather(1, gather_idx)
                target_scope_mask = target_scope_mask & fg_mask

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
                target_scope_mask=target_scope_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()


class E13TinyAwareFusionModel(E6MultiScaleFusionModel):
    """E6 architecture with optional tiny-aware training criterion."""

    def __init__(
        self,
        *args,
        loss_mode: str = "scale-aware",
        small_px: float = 32.0,
        scale_alpha: float = 1.0,
        scale_gamma: float = 0.5,
        scale_max_gain: float = 3.0,
        center_alpha: float = 0.25,
        center_max: float = 4.0,
        loss_scope: str = "all",
        aux_weight: float = 1.0,
        tiny_px: float = 29.7,
        dark_threshold: float = 33.50320816040039,
        low_contrast_threshold: float = 0.08425217866897583,
        contrast_ring_scale: float = 1.6,
        class_confusion_map: str | None = None,
        class_confusion_cls_gain: float = 1.0,
        **kwargs,
    ):
        self.loss_mode = str(loss_mode)
        self.small_px = float(small_px)
        self.scale_alpha = float(scale_alpha)
        self.scale_gamma = float(scale_gamma)
        self.scale_max_gain = float(scale_max_gain)
        self.center_alpha = float(center_alpha)
        self.center_max = float(center_max)
        self.loss_scope = str(loss_scope)
        self.aux_weight = float(aux_weight)
        self.tiny_px = float(tiny_px)
        self.dark_threshold = float(dark_threshold)
        self.low_contrast_threshold = float(low_contrast_threshold)
        self.contrast_ring_scale = float(contrast_ring_scale)
        self.class_confusion_map = class_confusion_map
        self.class_confusion_cls_gain = float(class_confusion_cls_gain)
        super().__init__(*args, **kwargs)

    def init_criterion(self):
        if self.loss_mode in {"baseline", "none"}:
            return super().init_criterion()
        supported = {"scale-aware", "center-aware", "scale-center-aware", "class-confusion-cls"}
        if self.loss_mode not in supported:
            raise ValueError(f"Unsupported E13 loss mode: {self.loss_mode}")
        return ScaleAwareDetectionLoss(
            self,
            small_px=self.small_px,
            alpha=self.scale_alpha,
            gamma=self.scale_gamma,
            max_gain=self.scale_max_gain,
            use_scale=self.loss_mode in {"scale-aware", "scale-center-aware"},
            center_alpha=self.center_alpha if self.loss_mode in {"center-aware", "scale-center-aware"} else 0.0,
            center_max=self.center_max,
            loss_scope=self.loss_scope,
            aux_weight=self.aux_weight,
            tiny_px=self.tiny_px,
            dark_threshold=self.dark_threshold,
            low_contrast_threshold=self.low_contrast_threshold,
            contrast_ring_scale=self.contrast_ring_scale,
            class_confusion_map=self.class_confusion_map,
            class_confusion_cls_gain=self.class_confusion_cls_gain,
        )


class E13DetectionTrainer(E6DetectionTrainer):
    """E6 trainer variant that builds E13TinyAwareFusionModel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_mode = "scale-aware"
        self.small_px = 32.0
        self.scale_alpha = 1.0
        self.scale_gamma = 0.5
        self.scale_max_gain = 3.0
        self.center_alpha = 0.25
        self.center_max = 4.0
        self.loss_scope = "all"
        self.aux_weight = 1.0
        self.tiny_px = 29.7
        self.dark_threshold = 33.50320816040039
        self.low_contrast_threshold = 0.08425217866897583
        self.contrast_ring_scale = 1.6
        self.class_confusion_map = None
        self.class_confusion_cls_gain = 1.0

    def set_loss_config(
        self,
        *,
        loss_mode: str = "scale-aware",
        small_px: float = 32.0,
        scale_alpha: float = 1.0,
        scale_gamma: float = 0.5,
        scale_max_gain: float = 3.0,
        center_alpha: float = 0.25,
        center_max: float = 4.0,
        loss_scope: str = "all",
        aux_weight: float = 1.0,
        tiny_px: float = 29.7,
        dark_threshold: float = 33.50320816040039,
        low_contrast_threshold: float = 0.08425217866897583,
        contrast_ring_scale: float = 1.6,
        class_confusion_map: str | None = None,
        class_confusion_cls_gain: float = 1.0,
    ) -> None:
        self.loss_mode = str(loss_mode)
        self.small_px = float(small_px)
        self.scale_alpha = float(scale_alpha)
        self.scale_gamma = float(scale_gamma)
        self.scale_max_gain = float(scale_max_gain)
        self.center_alpha = float(center_alpha)
        self.center_max = float(center_max)
        self.loss_scope = str(loss_scope)
        self.aux_weight = float(aux_weight)
        self.tiny_px = float(tiny_px)
        self.dark_threshold = float(dark_threshold)
        self.low_contrast_threshold = float(low_contrast_threshold)
        self.contrast_ring_scale = float(contrast_ring_scale)
        self.class_confusion_map = class_confusion_map
        self.class_confusion_cls_gain = float(class_confusion_cls_gain)

    def get_model(self, cfg: str | dict | None = None, weights: str | nn.Module | None = None, verbose: bool = True):
        model = E13TinyAwareFusionModel(
            cfg=cfg or "yolo11n.yaml",
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
            fusion_mode=self.fusion_mode,
            loss_mode=self.loss_mode,
            small_px=self.small_px,
            scale_alpha=self.scale_alpha,
            scale_gamma=self.scale_gamma,
            scale_max_gain=self.scale_max_gain,
            center_alpha=self.center_alpha,
            center_max=self.center_max,
            loss_scope=self.loss_scope,
            aux_weight=self.aux_weight,
            tiny_px=self.tiny_px,
            dark_threshold=self.dark_threshold,
            low_contrast_threshold=self.low_contrast_threshold,
            contrast_ring_scale=self.contrast_ring_scale,
            class_confusion_map=self.class_confusion_map,
            class_confusion_cls_gain=self.class_confusion_cls_gain,
        )
        if weights is not None:
            has_trained_ir_branch = self._weights_include_ir_branch(weights)
            model.load(weights)
            if not has_trained_ir_branch:
                model.initialize_ir_from_rgb()
        return model
