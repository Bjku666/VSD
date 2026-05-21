#!/usr/bin/env python3
"""E13: E6 multi-scale fusion with tiny-aware bbox loss.

The model architecture is intentionally identical to E6. The only training-time
change is a scale-aware weighting term in the bbox regression loss for small
objects, keeping the ablation focused on loss behavior rather than capacity.
"""

from __future__ import annotations

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
from ultralytics.utils.tal import bbox2dist

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "") or str(Path(__file__).resolve().parent)
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from e6_feature_fusion_multiscale_core import E6DetectionTrainer, E6MultiScaleFusionModel


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

    def _scale_gain(self, target_bboxes: torch.Tensor, fg_mask: torch.Tensor, imgsz: torch.Tensor, stride: torch.Tensor):
        target_fg = target_bboxes[fg_mask]
        if target_fg.numel() == 0:
            return torch.ones((0, 1), device=target_bboxes.device, dtype=target_bboxes.dtype)
        if not self.use_scale:
            return torch.ones((target_fg.shape[0], 1), device=target_bboxes.device, dtype=target_bboxes.dtype)

        stride_fg = stride.view(1, -1, 1).expand(target_bboxes.shape[0], -1, -1)[fg_mask]
        target_px = target_fg * stride_fg
        wh = (target_px[:, 2:4] - target_px[:, 0:2]).clamp(min=1.0)
        image_area = (imgsz[0] * imgsz[1]).clamp(min=1.0)
        area_norm = (wh[:, 0:1] * wh[:, 1:2]) / image_area
        small_area = torch.as_tensor((self.small_px * self.small_px), device=target_bboxes.device, dtype=target_bboxes.dtype)
        small_norm = small_area / image_area

        ratio = small_norm / area_norm.clamp(min=1e-9)
        gain = 1.0 + self.alpha * (ratio.clamp(min=1.0).pow(self.gamma) - 1.0)
        return gain.clamp(min=1.0, max=self.max_gain)

    def _center_penalty(self, pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor, fg_mask: torch.Tensor):
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        scale_gain = self._scale_gain(target_bboxes, fg_mask, imgsz, stride).to(base_weight.dtype)
        weight = base_weight * scale_gain

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        center_penalty = self._center_penalty(pred_bboxes, target_bboxes, fg_mask)
        if center_penalty is not None:
            loss_iou = loss_iou + (self.center_alpha * center_penalty * weight).sum() / target_scores_sum

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
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.bbox_loss = ScaleAwareBboxLoss(
            self.reg_max,
            small_px=small_px,
            alpha=alpha,
            gamma=gamma,
            max_gain=max_gain,
            use_scale=use_scale,
            center_alpha=center_alpha,
            center_max=center_max,
        ).to(self.device)


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
        **kwargs,
    ):
        self.loss_mode = str(loss_mode)
        self.small_px = float(small_px)
        self.scale_alpha = float(scale_alpha)
        self.scale_gamma = float(scale_gamma)
        self.scale_max_gain = float(scale_max_gain)
        self.center_alpha = float(center_alpha)
        self.center_max = float(center_max)
        super().__init__(*args, **kwargs)

    def init_criterion(self):
        if self.loss_mode in {"baseline", "none"}:
            return super().init_criterion()
        supported = {"scale-aware", "center-aware", "scale-center-aware"}
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
    ) -> None:
        self.loss_mode = str(loss_mode)
        self.small_px = float(small_px)
        self.scale_alpha = float(scale_alpha)
        self.scale_gamma = float(scale_gamma)
        self.scale_max_gain = float(scale_max_gain)
        self.center_alpha = float(center_alpha)
        self.center_max = float(center_max)

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
        )
        if weights is not None:
            has_trained_ir_branch = self._weights_include_ir_branch(weights)
            model.load(weights)
            if not has_trained_ir_branch:
                model.initialize_ir_from_rgb()
        return model
