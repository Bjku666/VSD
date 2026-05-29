#!/usr/bin/env python3
"""S7_1 UTAH-lite quality-aligned head for the E6 fusion model."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "") or str(Path(__file__).resolve().parent)
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from e6_feature_fusion_multiscale_core import E6DetectionTrainer, E6MultiScaleFusionModel


class UtahLiteDetect(Detect):
    """YOLO Detect head plus a light IoU-quality branch."""

    def __init__(
        self,
        nc: int = 80,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.quality_alpha = float(alpha)
        self.quality_beta = float(beta)
        c4 = max(16, ch[0] // 4) if ch else 16
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 1, 1)) for x in ch
        )
        self._init_quality_bias()

    def _init_quality_bias(self) -> None:
        for branch in self.cv4:
            if isinstance(branch[-1], nn.Conv2d):
                nn.init.constant_(branch[-1].bias, 0.0)

    @classmethod
    def from_detect(cls, detect: Detect, alpha: float = 0.6, beta: float = 0.4) -> "UtahLiteDetect":
        channels = tuple(int(branch[0].conv.in_channels) for branch in detect.cv2)
        out = cls(nc=detect.nc, reg_max=detect.reg_max, end2end=detect.end2end, ch=channels, alpha=alpha, beta=beta)
        out.cv2.load_state_dict(deepcopy(detect.cv2.state_dict()), strict=True)
        out.cv3.load_state_dict(deepcopy(detect.cv3.state_dict()), strict=True)
        out.dfl.load_state_dict(deepcopy(detect.dfl.state_dict()), strict=True)
        out.stride = detect.stride.clone()
        out.training = detect.training
        for attr in (
            "i",
            "f",
            "type",
            "np",
            "legacy",
            "xyxy",
            "export",
            "format",
            "dynamic",
            "max_det",
            "agnostic_nms",
        ):
            if hasattr(detect, attr):
                setattr(out, attr, getattr(detect, attr))
        if detect.end2end:
            out.one2one_cv2.load_state_dict(deepcopy(detect.one2one_cv2.state_dict()), strict=True)
            out.one2one_cv3.load_state_dict(deepcopy(detect.one2one_cv3.state_dict()), strict=True)
        return out

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        preds = super().forward_head(x, box_head=box_head, cls_head=cls_head)
        if preds:
            bs = x[0].shape[0]
            preds["quality"] = torch.cat([self.cv4[i](x[i]).view(bs, 1, -1) for i in range(self.nl)], dim=-1)
        return preds

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        dbox = self._get_decode_boxes(x)
        scores = x["scores"].sigmoid()
        quality = x.get("quality")
        if quality is not None:
            q = quality.sigmoid().clamp_(1e-4, 1.0)
            scores = scores.clamp_min(1e-8).pow(self.quality_alpha) * q.pow(self.quality_beta)
        return torch.cat((dbox, scores), 1)


class UtahQualityDetectionLoss(v8DetectionLoss):
    """Detection loss plus BCE quality supervision from assigned IoU."""

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        quality_weight: float = 0.2,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.quality_weight = float(quality_weight)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        (fg_mask, _target_gt_idx, target_bboxes, _anchor_points, stride_tensor), loss, loss_detach = (
            super().get_assigned_targets_and_loss(preds, batch)
        )
        quality_logits = preds.get("quality")
        if quality_logits is None:
            return (fg_mask, _target_gt_idx, target_bboxes, _anchor_points, stride_tensor), loss, loss_detach

        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_bboxes = self.bbox_decode(_anchor_points, pred_distri)
        quality_pred = quality_logits.permute(0, 2, 1).contiguous()
        quality_target = torch.zeros_like(quality_pred)
        quality_weight = torch.ones_like(quality_pred) * 0.25

        if fg_mask.sum():
            with torch.no_grad():
                iou_target = bbox_iou(
                    pred_bboxes[fg_mask].detach(),
                    (target_bboxes / stride_tensor)[fg_mask],
                    xywh=False,
                    CIoU=False,
                ).clamp(0.0, 1.0).view(-1, 1)
                quality_target[fg_mask] = iou_target.to(dtype=quality_target.dtype)
                quality_weight[fg_mask] = 1.0

        quality_loss = F.binary_cross_entropy_with_logits(
            quality_pred,
            quality_target,
            weight=quality_weight,
            reduction="sum",
        ) / max(float(quality_weight.sum().detach()), 1.0)
        loss = torch.cat((loss, quality_loss.unsqueeze(0) * self.quality_weight))
        loss_detach = torch.cat((loss_detach, quality_loss.detach().unsqueeze(0) * self.quality_weight))
        return (fg_mask, _target_gt_idx, target_bboxes, _anchor_points, stride_tensor), loss, loss_detach


class UtahLiteFusionModel(E6MultiScaleFusionModel):
    """E6 fusion architecture with a UTAH-lite quality-aligned detection head."""

    def __init__(
        self,
        *args,
        quality_alpha: float = 0.6,
        quality_beta: float = 0.4,
        quality_loss_weight: float = 0.2,
        **kwargs,
    ):
        self.quality_alpha = float(quality_alpha)
        self.quality_beta = float(quality_beta)
        self.quality_loss_weight = float(quality_loss_weight)
        super().__init__(*args, **kwargs)
        self._replace_detect_head()

    def _replace_detect_head(self) -> None:
        detect = self.model[-1]
        if isinstance(detect, UtahLiteDetect):
            detect.quality_alpha = self.quality_alpha
            detect.quality_beta = self.quality_beta
            return
        if not isinstance(detect, Detect):
            raise TypeError(f"UTAH-lite expects a Detect head, got {type(detect)!r}")
        self.model[-1] = UtahLiteDetect.from_detect(
            detect,
            alpha=self.quality_alpha,
            beta=self.quality_beta,
        )

    def init_criterion(self):
        return UtahQualityDetectionLoss(self, quality_weight=self.quality_loss_weight)


class UtahLiteDetectionTrainer(E6DetectionTrainer):
    """E6 trainer variant that builds the UTAH-lite quality head model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_alpha = 0.6
        self.quality_beta = 0.4
        self.quality_loss_weight = 0.2

    def set_utah_config(self, *, alpha: float = 0.6, beta: float = 0.4, quality_loss_weight: float = 0.2) -> None:
        self.quality_alpha = float(alpha)
        self.quality_beta = float(beta)
        self.quality_loss_weight = float(quality_loss_weight)

    def get_model(self, cfg: str | dict | None = None, weights: str | nn.Module | None = None, verbose: bool = True):
        model = UtahLiteFusionModel(
            cfg=cfg or "yolo11n.yaml",
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
            fusion_mode=self.fusion_mode,
            quality_alpha=self.quality_alpha,
            quality_beta=self.quality_beta,
            quality_loss_weight=self.quality_loss_weight,
        )
        if weights is not None:
            has_trained_ir_branch = self._weights_include_ir_branch(weights)
            model.load(weights)
            if not has_trained_ir_branch:
                model.initialize_ir_from_rgb()
            quality_keys = [k for k in model.state_dict() if ".cv4." in k]
            LOGGER.info(f"UTAH-lite quality head active: {len(quality_keys)} quality tensors")
        return model

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "quality_loss"
        return validator

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )
