#!/usr/bin/env python3
"""E14 CEBS: contrast-aware, error-guided background suppression for E6."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from ultralytics.utils import RANK

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "") or str(Path(__file__).resolve().parent)
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from e6_feature_fusion_multiscale_core import E6DetectionTrainer, E6MultiScaleFusionModel


class CEBSMultiScaleFusionModel(E6MultiScaleFusionModel):
    """E6 with lightweight target-preserving background attenuation on fused features."""

    def __init__(
        self,
        *args,
        cebs_alpha: float = 0.05,
        dark_threshold: float = 33.50320816040039,
        low_contrast_threshold: float = 0.08425217866897583,
        contrast_kernel: int = 7,
        suppression_temperature: float = 0.08,
        **kwargs,
    ):
        self.cebs_alpha = float(cebs_alpha)
        self.dark_threshold = float(dark_threshold)
        self.low_contrast_threshold = float(low_contrast_threshold)
        self.contrast_kernel = int(contrast_kernel)
        self.suppression_temperature = float(suppression_temperature)
        self.last_cebs_maps: dict[str, torch.Tensor] = {}
        super().__init__(*args, **kwargs)

    def _gray01(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        if gray.detach().numel() and float(gray.detach().max()) > 2.0:
            gray = gray / 255.0
        return gray.clamp(0.0, 1.0)

    def _local_contrast(self, gray: torch.Tensor) -> torch.Tensor:
        k = max(3, int(self.contrast_kernel))
        if k % 2 == 0:
            k += 1
        mean = F.avg_pool2d(gray, kernel_size=k, stride=1, padding=k // 2)
        return (gray - mean).abs() / mean.abs().clamp(min=1e-3)

    def _cebs_maps(self, x_rgb: torch.Tensor, x_ir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb_gray = self._gray01(x_rgb)
        ir_gray = self._gray01(x_ir)
        gray = 0.5 * (rgb_gray + ir_gray)
        contrast = torch.maximum(self._local_contrast(rgb_gray), self._local_contrast(ir_gray))

        dark_thr = torch.as_tensor(self.dark_threshold / 255.0, device=gray.device, dtype=gray.dtype)
        low_thr = torch.as_tensor(self.low_contrast_threshold, device=gray.device, dtype=gray.dtype)
        temp = max(float(self.suppression_temperature), 1e-3)

        dark_map = torch.sigmoid((dark_thr - gray) / temp)
        low_contrast_map = torch.sigmoid((low_thr - contrast) / temp)
        background_map = (dark_map * low_contrast_map).clamp(0.0, 1.0)

        # Preserve locally salient regions instead of globally enhancing dark images.
        target_mask = torch.sigmoid((contrast - low_thr) / temp).clamp(0.0, 1.0)
        return background_map, target_mask

    def _apply_cebs(self, fused: torch.Tensor, background: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.cebs_alpha <= 0:
            return fused
        b = F.interpolate(background, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        t = F.interpolate(target, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        gate = ((1.0 - t) * b).clamp(0.0, 1.0)
        return fused + self.cebs_alpha * gate * (-fused)

    def predict(self, x, profile: bool = False, visualize: bool = False, augment: bool = False, embed=None):
        if not self._e6_ready or self.fusion_mode != "rgb_ir" or augment:
            return super().predict(x, profile=profile, visualize=visualize, augment=augment, embed=embed)
        if x.shape[1] != 6:
            raise ValueError(f"rgb_ir mode expects 6 channels, got {x.shape[1]}")

        y: list[Any] = [None] * len(self.model)
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:6, :, :]
        background, target = self._cebs_maps(x_rgb, x_ir)
        self.last_cebs_maps = {
            "background_suppression": background.detach(),
            "target_preserving": target.detach(),
        }

        rgb_top, rgb_feats = self._run_backbone_with_storage(x_rgb, use_ir=False, y=y)
        _, ir_feats = self._run_backbone_with_storage(x_ir, use_ir=True, y=None)

        fused_top = rgb_top
        for i in self.fusion_indices:
            if i not in rgb_feats or i not in ir_feats:
                raise RuntimeError(f"Missing feature for fusion index {i}")
            fused = self.fusion_convs[str(i)](torch.cat((rgb_feats[i], ir_feats[i]), dim=1))
            fused = self._apply_cebs(fused, background, target)
            y[i] = fused
            if i == self.backbone_last_idx:
                fused_top = fused

        return self._run_shared_neck_head(fused_top, y)


class E14CEBSTrainer(E6DetectionTrainer):
    """Trainer that builds the E14 CEBS model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cebs_alpha = 0.05
        self.dark_threshold = 33.50320816040039
        self.low_contrast_threshold = 0.08425217866897583
        self.contrast_kernel = 7
        self.suppression_temperature = 0.08

    def set_cebs_config(
        self,
        *,
        cebs_alpha: float = 0.05,
        dark_threshold: float = 33.50320816040039,
        low_contrast_threshold: float = 0.08425217866897583,
        contrast_kernel: int = 7,
        suppression_temperature: float = 0.08,
    ) -> None:
        self.cebs_alpha = float(cebs_alpha)
        self.dark_threshold = float(dark_threshold)
        self.low_contrast_threshold = float(low_contrast_threshold)
        self.contrast_kernel = int(contrast_kernel)
        self.suppression_temperature = float(suppression_temperature)

    def get_model(self, cfg: str | dict | None = None, weights: str | None = None, verbose: bool = True):
        model = CEBSMultiScaleFusionModel(
            cfg=cfg or "yolo11n.yaml",
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
            fusion_mode=self.fusion_mode,
            cebs_alpha=self.cebs_alpha,
            dark_threshold=self.dark_threshold,
            low_contrast_threshold=self.low_contrast_threshold,
            contrast_kernel=self.contrast_kernel,
            suppression_temperature=self.suppression_temperature,
        )
        if weights is not None:
            has_trained_ir_branch = self._weights_include_ir_branch(weights)
            model.load(weights)
            if not has_trained_ir_branch:
                model.initialize_ir_from_rgb()
        return model
