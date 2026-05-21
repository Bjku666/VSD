#!/usr/bin/env python3
"""E12: E6 multi-scale fusion with lightweight residual gated fusion."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv
from ultralytics.utils import RANK

from e6_feature_fusion_multiscale_core import E6DetectionTrainer, E6MultiScaleFusionModel


class ResidualGatedFusion(nn.Module):
    """Average-preserving residual gate for same-channel RGB/IR features."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.avg = Conv(channels * 2, channels, k=1, s=1)
        self.residual = Conv(channels * 2, channels, k=1, s=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            conv = self.avg.conv
            conv.weight.zero_()
            for c in range(self.channels):
                conv.weight[c, c, 0, 0] = 0.5
                conv.weight[c, c + self.channels, 0, 0] = 0.5
            if conv.bias is not None:
                conv.bias.zero_()

            self.residual.conv.weight.zero_()
            if self.residual.conv.bias is not None:
                self.residual.conv.bias.zero_()

            gate_conv = self.gate[0]
            gate_conv.weight.zero_()
            if gate_conv.bias is not None:
                gate_conv.bias.zero_()

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        cat = torch.cat((rgb, ir), dim=1)
        diff = torch.abs(rgb - ir)
        gate = self.gate(torch.cat((rgb, ir, diff), dim=1))
        return self.avg(cat) + gate * self.residual(cat)


class E12ResidualGatedFusionModel(E6MultiScaleFusionModel):
    """YOLO11n E6 backbone with residual gated fusion at P3/P4/P5/top save nodes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channels_map = self._infer_fusion_channels()
        self.fusion_convs = nn.ModuleDict(
            {str(i): ResidualGatedFusion(channels_map[i]) for i in self.fusion_indices}
        )

    def predict(self, x, profile: bool = False, visualize: bool = False, augment: bool = False, embed=None):
        if not self._e6_ready:
            return super().predict(x, profile=profile, visualize=visualize, augment=augment, embed=embed)
        if augment:
            return self._predict_augment(x)

        y: list[Any] = [None] * len(self.model)

        if self.fusion_mode == "rgb":
            if x.shape[1] < 3:
                raise ValueError(f"RGB mode expects >=3 channels, got {x.shape[1]}")
            x_rgb = x[:, :3, :, :]
            x_rgb, _ = self._run_backbone_with_storage(x_rgb, use_ir=False, y=y)
            return self._run_shared_neck_head(x_rgb, y)

        if self.fusion_mode == "ir":
            if x.shape[1] >= 6:
                x_ir = x[:, 3:6, :, :]
            elif x.shape[1] >= 3:
                x_ir = x[:, :3, :, :]
            else:
                raise ValueError(f"IR mode expects >=3 channels, got {x.shape[1]}")
            x_ir, _ = self._run_backbone_with_storage(x_ir, use_ir=True, y=y)
            return self._run_shared_neck_head(x_ir, y)

        if x.shape[1] != 6:
            raise ValueError(f"rgb_ir mode expects 6 channels, got {x.shape[1]}")

        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:6, :, :]
        rgb_top, rgb_feats = self._run_backbone_with_storage(x_rgb, use_ir=False, y=y)
        _, ir_feats = self._run_backbone_with_storage(x_ir, use_ir=True, y=None)

        fused_top = rgb_top
        for i in self.fusion_indices:
            if i not in rgb_feats or i not in ir_feats:
                raise RuntimeError(f"Missing feature for fusion index {i}")
            fused = self.fusion_convs[str(i)](rgb_feats[i], ir_feats[i])
            y[i] = fused
            if i == self.backbone_last_idx:
                fused_top = fused
        return self._run_shared_neck_head(fused_top, y)


class E12DetectionTrainer(E6DetectionTrainer):
    """Trainer for E12 residual gated fusion experiments."""

    model_cfg = "yolo11n.yaml"

    @staticmethod
    def _load_state_from_weights(weights: str | nn.Module) -> dict[str, torch.Tensor]:
        if isinstance(weights, nn.Module):
            return weights.state_dict()
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        model = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
        return model.state_dict() if hasattr(model, "state_dict") else {}

    @staticmethod
    def _copy_e6_fusion_weights(model: E12ResidualGatedFusionModel, weights: str | nn.Module) -> None:
        try:
            state = E12DetectionTrainer._load_state_from_weights(weights)
        except Exception:
            return
        own = model.state_dict()
        updates: dict[str, torch.Tensor] = {}
        for idx in model.fusion_indices:
            old_prefix = f"fusion_convs.{idx}."
            new_prefix = f"fusion_convs.{idx}.avg."
            for key, value in state.items():
                if not key.startswith(old_prefix):
                    continue
                new_key = new_prefix + key[len(old_prefix):]
                if new_key in own and own[new_key].shape == value.shape:
                    updates[new_key] = value.float()
        if updates:
            own.update(updates)
            model.load_state_dict(own, strict=False)

    def get_model(self, cfg: str | dict | None = None, weights: str | nn.Module | None = None, verbose: bool = True):
        model = E12ResidualGatedFusionModel(
            cfg=self.model_cfg,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
            fusion_mode=self.fusion_mode,
        )
        if weights is not None:
            has_trained_ir_branch = self._weights_include_ir_branch(weights)
            model.load(weights)
            self._copy_e6_fusion_weights(model, weights)
            if not has_trained_ir_branch:
                model.initialize_ir_from_rgb()
        return model
