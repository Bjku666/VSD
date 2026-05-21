#!/usr/bin/env python3
"""E11: E6 multi-scale RGB-IR fusion with an added P2 detection head.

P2 is used as a high-resolution detection feature only. The E6 P3/P4/P5 fusion
path is preserved, while layer 2 is explicitly excluded from cross-modal fusion.
"""

from __future__ import annotations

import torch.nn as nn
from ultralytics.utils import RANK

from e6_feature_fusion_multiscale_core import E6DetectionTrainer, E6MultiScaleFusionModel


class E11P2HeadFusionModel(E6MultiScaleFusionModel):
    """E6-style fusion model with P2 head and no P2 cross-modal re-fusion."""

    p2_index = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.p2_index in self.fusion_indices:
            self.fusion_indices = [i for i in self.fusion_indices if i != self.p2_index]
            del self.fusion_convs[str(self.p2_index)]


class E11DetectionTrainer(E6DetectionTrainer):
    """Trainer for E11 P2-head fusion experiments."""

    model_cfg = "/mnt/disk2/lhr/VSD/configs/models/yolo11n_e11_p2.yaml"

    @staticmethod
    def _weights_include_p2_head(weights: str | nn.Module) -> bool:
        if isinstance(weights, nn.Module):
            state = weights.state_dict()
        else:
            try:
                import torch

                ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
            except Exception:
                return False
            model = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
            if not hasattr(model, "state_dict"):
                return False
            state = model.state_dict()
        return any("model.29" in key or "model.30" in key for key in state)

    def get_model(self, cfg: str | dict | None = None, weights: str | nn.Module | None = None, verbose: bool = True):
        model = E11P2HeadFusionModel(
            cfg=self.model_cfg,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
            fusion_mode=self.fusion_mode,
        )
        if weights is not None:
            has_trained_ir_branch = self._weights_include_ir_branch(weights)
            model.load(weights)
            if not has_trained_ir_branch:
                model.initialize_ir_from_rgb()
        return model
