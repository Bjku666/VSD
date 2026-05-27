#!/usr/bin/env python3
"""E6: dual-backbone multi-scale feature fusion for YOLO11n.

This module provides:
1) E6MultiScaleFusionModel: RGB/IR dual-backbone with multi-scale fusion (concat + 1x1 conv).
2) PairedYOLODataset: strict RGB/IR synchronized augmentation wrapper.
3) E6DetectionTrainer: minimal-intrusion trainer extension for rgb/ir/rgb_ir modes.
"""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from ultralytics.data import build_yolo_dataset
from ultralytics.data import build as data_build
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class PairedYOLODataset(torch.utils.data.Dataset):
    """Wrap two YOLO datasets and keep stochastic augmentations strictly synchronized."""

    collate_fn = staticmethod(YOLODataset.collate_fn)

    def __init__(self, rgb_dataset: YOLODataset, ir_dataset: YOLODataset):
        self.rgb_dataset = rgb_dataset
        self.ir_dataset = ir_dataset
        self._align_ir_to_rgb()

    def __len__(self) -> int:
        return len(self.rgb_dataset)

    @property
    def rect(self) -> bool:
        return bool(getattr(self.rgb_dataset, "rect", False))

    def __getattr__(self, item: str) -> Any:
        return getattr(self.rgb_dataset, item)

    def _align_ir_to_rgb(self) -> None:
        rgb_files = [Path(p) for p in self.rgb_dataset.im_files]
        ir_files = [Path(p) for p in self.ir_dataset.im_files]
        rgb_stems = [p.stem for p in rgb_files]
        ir_stems = [p.stem for p in ir_files]

        if rgb_stems == ir_stems:
            return

        ir_map = {stem: i for i, stem in enumerate(ir_stems)}
        reorder: list[int] = []
        missing: list[str] = []
        for stem in rgb_stems:
            idx = ir_map.get(stem)
            if idx is None:
                missing.append(stem)
            else:
                reorder.append(idx)

        if missing:
            raise RuntimeError(
                f"IR dataset missing {len(missing)} paired files, first missing stem: {missing[0]}"
            )

        self._reorder_dataset_lists(self.ir_dataset, reorder)

    @staticmethod
    def _reorder_dataset_lists(dataset: YOLODataset, order: list[int]) -> None:
        for attr in ("im_files", "labels", "npy_files", "ims", "im_hw0", "im_hw"):
            if not hasattr(dataset, attr):
                continue
            value = getattr(dataset, attr)
            if isinstance(value, list) and len(value) == len(order):
                setattr(dataset, attr, [value[i] for i in order])

    def __getitem__(self, index: int) -> dict[str, Any]:
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()

        rgb_item = self.rgb_dataset[index]

        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        ir_item = self.ir_dataset[index]

        rgb_img = rgb_item["img"]
        ir_img = ir_item["img"]
        if rgb_img.shape[1:] != ir_img.shape[1:]:
            raise RuntimeError(
                f"RGB/IR transformed shapes mismatch: {tuple(rgb_img.shape)} vs {tuple(ir_img.shape)}"
            )

        rgb_item["img"] = torch.cat((rgb_img, ir_img), dim=0)
        return rgb_item


class E6MultiScaleFusionModel(DetectionModel):
    """YOLO11n dual-backbone model with multi-scale fusion before shared neck/head."""

    def __init__(
        self,
        cfg: str | dict = "yolo11n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        fusion_mode: str = "rgb_ir",
    ):
        self._e6_ready = False
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.backbone_end = len(self.yaml["backbone"])
        self.backbone_last_idx = self.backbone_end - 1
        self.fusion_indices = sorted(i for i in self.save if i < self.backbone_end)
        if not self.fusion_indices:
            raise RuntimeError("No valid fusion indices found in backbone save list")

        self.ir_backbone = nn.ModuleList(deepcopy(m) for m in self.model[: self.backbone_end])

        channels_map = self._infer_fusion_channels()
        self.fusion_convs = nn.ModuleDict(
            {
                str(i): Conv(channels_map[i] * 2, channels_map[i], k=1, s=1)
                for i in self.fusion_indices
            }
        )
        self._init_fusion_as_average(channels_map)

        self.fusion_mode = "rgb_ir"
        self.set_fusion_mode(fusion_mode)
        self._e6_ready = True

    def set_fusion_mode(self, fusion_mode: str) -> None:
        mode = str(fusion_mode).lower().strip()
        if mode not in {"rgb", "ir", "rgb_ir"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        self.fusion_mode = mode

    def initialize_ir_from_rgb(self) -> None:
        for i, dst in enumerate(self.ir_backbone):
            src = self.model[i]
            dst.load_state_dict(deepcopy(src.state_dict()), strict=True)

    def _infer_fusion_channels(self) -> dict[int, int]:
        out: dict[int, int] = {}
        with torch.no_grad():
            x = torch.zeros(1, 3, 64, 64)
            for i in range(self.backbone_end):
                x = self.model[i](x)
                if i in self.fusion_indices:
                    out[i] = int(x.shape[1])
        return out

    def _init_fusion_as_average(self, channels_map: dict[int, int]) -> None:
        with torch.no_grad():
            for i, channels in channels_map.items():
                conv = self.fusion_convs[str(i)].conv
                conv.weight.zero_()
                for c in range(channels):
                    conv.weight[c, c, 0, 0] = 0.5
                    conv.weight[c, c + channels, 0, 0] = 0.5
                if conv.bias is not None:
                    conv.bias.zero_()

    def _run_backbone_with_storage(
        self, x: torch.Tensor, use_ir: bool, y: list[Any] | None
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        modules = self.ir_backbone if use_ir else self.model[: self.backbone_end]
        feats: dict[int, torch.Tensor] = {}
        for i, m in enumerate(modules):
            x = m(x)
            if y is not None:
                y[i] = x if i in self.save else None
            if i in self.fusion_indices:
                feats[i] = x
        return x, feats

    def _run_shared_neck_head(self, x: torch.Tensor, y: list[Any]) -> torch.Tensor:
        for i in range(self.backbone_end, len(self.model)):
            m = self.model[i]
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y[i] = x if m.i in self.save else None
        return x

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
            fused = self.fusion_convs[str(i)](torch.cat((rgb_feats[i], ir_feats[i]), dim=1))
            y[i] = fused
            if i == self.backbone_last_idx:
                fused_top = fused

        return self._run_shared_neck_head(fused_top, y)


class E6DetectionTrainer(DetectionTrainer):
    """DetectionTrainer extension for E6 fusion modes with minimal framework changes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_mode = "rgb_ir"
        self.ir_data_yaml: str | None = None
        self._ir_data: dict[str, Any] | None = None

    def set_fusion_mode(self, fusion_mode: str) -> None:
        mode = str(fusion_mode).lower().strip()
        if mode not in {"rgb", "ir", "rgb_ir"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        self.fusion_mode = mode

    def set_ir_data(self, ir_data_yaml: str | None) -> None:
        self.ir_data_yaml = ir_data_yaml
        self._ir_data = None

    def _load_ir_data(self) -> dict[str, Any]:
        if self._ir_data is not None:
            return self._ir_data
        ir_cfg = self.data.get("ir") if isinstance(self.data, dict) else None
        if isinstance(ir_cfg, dict):
            merged = dict(self.data)
            ir_root = Path(ir_cfg.get("path", merged.get("path", ".")))
            for split in ("train", "val", "test"):
                split_value = ir_cfg.get(split)
                if split_value is None:
                    continue
                split_path = Path(str(split_value))
                merged[split] = str(split_path if split_path.is_absolute() else (ir_root / split_path))
            merged["path"] = str(ir_root)
            merged["channels"] = 3
            self._ir_data = merged
            return self._ir_data
        if not self.ir_data_yaml:
            raise ValueError("ir_data_yaml is required for ir/rgb_ir modes")
        self._ir_data = check_det_dataset(self.ir_data_yaml)
        return self._ir_data

    @staticmethod
    def _weights_include_ir_branch(weights: str | nn.Module) -> bool:
        if isinstance(weights, nn.Module):
            return any("ir_backbone" in key for key in weights.state_dict())
        try:
            ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        except Exception:
            return False
        model = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
        if not hasattr(model, "state_dict"):
            return False
        return any("ir_backbone" in key for key in model.state_dict())

    def get_model(self, cfg: str | dict | None = None, weights: str | nn.Module | None = None, verbose: bool = True):
        model = E6MultiScaleFusionModel(
            cfg=cfg or "yolo11n.yaml",
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

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)

        if self.fusion_mode == "rgb":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

        if self.fusion_mode == "ir":
            ir_data = self._load_ir_data()
            ir_img_path = ir_data[mode]
            return build_yolo_dataset(self.args, ir_img_path, batch, ir_data, mode=mode, rect=mode == "val", stride=gs)

        ir_data = self._load_ir_data()
        rgb_dataset = build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        ir_dataset = build_yolo_dataset(
            self.args,
            ir_data[mode],
            batch,
            ir_data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
        )
        return PairedYOLODataset(rgb_dataset, ir_dataset)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return self._build_seeded_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def _build_seeded_dataloader(
        self,
        dataset,
        batch: int,
        workers: int,
        shuffle: bool = True,
        rank: int = -1,
        drop_last: bool = False,
        pin_memory: bool = True,
    ):
        batch = min(batch, len(dataset))
        nd = torch.cuda.device_count()
        nw = min((data_build.os.cpu_count() or 1) // max(nd, 1), workers)
        sampler = (
            None
            if rank == -1
            else data_build.distributed.DistributedSampler(dataset, shuffle=shuffle)
            if shuffle
            else data_build.ContiguousDistributedSampler(dataset)
        )
        generator = torch.Generator()
        rank_offset = RANK if RANK != -1 else 0
        generator.manual_seed(6148914691236517205 + int(self.args.seed) * 1000003 + rank_offset)
        return data_build.InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=shuffle and sampler is None,
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=nd > 0 and pin_memory,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=data_build.seed_worker,
            generator=generator,
            drop_last=drop_last and len(dataset) % batch != 0,
        )
