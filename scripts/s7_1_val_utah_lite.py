#!/usr/bin/env python3
"""Validate S7_1 UTAH-lite model with the unified E6 metric protocol."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "") or str(Path(__file__).resolve().parent)
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import e6_val_feature_fusion_multiscale as e6val
from s7_utah_quality_head_core import UtahLiteDetectionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--mode", default="rgb_ir", choices=["rgb", "ir", "rgb_ir"])
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out-dir", default="/mnt/disk2/lhr/VSD/results/S7_architecture_incubation/s7_1a_utah_lite_a06_b04_val")
    parser.add_argument("--quality-alpha", type=float, default=0.6)
    parser.add_argument("--quality-beta", type=float, default=0.4)
    parser.add_argument("--quality-loss-weight", type=float, default=0.2)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


class _ConfiguredUtahLiteTrainer(UtahLiteDetectionTrainer):
    utah_args: argparse.Namespace | None = None

    def setup_model(self):
        if self.utah_args is not None:
            self.set_utah_config(
                alpha=self.utah_args.quality_alpha,
                beta=self.utah_args.quality_beta,
                quality_loss_weight=self.utah_args.quality_loss_weight,
            )
        return super().setup_model()


def main() -> None:
    args = parse_args()
    _ConfiguredUtahLiteTrainer.utah_args = args
    e6val.E6DetectionTrainer = _ConfiguredUtahLiteTrainer
    forwarded = [
        "--weights",
        args.weights,
        "--mode",
        args.mode,
        "--split",
        args.split,
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        args.device,
        "--out-dir",
        args.out_dir,
        "--exist-ok",
    ]
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *forwarded]
        e6val.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
