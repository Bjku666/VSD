#!/usr/bin/env python3
"""Train E6 dual-backbone multi-scale fusion model on DroneVehicle."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "")
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from e6_feature_fusion_multiscale_core import E6DetectionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11n E6 multi-scale fusion model")
    parser.add_argument("--mode", type=str, default="rgb_ir", choices=["rgb", "ir", "rgb_ir"])
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/disk2/lhr/VSD/weights/pretrained/yolo11n.pt",
        help="Pretrained checkpoint (.pt) or model yaml",
    )
    parser.add_argument(
        "--data-rgb",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb.yaml",
    )
    parser.add_argument(
        "--data-rgb-ir",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml",
    )
    parser.add_argument(
        "--data-ir",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="/mnt/disk2/lhr/VSD/results/val")
    parser.add_argument("--name", type=str, default="yolo11n_e6_rgb_ir_640")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "rgb":
        data_yaml = args.data_rgb
    elif args.mode == "ir":
        data_yaml = args.data_ir
    else:
        data_yaml = args.data_rgb_ir

    if args.mode in {"ir", "rgb_ir"} and not args.data_ir:
        raise ValueError("--data-ir is required for ir/rgb_ir modes")

    overrides = {
        "task": "detect",
        "mode": "train",
        "model": str(Path(args.model)),
        "data": str(Path(data_yaml)),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "device": args.device,
        "project": str(Path(args.project)),
        "name": args.name,
        "seed": int(args.seed),
        "deterministic": True,
        "patience": int(args.patience),
        "fraction": float(args.fraction),
        "close_mosaic": int(args.close_mosaic),
        "resume": bool(args.resume),
        "exist_ok": bool(args.exist_ok),
    }

    trainer = E6DetectionTrainer(overrides=overrides)
    trainer.set_fusion_mode(args.mode)
    trainer.set_ir_data(str(Path(args.data_ir)) if args.mode in {"ir", "rgb_ir"} else None)
    trainer.train()


if __name__ == "__main__":
    main()
