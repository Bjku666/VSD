#!/usr/bin/env python3
"""Train E13 E6-derived model with tiny-aware / SLS-like bbox loss."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "") or str(Path(__file__).resolve().parent)
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from e13_tiny_aware_loss_core import E13DetectionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", type=str, default="rgb_ir", choices=["rgb", "ir", "rgb_ir"])
    parser.add_argument("--base", type=str, default="e6", help="Compatibility flag for demo scripts; E13 currently derives from E6.")
    parser.add_argument("--loss", type=str, default="scale-aware", choices=["baseline", "scale-aware", "center-aware", "scale-center-aware"])
    parser.add_argument("--model", type=str, default="/mnt/disk2/lhr/VSD/results/val/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt")
    parser.add_argument("--data-rgb", type=str, default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb.yaml")
    parser.add_argument("--data-rgb-ir", type=str, default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml")
    parser.add_argument("--data-ir", type=str, default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--device", type=str, default="0,1")
    parser.add_argument("--project", type=str, default="/mnt/disk2/lhr/VSD/results/val")
    parser.add_argument("--name", type=str, default="e13_2_e6_scale_aware_loss")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--small-px", type=float, default=32.0)
    parser.add_argument("--scale-alpha", type=float, default=1.0)
    parser.add_argument("--scale-gamma", type=float, default=0.5)
    parser.add_argument("--scale-max-gain", type=float, default=3.0)
    parser.add_argument("--center-alpha", type=float, default=0.25)
    parser.add_argument("--center-max", type=float, default=4.0)
    parser.add_argument("--small-threshold-source", default=None, help="Accepted for demo compatibility; fixed --small-px is used unless explicitly changed.")
    parser.add_argument("--validate-out", default=None, help="Accepted for demo compatibility; validation is handled by e13_val_tiny_aware_loss.py.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "rgb":
        data_yaml = args.data_rgb
    elif args.mode == "ir":
        data_yaml = args.data_ir
    else:
        data_yaml = args.data_rgb_ir

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

    payload = {
        "script": Path(__file__).name,
        "status": "ready",
        "dry_run": bool(args.dry_run),
        "mode": args.mode,
        "loss": args.loss,
        "small_px": args.small_px,
        "scale_alpha": args.scale_alpha,
        "scale_gamma": args.scale_gamma,
        "scale_max_gain": args.scale_max_gain,
        "center_alpha": args.center_alpha,
        "center_max": args.center_max,
        "overrides": overrides,
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    trainer = E13DetectionTrainer(overrides=overrides)
    trainer.set_fusion_mode(args.mode)
    trainer.set_ir_data(str(Path(args.data_ir)) if args.mode in {"ir", "rgb_ir"} else None)
    trainer.set_loss_config(
        loss_mode=args.loss,
        small_px=args.small_px,
        scale_alpha=args.scale_alpha,
        scale_gamma=args.scale_gamma,
        scale_max_gain=args.scale_max_gain,
        center_alpha=args.center_alpha,
        center_max=args.center_max,
    )
    trainer.train()


if __name__ == "__main__":
    main()
