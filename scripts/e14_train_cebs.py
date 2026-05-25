#!/usr/bin/env python3
"""Train E14 CEBS model."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
from pathlib import Path

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "") or str(Path(__file__).resolve().parent)
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from e14_cebs_core import E14CEBSTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", default="rgb_ir", choices=["rgb", "ir", "rgb_ir"])
    parser.add_argument("--model", default="/mnt/disk2/lhr/VSD/results/val/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt")
    parser.add_argument("--data-rgb-ir", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml")
    parser.add_argument("--data-ir", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="/mnt/disk2/lhr/VSD/results/val")
    parser.add_argument("--name", default="e14_1_e6_cebs_a005")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cebs-alpha", type=float, default=0.05)
    parser.add_argument("--dark-threshold", type=float, default=33.50320816040039)
    parser.add_argument("--low-contrast-threshold", type=float, default=0.08425217866897583)
    parser.add_argument("--contrast-kernel", type=int, default=7)
    parser.add_argument("--suppression-temperature", type=float, default=0.08)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "task": "detect",
        "mode": "train",
        "model": str(Path(args.model)),
        "data": str(Path(args.data_rgb_ir)),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "device": args.device,
        "project": str(Path(args.project)),
        "name": args.name,
        "seed": int(args.seed),
        "deterministic": True,
        "close_mosaic": int(args.close_mosaic),
        "resume": bool(args.resume),
        "exist_ok": bool(args.exist_ok),
    }
    payload = {
        "script": Path(__file__).name,
        "status": "dry_run" if args.dry_run else "start",
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "cebs_alpha": args.cebs_alpha,
        "dark_threshold": args.dark_threshold,
        "low_contrast_threshold": args.low_contrast_threshold,
        "contrast_kernel": args.contrast_kernel,
        "suppression_temperature": args.suppression_temperature,
        "overrides": overrides,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    trainer = E14CEBSTrainer(overrides=overrides)
    trainer.set_fusion_mode(args.mode)
    trainer.set_ir_data(str(Path(args.data_ir)) if args.mode in {"ir", "rgb_ir"} else None)
    trainer.set_cebs_config(
        cebs_alpha=args.cebs_alpha,
        dark_threshold=args.dark_threshold,
        low_contrast_threshold=args.low_contrast_threshold,
        contrast_kernel=args.contrast_kernel,
        suppression_temperature=args.suppression_temperature,
    )
    trainer.train()


if __name__ == "__main__":
    main()
