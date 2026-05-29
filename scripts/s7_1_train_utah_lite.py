#!/usr/bin/env python3
"""Train S7_1 UTAH-lite quality-aligned head from the E6 baseline."""

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

from s7_utah_quality_head_core import UtahLiteDetectionTrainer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", default="rgb_ir", choices=["rgb", "ir", "rgb_ir"])
    parser.add_argument("--model", default="/mnt/disk2/lhr/VSD/results/S2_fusion_mainline/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt")
    parser.add_argument("--data-rgb", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb.yaml")
    parser.add_argument("--data-rgb-ir", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml")
    parser.add_argument("--data-ir", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="/mnt/disk2/lhr/VSD/results/S7_architecture_incubation")
    parser.add_argument("--name", default="s7_1a_utah_lite_a06_b04")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", default="")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--quality-alpha", type=float, default=0.6)
    parser.add_argument("--quality-beta", type=float, default=0.4)
    parser.add_argument("--quality-loss-weight", type=float, default=0.2)
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
        "resume": str(Path(args.resume_path)) if args.resume_path else bool(args.resume),
        "exist_ok": bool(args.exist_ok),
    }
    payload = {
        "script": Path(__file__).name,
        "status": "dry_run" if args.dry_run else "start",
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "quality_alpha": args.quality_alpha,
        "quality_beta": args.quality_beta,
        "quality_loss_weight": args.quality_loss_weight,
        "overrides": overrides,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return

    trainer = UtahLiteDetectionTrainer(overrides=overrides)
    trainer.set_fusion_mode(args.mode)
    trainer.set_ir_data(str(Path(args.data_ir)) if args.mode in {"ir", "rgb_ir"} else None)
    trainer.set_utah_config(
        alpha=args.quality_alpha,
        beta=args.quality_beta,
        quality_loss_weight=args.quality_loss_weight,
    )
    trainer.train()


if __name__ == "__main__":
    main()
