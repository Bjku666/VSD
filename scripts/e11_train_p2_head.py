#!/usr/bin/env python3
"""Train E11: E6 RGB-IR fusion with an added P2 detection head."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "/mnt/disk2/lhr/VSD/scripts")
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from e11_p2_head_core import E11DetectionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="rgb_ir", choices=["rgb", "ir", "rgb_ir"])
    parser.add_argument("--base", default="fusion")
    parser.add_argument("--model", default="/mnt/disk2/lhr/VSD/results/S2_fusion_mainline/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt")
    parser.add_argument("--cfg", default="/mnt/disk2/lhr/VSD/configs/models/yolo11n_e11_p2.yaml")
    parser.add_argument("--data-rgb", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb.yaml")
    parser.add_argument("--data-rgb-ir", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml")
    parser.add_argument("--data-ir", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--project", default="/mnt/disk2/lhr/VSD/results/S4_head_gate_loss")
    parser.add_argument("--name", default="e11_1_e6_p2_head")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--keep-p345-fusion", action="store_true")
    parser.add_argument("--validate-out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data_rgb if args.mode == "rgb" else args.data_ir if args.mode == "ir" else args.data_rgb_ir
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
    if args.dry_run:
        print(json.dumps({"script": Path(__file__).name, "status": "dry_run", "overrides": overrides, "weights": args.model, "cfg": args.cfg}, indent=2))
        return

    trainer = E11DetectionTrainer(overrides=overrides)
    trainer.set_fusion_mode(args.mode)
    trainer.set_ir_data(str(Path(args.data_ir)) if args.mode in {"ir", "rgb_ir"} else None)
    trainer.train()

    if args.validate_out:
        weights = Path(args.project) / args.name / "weights" / "best.pt"
        subprocess.run(
            [
                sys.executable,
                "/mnt/disk2/lhr/VSD/scripts/e11_val_p2_head.py",
                "--weights",
                str(weights),
                "--mode",
                args.mode,
                "--split",
                "val",
                "--imgsz",
                str(args.imgsz),
                "--batch",
                str(args.batch),
                "--workers",
                str(args.workers),
                "--device",
                args.device,
                "--out-dir",
                args.validate_out,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
