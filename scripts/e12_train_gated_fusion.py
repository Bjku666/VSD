#!/usr/bin/env python3
"""Train E12: E6 RGB-IR fusion with residual gated fusion."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "/mnt/disk2/lhr/VSD/scripts")
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from e12_gated_fusion_core import E12DetectionTrainer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", default="rgb_ir", choices=["rgb", "ir", "rgb_ir"])
    parser.add_argument("--gate", default="residual", choices=["residual", "channel", "spatial", "dark-aware"])
    parser.add_argument("--gate-lambda", type=float, default=1.0, help="Scale for the gated residual branch; E12_1b uses 0.1.")
    parser.add_argument("--model", default="/mnt/disk2/lhr/VSD/results/S2_fusion_mainline/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt")
    parser.add_argument("--data-rgb", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb.yaml")
    parser.add_argument("--data-rgb-ir", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml")
    parser.add_argument("--data-ir", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--project", default="/mnt/disk2/lhr/VSD/results/S4_head_gate_loss")
    parser.add_argument("--name", default="e12_1_residual_gated_fusion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--validate-out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args, unknown = parser.parse_known_args()
    args.extra = unknown
    return args


def main() -> None:
    args = parse_args()
    if args.gate != "residual":
        raise SystemExit("Only --gate residual is implemented for E12_1; other gate modes are reserved demos.")
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
        print(
            json.dumps(
                {
                    "script": Path(__file__).name,
                    "status": "dry_run",
                    "gate": args.gate,
                    "gate_lambda": args.gate_lambda,
                    "overrides": overrides,
                    "extra_args": args.extra,
                },
                indent=2,
            ),
            flush=True,
        )
        return
    print(
        json.dumps(
            {
                "script": Path(__file__).name,
                "status": "start",
                "time": datetime.now().isoformat(timespec="seconds"),
                "gate": args.gate,
                "gate_lambda": args.gate_lambda,
                "overrides": overrides,
                "validate_out": args.validate_out,
                "extra_args": args.extra,
            },
            indent=2,
        ),
        flush=True,
    )
    trainer = E12DetectionTrainer(overrides=overrides)
    trainer.set_fusion_mode(args.mode)
    trainer.set_gate_lambda(args.gate_lambda)
    trainer.set_ir_data(str(Path(args.data_ir)) if args.mode in {"ir", "rgb_ir"} else None)
    trainer.train()

    if args.validate_out:
        weights = Path(args.project) / args.name / "weights" / "best.pt"
        print(
            json.dumps(
                {
                    "script": Path(__file__).name,
                    "status": "validate_start",
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "weights": str(weights),
                    "validate_out": args.validate_out,
                },
                indent=2,
            ),
            flush=True,
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(
            [
                sys.executable,
                "/mnt/disk2/lhr/VSD/scripts/e12_val_gated_fusion.py",
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
                "--gate-lambda",
                str(args.gate_lambda),
            ],
            check=True,
            env=env,
        )


if __name__ == "__main__":
    main()
