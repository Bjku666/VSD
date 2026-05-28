#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path("/mnt/disk2/lhr/VSD")
PY = ROOT / "../conda_envs/vsd/bin/python"
LOG_DIR = ROOT / "results/S6_5_reliability_calibration/logs"


def run_cmd(argv: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/mnt/disk2/lhr/conda_envs/vsd/lib:{env.get('LD_LIBRARY_PATH', '')}"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(argv) + "\n")
        f.flush()
        proc = subprocess.run(argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(argv)}")


def write_trace_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def backfill_e25_e26() -> None:
    run_cmd(
        [
            str(PY),
            "scripts/e25_e26_full_calibration.py",
            "e25_1_full",
            "--out-dir",
            str(ROOT / "results/S6_5_reliability_calibration/e25_1_full_e6_calibration_sweep"),
            "--weights",
            str(ROOT / "results/S2_fusion_mainline/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt"),
            "--imgsz",
            "640",
            "--batch",
            "32",
            "--workers",
            "8",
            "--device",
            "0",
        ],
        LOG_DIR / "backfill_e25_1_full_20260528.log",
    )
    run_cmd(
        [
            str(PY),
            "scripts/e25_e26_full_calibration.py",
            "e26_1_full",
            "--out-dir",
            str(ROOT / "results/S6_5_reliability_calibration/e26_1_full_classwise_threshold_calibration"),
            "--pred-root",
            str(ROOT / "results/S6_5_reliability_calibration/e25_1_full_e6_calibration_sweep/predictions"),
        ],
        LOG_DIR / "backfill_e26_1_full_20260528.log",
    )


def backfill_e27_trace_only() -> None:
    pred_dir = ROOT / "results/S6_5_reliability_calibration/e27_1_full_metadata_verifier_predictions"
    result_dir = ROOT / "results/S6_5_reliability_calibration/e27_1_full_metadata_verifier"
    summary = {
        "experiment": "E27_1_full",
        "status": "done_not_candidate",
        "reason": "full required_metrics.json already exists locally; verifier reduces FP/FPPI but post-calibration object-level AP drops below E6 gate",
        "existing_result_dir": str(result_dir),
        "existing_prediction_dir": str(pred_dir),
        "files_present": sorted(str(p.relative_to(ROOT)) for p in result_dir.glob("*")),
        "prediction_subdirs": sorted(str(p.relative_to(ROOT)) for p in pred_dir.glob("*")),
    }
    write_trace_log(LOG_DIR / "backfill_e27_1_full_trace_20260528.json", summary)


def refresh_leaderboard() -> None:
    run_cmd(
        [
            str(PY),
            "scripts/dark_small_experiment_runner.py",
            "--manifest",
            str(ROOT / "configs/experiments/dark_small_next.yaml"),
            "aggregate",
            "--out-dir",
            str(ROOT / "results/S6_5_reliability_calibration"),
        ],
        LOG_DIR / "backfill_leaderboard_20260528.log",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-e25-e26", action="store_true")
    parser.add_argument("--skip-e27-trace", action="store_true")
    parser.add_argument("--skip-leaderboard", action="store_true")
    args = parser.parse_args()

    if not args.skip_e25_e26:
        backfill_e25_e26()
    if not args.skip_e27_trace:
        backfill_e27_trace_only()
    if not args.skip_leaderboard:
        refresh_leaderboard()


if __name__ == "__main__":
    main()
