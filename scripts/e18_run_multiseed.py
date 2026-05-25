#!/usr/bin/env python3
"""E18_check: audit whether E13_3b seed=1 and seed=2 runs are independent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


IMPORTANT_METRICS = (
    "mAP50",
    "mAP50-95",
    "Precision",
    "Recall",
    "AP_small",
    "Recall_small",
    "AP_dark",
    "Recall_dark",
    "AP_dark-small",
    "Recall_dark-small",
    "AP_tiny",
    "Recall_tiny",
    "AP_low-contrast",
    "Recall_low-contrast",
    "False Positives/image",
    "FPPI_dark",
    "FPPI_low-contrast",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in IMPORTANT_METRICS if key in metrics}


def _all_numeric_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if set(a) != set(b):
        return False
    for key in a:
        try:
            if float(a[key]) != float(b[key]):
                return False
        except Exception:
            if a[key] != b[key]:
                return False
    return True


def _audit_run(run_dir: Path, val_dir: Path, log_path: Path) -> dict[str, Any]:
    args = _read_yaml(run_dir / "args.yaml")
    metrics = _read_json(val_dir / "required_metrics.json")
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    return {
        "run_dir": str(run_dir),
        "val_dir": str(val_dir),
        "log_path": str(log_path),
        "exists": run_dir.exists(),
        "args_yaml_exists": (run_dir / "args.yaml").exists(),
        "required_metrics_exists": (val_dir / "required_metrics.json").exists(),
        "best_pt_exists": best.exists(),
        "last_pt_exists": last.exists(),
        "seed": args.get("seed"),
        "device": args.get("device"),
        "name": args.get("name"),
        "model": args.get("model"),
        "data": args.get("data"),
        "save_dir": args.get("save_dir"),
        "args_hash": _sha256(run_dir / "args.yaml"),
        "best_sha256": _sha256(best),
        "last_sha256": _sha256(last),
        "metrics": _metric_subset(metrics),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--seed1-run", default="/mnt/disk2/lhr/VSD/results/val/e18_5_e13_3b_seed1")
    parser.add_argument("--seed2-run", default="/mnt/disk2/lhr/VSD/results/val/e18_6_e13_3b_seed2")
    parser.add_argument("--seed1-val", default="/mnt/disk2/lhr/VSD/results/val/e18_5_e13_3b_seed1_val")
    parser.add_argument("--seed2-val", default="/mnt/disk2/lhr/VSD/results/val/e18_6_e13_3b_seed2_val")
    parser.add_argument("--seed1-log", default="/mnt/disk2/lhr/VSD/results/val/logs/e18_5_e13_3b_seed1_b16w4_gpu0_20260523.log")
    parser.add_argument("--seed2-log", default="/mnt/disk2/lhr/VSD/results/val/logs/e18_6_e13_3b_seed2_b16w4_gpu1_20260524.log")
    parser.add_argument("--out-dir", default="/mnt/disk2/lhr/VSD/results/val/e18_check_e13_3b_seed_integrity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed1 = _audit_run(Path(args.seed1_run), Path(args.seed1_val), Path(args.seed1_log))
    seed2 = _audit_run(Path(args.seed2_run), Path(args.seed2_val), Path(args.seed2_log))

    checks = {
        "args_seed_different": seed1.get("seed") != seed2.get("seed"),
        "run_dirs_different": seed1.get("run_dir") != seed2.get("run_dir"),
        "val_dirs_different": seed1.get("val_dir") != seed2.get("val_dir"),
        "save_dirs_different": seed1.get("save_dir") != seed2.get("save_dir"),
        "best_pt_hash_different": bool(seed1.get("best_sha256")) and seed1.get("best_sha256") != seed2.get("best_sha256"),
        "last_pt_hash_different": bool(seed1.get("last_sha256")) and seed1.get("last_sha256") != seed2.get("last_sha256"),
        "important_metrics_not_identical": not _all_numeric_equal(seed1["metrics"], seed2["metrics"]),
    }

    invalid_reasons: list[str] = []
    if not checks["args_seed_different"]:
        invalid_reasons.append("args.yaml seed values are not different")
    if not checks["run_dirs_different"] or not checks["val_dirs_different"]:
        invalid_reasons.append("run or validation directories overlap")
    if not checks["best_pt_hash_different"]:
        invalid_reasons.append("best.pt hashes are identical or missing")
    if not checks["important_metrics_not_identical"]:
        invalid_reasons.append("all important validation metrics are bit-identical")

    status = "valid_independent" if not invalid_reasons else "invalid_requires_seed2_rerun"
    summary = {
        "experiment": "E18_check",
        "status": status,
        "invalid_reasons": invalid_reasons,
        "checks": checks,
        "seed1": seed1,
        "seed2": seed2,
        "recommendation": (
            "Treat E13_3b multi-seed as invalid and rerun seed=2 from training if status is invalid."
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# E18_check E13_3b Seed Integrity",
        "",
        f"- Status: `{status}`",
        f"- Seed values: `{seed1.get('seed')}` vs `{seed2.get('seed')}`",
        f"- best.pt same hash: `{seed1.get('best_sha256') == seed2.get('best_sha256')}`",
        f"- important metrics identical: `{_all_numeric_equal(seed1['metrics'], seed2['metrics'])}`",
        "",
        "## Invalid Reasons",
    ]
    if invalid_reasons:
        lines.extend(f"- {reason}" for reason in invalid_reasons)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Key Metric Diff",
            "",
            "| Metric | seed=1 | seed=2 |",
            "| --- | ---: | ---: |",
        ]
    )
    for key in IMPORTANT_METRICS:
        if key in seed1["metrics"] or key in seed2["metrics"]:
            lines.append(f"| {key} | {seed1['metrics'].get(key, '')} | {seed2['metrics'].get(key, '')} |")
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "metrics_summary.md")


if __name__ == "__main__":
    main()
