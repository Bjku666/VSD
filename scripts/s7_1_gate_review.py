#!/usr/bin/env python3
"""Review S7-A gate metrics for a single val-only candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE = {
    "mAP50-95": 0.635715,
    "AP_dark-small": 0.512464,
    "AP_tiny": 0.555855,
    "AP_low-contrast": 0.637506,
    "False Positives/image": 1.469027,
    "FPPI_dark": 2.536932,
    "FPPI_low-contrast": 1.612707,
    "AP_dark-small_object": 0.100028,
    "AP_tiny_object": 0.054049,
    "AP_low-contrast_object": 0.246427,
}

FLEXIBLE_FLOORS = {
    "AP_dark-small_object": 0.092,
    "AP_tiny_object": 0.053,
    "AP_low-contrast_object": 0.240,
}

KEYS = (
    "mAP50-95",
    "AP_dark-small",
    "AP_tiny",
    "AP_low-contrast",
    "False Positives/image",
    "FPPI_dark",
    "FPPI_low-contrast",
    "AP_dark-small_object",
    "AP_tiny_object",
    "AP_low-contrast_object",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON: {path}")
    return data


def _f(data: dict[str, Any], key: str) -> float | None:
    try:
        return float(data[key])
    except Exception:
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--image-metrics", required=True)
    parser.add_argument("--object-metrics", required=True)
    parser.add_argument("--pred-dir", default="")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = _read_json(Path(args.image_metrics))
    obj = _read_json(Path(args.object_metrics))
    metrics = {**image, **obj}

    fp_keys = ("False Positives/image", "FPPI_dark", "FPPI_low-contrast")
    fp_down = sum((_f(metrics, key) is not None and _f(metrics, key) < BASELINE[key]) for key in fp_keys)
    object_full = all(_f(metrics, key) is not None and _f(metrics, key) >= BASELINE[key] for key in (
        "AP_dark-small_object",
        "AP_tiny_object",
        "AP_low-contrast_object",
    ))
    object_flexible = all(_f(metrics, key) is not None and _f(metrics, key) >= FLEXIBLE_FLOORS[key] for key in FLEXIBLE_FLOORS)
    dark_image_up = (_f(metrics, "AP_dark-small") or 0.0) > BASELINE["AP_dark-small"]
    dark_object_full = (_f(metrics, "AP_dark-small_object") or 0.0) >= BASELINE["AP_dark-small_object"]

    if object_full and fp_down >= 2:
        label = "pass_candidate"
    elif dark_image_up and not dark_object_full:
        label = "image_level_trap"
    elif object_flexible and fp_down >= 2:
        label = "signal_only"
    else:
        label = "not_candidate"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for key in KEYS:
        value = _f(metrics, key)
        baseline = BASELINE[key]
        rows.append(
            {
                "metric": key,
                "value": value,
                "baseline": baseline,
                "delta": None if value is None else value - baseline,
            }
        )

    summary = {
        "exp_id": args.exp_id,
        "label": label,
        "fp_metrics_down_count": fp_down,
        "image_metrics": str(Path(args.image_metrics)),
        "object_metrics": str(Path(args.object_metrics)),
        "prediction_export": args.pred_dir,
        "baseline": BASELINE,
        "flexible_floors": FLEXIBLE_FLOORS,
        "comparison": rows,
    }
    (out_dir / "s7_gate_review.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        f"# {args.exp_id} S7 Gate Review",
        "",
        f"- Label: {label}",
        f"- FP metrics down: {fp_down}/3",
        f"- Prediction export: {args.pred_dir or 'pending'}",
        "",
        "| Metric | Value | E6 baseline | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['metric']} | {_fmt(row['value'])} | {_fmt(row['baseline'])} | {_fmt(row['delta'])} |"
        )
    (out_dir / "s7_gate_review.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out_dir / "s7_gate_review.md")


if __name__ == "__main__":
    main()
