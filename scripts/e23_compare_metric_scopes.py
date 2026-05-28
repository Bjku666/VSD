#!/usr/bin/env python3
"""Compare image-level and object-level metric scopes from validation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCOPES = ("dark-small", "tiny", "low-contrast")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return ""


def _row_for(metrics_path: Path, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run = metrics_path.parent.name
    for scope in SCOPES:
        ap_image = metrics.get(f"AP_{scope}")
        recall_image = metrics.get(f"Recall_{scope}")
        ap_object = metrics.get(f"AP_{scope}_object")
        recall_object = metrics.get(f"Recall_{scope}_object")
        if ap_image is None and ap_object is None:
            continue
        try:
            delta = float(ap_object) - float(ap_image)
        except Exception:
            delta = ""
        rows.append(
            {
                "run": run,
                "metrics_path": str(metrics_path),
                "scope": scope,
                "AP_image": ap_image if ap_image is not None else "",
                "Recall_image": recall_image if recall_image is not None else "",
                "AP_object": ap_object if ap_object is not None else "",
                "Recall_object": recall_object if recall_object is not None else "",
                "AP_object_minus_image": delta,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--results-val", default="/mnt/disk2/lhr/VSD/results/S6_object_background_suppression")
    parser.add_argument("--out-dir", default="/mnt/disk2/lhr/VSD/results/S6_object_background_suppression/e23_metric_scope_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_val = Path(args.results_val)
    rows: list[dict[str, Any]] = []
    for path in sorted(results_val.glob("*/required_metrics.json")):
        metrics = _read_json(path)
        if metrics:
            rows.extend(_row_for(path, metrics))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metric_scope_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "run",
            "metrics_path",
            "scope",
            "AP_image",
            "Recall_image",
            "AP_object",
            "Recall_object",
            "AP_object_minus_image",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# E23 Metric Scope Comparison",
        "",
        "| Run | Scope | AP image | AP object | Recall image | Recall object | AP object - image |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['scope']} | {_fmt(row['AP_image'])} | {_fmt(row['AP_object'])} | "
            f"{_fmt(row['Recall_image'])} | {_fmt(row['Recall_object'])} | {_fmt(row['AP_object_minus_image'])} |"
        )
    md_path = out_dir / "metrics_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
