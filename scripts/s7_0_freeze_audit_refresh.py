#!/usr/bin/env python3
"""Write the S7_0 freeze/audit refresh artifact for S7-A."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/mnt/disk2/lhr/VSD")
OUT = ROOT / "results/S7_architecture_incubation/s7_0_freeze_audit_refresh"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "S7_0",
        "status": "done",
        "time": datetime.now().isoformat(timespec="seconds"),
        "stage": "S7-A val-only architecture incubation",
        "allowed_next": ["S7_1a"],
        "blocked": ["E24_full", "E15", "E16", "E19", "E18_full", "E20_full", "E21"],
        "baseline": {
            "model": "E6 multi-scale fusion",
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
        },
        "s6_5_result": "audit_only_completed_no_candidate",
        "notes": [
            "E25_1_full, E26_1_full and E27_1_full reduce FP at fixed operating points but fail object-level AP gates.",
            "E26_2a/E26_2b remain failed_train_split_source_unverified.",
            "S7-A must use val-only full re-inference, object-level review and prediction export before any candidate promotion.",
        ],
    }
    (OUT / "s7_0_freeze_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# S7_0 Freeze & Audit Refresh",
        "",
        "- Status: done",
        "- Current stage: S7-A val-only architecture incubation",
        "- Next executable experiment: S7_1a UTAH-lite quality-aligned head",
        "- Blocked: E24_full, E15/E16/E19, E18_full, E20_full, E21",
        "",
        "## E6 Baseline",
        "",
    ]
    for key, value in summary["baseline"].items():
        md.append(f"- {key}: {value}")
    md.extend(
        [
            "",
            "## Audit Position",
            "",
            "- S6.5 closed as audit-only with no valid new candidate.",
            "- Cached calibration/verifier outputs remain preliminary only.",
            "- S7 candidates require full validation, object-level metrics and prediction export.",
        ]
    )
    (OUT / "s7_0_freeze_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(OUT / "s7_0_freeze_summary.md")


if __name__ == "__main__":
    main()
