#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/mnt/disk2/lhr/VSD")
RESULTS = ROOT / "results/S6_5_reliability_calibration"
LOG_DIR = RESULTS / "logs"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_trace(name: str, payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_json(LOG_DIR / name, payload)


def backfill_e25_1_full() -> None:
    out = RESULTS / "e25_1_full_e6_calibration_sweep"
    required = load_json(out / "required_metrics.json")
    best = load_json(out / "best_operating_points.json")
    target = best.get("best_accepting_gate") or best["best_f1"]
    required.update(
        {
            "mAP50-95": target.get("AP50-95_full"),
            "AP_dark-small": target.get("AP_dark-small"),
            "AP_tiny": target.get("AP_tiny"),
            "AP_low-contrast": target.get("AP_low-contrast"),
            "class_confusion_fp": target.get("val_class_confusion_fp"),
            "background_far_fp": target.get("val_background_far_fp"),
            "localization_error_fp": target.get("val_localization_error_fp"),
            "Recall_dark-small_object": target.get("Recall_dark-small_object"),
            "Recall_tiny_object": target.get("Recall_tiny_object"),
            "Recall_low-contrast_object": target.get("Recall_low-contrast_object"),
        }
    )
    # The previous full calibration grid did not store mAP50. Keep the missing
    # field explicit instead of inventing a value from AP50-95.
    required.setdefault("mAP50", None)
    write_json(out / "required_metrics.json", required)
    write_trace(
        "backfill_e25_1_full_trace_20260528.json",
        {
            "experiment": "E25_1_full",
            "source": str(out / "best_operating_points.json"),
            "status": "backfilled_existing_summary_fields",
            "unfilled_fields": ["mAP50"],
            "note": "mAP50 was not present in the existing full calibration grid; no test set or new training was run.",
        },
    )


def backfill_e26_1_full() -> None:
    out = RESULTS / "e26_1_full_classwise_threshold_calibration"
    required = load_json(out / "required_metrics.json")
    best = load_json(out / "best_operating_points.json")
    val = best["val"]
    obj = best["object"]
    required.update(
        {
            "mAP50": None,
            "mAP50-95": None,
            "AP_dark-small": None,
            "AP_tiny": None,
            "AP_low-contrast": None,
            "background_far_fp": val.get("background_far_fp"),
            "localization_error_fp": val.get("localization_error_fp"),
            "Recall_dark-small_object": obj.get("Recall_dark-small_object"),
            "Recall_tiny_object": obj.get("Recall_tiny_object"),
            "Recall_low-contrast_object": obj.get("Recall_low-contrast_object"),
        }
    )
    write_json(out / "required_metrics.json", required)
    write_trace(
        "backfill_e26_1_full_trace_20260528.json",
        {
            "experiment": "E26_1_full",
            "source": str(out / "best_operating_points.json"),
            "status": "backfilled_existing_summary_fields",
            "unfilled_fields": ["mAP50", "mAP50-95", "AP_dark-small", "AP_tiny", "AP_low-contrast"],
            "note": "The existing full class-wise threshold summary stores detection counts and object AP, but not image AP curves.",
        },
    )


def backfill_e26_2_taxonomy() -> None:
    for exp in ("e26_2a_class_confusion_cls125", "e26_2b_class_confusion_cls150"):
        val_dir = RESULTS / f"{exp}_val"
        required = load_json(val_dir / "required_metrics.json")
        summary = load_json(val_dir / "metrics_summary.json")
        full_err = summary.get("error_metrics", {}).get("full", {})
        # Confusion matrix-derived error metrics are not taxonomy FP. Keep
        # taxonomy missing explicit; this prevents accidental overclaiming.
        required.setdefault("class_confusion_fp", None)
        required.setdefault("background_far_fp", None)
        required.setdefault("localization_error_fp", None)
        required["confusion_offdiag"] = full_err.get("confusion_offdiag")
        write_json(val_dir / "required_metrics.json", required)


def backfill_e27_trace() -> None:
    out = RESULTS / "e27_1_full_metadata_verifier"
    pred = RESULTS / "e27_1_full_metadata_verifier_predictions"
    write_trace(
        "backfill_e27_1_full_trace_20260528.json",
        {
            "experiment": "E27_1_full",
            "status": "done_not_candidate",
            "existing_result_files": sorted(p.name for p in out.glob("*")),
            "existing_prediction_dirs": sorted(p.name for p in pred.glob("*")),
            "missing": [],
            "note": "Verifier full summary, predictions and required_metrics.json exist; experiment remains not_candidate because post-calibration object-level AP drops below E6 gate.",
        },
    )


def main() -> None:
    backfill_e25_1_full()
    backfill_e26_1_full()
    backfill_e26_2_taxonomy()
    backfill_e27_trace()


if __name__ == "__main__":
    main()
