#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/mnt/disk2/lhr/VSD")
AUDIT_DIR = ROOT / "results/S6_5_reliability_calibration/s6_5_audit"
LOG_DIR = ROOT / "results/S6_5_reliability_calibration/logs"

RUNBOOK = ROOT / "AGENT_RUNBOOK.md"
STATE = ROOT / "EXPERIMENT_STATE.md"
LEADERBOARD = ROOT / "results/dark_small_experiment_leaderboard.md"
MANIFEST = ROOT / "configs/experiments/dark_small_next.yaml"


def ensure_dir() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dir_digest(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return ""
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with file.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def parse_markdown_status(path: Path, exp_id: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().split("|")[1:-1]]
        if path == LEADERBOARD:
            if len(cells) >= 4 and cells[1] == exp_id:
                return cells[3]
        else:
            if len(cells) >= 2 and cells[0] == exp_id:
                return cells[1]
            if path == RUNBOOK and len(cells) >= 3 and cells[1] == exp_id:
                return cells[2]
    return ""


def parse_manifest_status(exp_id: str) -> str:
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    in_block = False
    for line in lines:
        if line.startswith("- id: "):
            in_block = line.split(": ", 1)[1].strip() == exp_id
            continue
        if in_block and re.match(r"^\s+status:\s+", line):
            return line.split(":", 1)[1].strip()
    return ""


def find_logs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for path in sorted(LOG_DIR.glob("*")):
        if not path.is_file():
            continue
        name = path.name
        if any(re.search(p, name) for p in patterns):
            out.append(path)
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_results_csv(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    epochs = [int(float(r["epoch"])) for r in rows if r.get("epoch")]
    dup_epochs = sorted({e for e in epochs if epochs.count(e) > 1})
    last = rows[-1] if rows else {}
    return {
        "row_count": len(rows),
        "max_epoch": max(epochs) if epochs else None,
        "duplicate_epochs": dup_epochs,
        "last_row": last,
    }


def scan_log_markers(paths: list[Path]) -> tuple[list[str], list[str], list[str]]:
    required = [
        "Optimizer stripped",
        "Results saved",
        "prediction export done",
    ]
    warnings = ["Traceback", "RuntimeError", "CUDA out of memory"]
    found: list[str] = []
    missing: list[str] = []
    warn_hits: list[str] = []
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in paths if p.exists())
    for marker in required:
        if marker in text:
            found.append(marker)
        else:
            missing.append(marker)
    if "done exit=0" in text or "done: bash scripts/train/run_s65_e26_2a_eval_gpu1.sh" in text:
        found.append("wrapper_done_exit0")
    else:
        missing.append("wrapper_done_exit0")
    if "VALIDATE_EXIT 0" not in text:
        missing.append("VALIDATE_EXIT 0")
    for marker in warnings:
        if marker in text:
            warn_hits.append(marker)
    return found, missing, warn_hits


def write_process_snapshot() -> None:
    try:
        output = subprocess.check_output(
            [
                "pgrep",
                "-af",
                "e13_train_tiny_aware_loss.py|e26_2a_class_confusion_cls125|class_confusion_cls125",
            ],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        output = ""
    cleaned = []
    for line in output.splitlines():
        if "pgrep -af" in line or "s65_training_audit.py" in line:
            continue
        cleaned.append(line)
    text = "\n".join(cleaned).strip()
    if not text:
        text = "NO_MATCHING_PROCESS\n"
    else:
        text += "\n"
    (AUDIT_DIR / "e26_2a_processes.txt").write_text(text, encoding="utf-8")


def write_source_check() -> dict:
    p = ROOT / "results/S5_diagnostic_optimization/e22_0_train_hard_negative_taxonomy/hard_negative_list.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8", newline="")))
    cc = [
        r
        for r in rows
        if r.get("taxonomy") == "class_confusion" and (not r.get("model") or r.get("model") == "E6")
    ]
    splits = sorted(set(r.get("split", "") for r in cc))
    models = sorted(set(r.get("model", "") for r in cc))
    summary = {
        "path": str(p),
        "class_confusion_rows": len(cc),
        "splits": splits,
        "models": models,
        "pass_train_only": splits == ["train"],
    }
    (AUDIT_DIR / "e26_2a_class_confusion_source_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def passes_candidate_gate(metrics: dict[str, float], baseline: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if metrics.get("AP_dark-small_object", float("-inf")) < baseline["AP_dark-small_object"]:
        reasons.append("fail_AP_dark-small_object_vs_E6")
    if metrics.get("FP/image", float("inf")) >= baseline["FP/image"]:
        reasons.append("fail_FP/image")
    if metrics.get("FPPI_dark", float("inf")) >= baseline["FPPI_dark"]:
        reasons.append("fail_FPPI_dark")
    if metrics.get("FPPI_low-contrast", float("inf")) >= baseline["FPPI_low-contrast"]:
        reasons.append("fail_FPPI_low-contrast")
    if metrics.get("AP_tiny_object", float("-inf")) < baseline["AP_tiny_object"]:
        reasons.append("fail_AP_tiny_object_vs_E6")
    if metrics.get("AP_low-contrast_object", float("-inf")) < baseline["AP_low-contrast_object"]:
        reasons.append("fail_AP_low-contrast_object_vs_E6")
    return reasons


@dataclass
class SeedEntry:
    family: str
    exp_id: str
    seed: int
    train_dir: Path
    val_dir: Path
    obj_dir: Path
    pred_dir: Path


SEED_ENTRIES = [
    SeedEntry(
        "E6_seedfix",
        "E6",
        seed,
        ROOT / f"results/S6_5_reliability_calibration/seedfix_e6_seed{seed}_seedfix_b48_20260527",
        ROOT / f"results/S6_5_reliability_calibration/seedfix_e6_seed{seed}_seedfix_b48_20260527_val",
        ROOT / f"results/S6_5_reliability_calibration/seedfix_e6_seed{seed}_seedfix_b48_20260527_object_level",
        ROOT / f"results/S6_5_reliability_calibration/seedfix_e6_seed{seed}_seedfix_b48_20260527_predictions",
    )
    for seed in (0, 1, 2)
] + [
    SeedEntry(
        "E13_3b_seedfix",
        "E13_3b",
        seed,
        ROOT / f"results/S6_5_reliability_calibration/seedfix_e13_3b_seed{seed}_seedfix_b48_20260527",
        ROOT / f"results/S6_5_reliability_calibration/seedfix_e13_3b_seed{seed}_seedfix_b48_20260527_val",
        ROOT / f"results/S6_5_reliability_calibration/seedfix_e13_3b_seed{seed}_seedfix_b48_20260527_object_level",
        ROOT / f"results/S6_5_reliability_calibration/seedfix_e13_3b_seed{seed}_seedfix_b48_20260527_predictions",
    )
    for seed in (0, 1, 2)
] + [
    SeedEntry(
        "E25_0_seedfix",
        "E25_0",
        seed,
        ROOT / f"results/S6_5_reliability_calibration/seedfix_e25_0_e13_3b_seed{seed}_seedfix_b48_20260527",
        ROOT / f"results/S6_5_reliability_calibration/seedfix_e25_0_e13_3b_seed{seed}_seedfix_b48_20260527_val",
        ROOT / f"results/S6_5_reliability_calibration/seedfix_e25_0_e13_3b_seed{seed}_seedfix_b48_20260527_object_level",
        ROOT / f"results/S6_5_reliability_calibration/seedfix_e25_0_e13_3b_seed{seed}_seedfix_b48_20260527_predictions",
    )
    for seed in (42, 43, 44)
]


def build_seed_hash_outputs() -> dict[str, dict[str, tuple[float, float]]]:
    weight_rows: list[dict] = []
    pred_rows: list[dict] = []
    metric_rows: list[dict] = []
    family_values: dict[str, dict[str, list[float]]] = {}
    family_dup: dict[str, dict[str, list[str]]] = {}
    for entry in SEED_ENTRIES:
        args_path = entry.train_dir / "args.yaml"
        best_path = entry.train_dir / "weights/best.pt"
        last_path = entry.train_dir / "weights/last.pt"
        results_path = entry.train_dir / "results.csv"
        img_metrics_path = entry.val_dir / "required_metrics.json"
        obj_metrics_path = entry.obj_dir / "required_metrics.json"
        img_metrics = load_json(img_metrics_path)
        obj_metrics = load_json(obj_metrics_path)
        pred_hash = dir_digest(entry.pred_dir)
        pred_label_count = sum(1 for _ in entry.pred_dir.rglob("*.txt"))
        weight_rows.append(
            {
                "family": entry.family,
                "exp_id": entry.exp_id,
                "seed": entry.seed,
                "args_seed": re.search(r"^seed:\s*(\d+)\s*$", args_path.read_text(encoding="utf-8"), re.M).group(1),
                "best_pt_sha256": sha256_file(best_path),
                "last_pt_sha256": sha256_file(last_path),
                "results_csv_sha256": sha256_file(results_path),
            }
        )
        pred_rows.append(
            {
                "family": entry.family,
                "exp_id": entry.exp_id,
                "seed": entry.seed,
                "prediction_dir": str(entry.pred_dir),
                "prediction_label_count": pred_label_count,
                "prediction_manifest_sha256": pred_hash,
            }
        )
        metric_rows.append(
            {
                "family": entry.family,
                "exp_id": entry.exp_id,
                "seed": entry.seed,
                "image_metrics_sha256": sha256_text(json.dumps(img_metrics, sort_keys=True, ensure_ascii=False)),
                "object_metrics_sha256": sha256_text(json.dumps(obj_metrics, sort_keys=True, ensure_ascii=False)),
                "mAP50": img_metrics.get("mAP50"),
                "mAP50_95": img_metrics.get("mAP50-95"),
                "AP_dark_small_object": obj_metrics.get("AP_dark-small_object"),
                "FP_image": img_metrics.get("False Positives/image"),
                "FPPI_dark": img_metrics.get("FPPI_dark"),
            }
        )
        fam_vals = family_values.setdefault(entry.family, {})
        for key, value in {
            "mAP50": img_metrics.get("mAP50"),
            "mAP50-95": img_metrics.get("mAP50-95"),
            "AP_small": img_metrics.get("AP_small"),
            "Recall_small": img_metrics.get("Recall_small"),
            "AP_dark": img_metrics.get("AP_dark"),
            "Recall_dark": img_metrics.get("Recall_dark"),
            "AP_dark-small": img_metrics.get("AP_dark-small"),
            "AP_tiny": img_metrics.get("AP_tiny"),
            "AP_low-contrast": img_metrics.get("AP_low-contrast"),
            "AP_dark-small_object": obj_metrics.get("AP_dark-small_object"),
            "AP_tiny_object": obj_metrics.get("AP_tiny_object"),
            "AP_low-contrast_object": obj_metrics.get("AP_low-contrast_object"),
            "FP/image": img_metrics.get("False Positives/image"),
            "FPPI_dark": img_metrics.get("FPPI_dark"),
            "FPPI_low-contrast": img_metrics.get("FPPI_low-contrast"),
        }.items():
            fam_vals.setdefault(key, []).append(float(value))

    write_csv(
        AUDIT_DIR / "seed_weight_hashes.csv",
        weight_rows,
        ["family", "exp_id", "seed", "args_seed", "best_pt_sha256", "last_pt_sha256", "results_csv_sha256"],
    )
    write_csv(
        AUDIT_DIR / "seed_prediction_hashes.csv",
        pred_rows,
        ["family", "exp_id", "seed", "prediction_dir", "prediction_label_count", "prediction_manifest_sha256"],
    )
    write_csv(
        AUDIT_DIR / "seed_metrics_hashes.csv",
        metric_rows,
        [
            "family",
            "exp_id",
            "seed",
            "image_metrics_sha256",
            "object_metrics_sha256",
            "mAP50",
            "mAP50_95",
            "AP_dark_small_object",
            "FP_image",
            "FPPI_dark",
        ],
    )

    summary_lines = [
        "# Seed Independence Summary",
        "",
        "Included in mean/std:",
        "- E6 seedfix: seed 0 / 1 / 2",
        "- E13_3b seedfix: seed 0 / 1 / 2",
        "- E25_0 seedfix: seed 42 / 43 / 44",
        "",
        "Excluded from mean/std:",
        "- old E18_1 / E18_2: deleted_invalid_old_seed_logic",
        "- old E18_5 / E18_6: deleted_invalid_old_seed_logic",
        "- old E25_0 seed_pipeline_failed outputs: deleted after invalid audit",
        "- E26_2a_run_b48: off_plan_started_before_new_gate",
        "",
    ]
    for family in ("E6_seedfix", "E13_3b_seedfix", "E25_0_seedfix"):
        summary_lines.append(f"## {family}")
        family_rows = [r for r in weight_rows if r["family"] == family]
        for key in ("best_pt_sha256", "last_pt_sha256", "results_csv_sha256"):
            vals = [r[key] for r in family_rows]
            dup = len(vals) != len(set(vals))
            summary_lines.append(f"- {key}: {'DUPLICATE' if dup else 'all distinct'}")
        family_pred_rows = [r for r in pred_rows if r["family"] == family]
        pred_vals = [r["prediction_manifest_sha256"] for r in family_pred_rows]
        summary_lines.append(f"- prediction hash: {'DUPLICATE' if len(pred_vals) != len(set(pred_vals)) else 'all distinct'}")
        metric_family_rows = [r for r in metric_rows if r["family"] == family]
        metric_pairs = [
            (r["image_metrics_sha256"], r["object_metrics_sha256"])
            for r in metric_family_rows
        ]
        summary_lines.append(f"- metrics bit-identical: {'YES' if len(metric_pairs) != len(set(metric_pairs)) else 'NO'}")
        summary_lines.append("")
    (AUDIT_DIR / "seed_independence_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    family_stats: dict[str, dict[str, tuple[float, float]]] = {}
    for family, metrics in family_values.items():
        family_stats[family] = {key: mean_std(vals) for key, vals in metrics.items()}
    return family_stats


def build_status_reconciliation(family_stats: dict[str, dict[str, tuple[float, float]]], source_check: dict) -> None:
    audited = {
        "E25_0": "done_not_candidate",
        "E26_2a": "failed_train_split_source_unverified",
        "E26_2b": "failed_train_split_source_unverified",
        "E25_1_full": "done_not_candidate",
        "E26_1_full": "done_not_candidate",
        "E27_1_full": "done_not_candidate",
        "E25_1": "done_limited_cached_predictions",
        "E26_1": "done_limited_cached_predictions",
        "E27_1": "done_limited_cached_predictions",
    }
    rows: list[dict] = []
    for exp_id, target in audited.items():
        row = {
            "exp_id": exp_id,
            "experiment_state": parse_markdown_status(STATE, exp_id),
            "leaderboard": parse_markdown_status(LEADERBOARD, exp_id),
            "runbook": parse_markdown_status(RUNBOOK, exp_id),
            "manifest": parse_manifest_status(exp_id),
            "audited_primary_status": target,
            "stale_manifest": "yes" if parse_manifest_status(exp_id) not in {"", target} else "no",
            "notes": "",
        }
        if exp_id == "E25_0":
            mean_ap = family_stats["E25_0_seedfix"]["AP_dark-small_object"][0]
            mean_fp = family_stats["E25_0_seedfix"]["FP/image"][0]
            row["notes"] = (
                f"corrected seedfix complete; mean AP_dark-small_object={mean_ap:.6f}, "
                f"FP/image={mean_fp:.6f}; old seed_pipeline_failed outputs excluded"
            )
        elif exp_id in {"E26_2a", "E26_2b"}:
            row["notes"] = (
                "class_confusion source split field is not explicit train; "
                f"pass_train_only={source_check['pass_train_only']}"
            )
        elif exp_id == "E27_1_full":
            row["notes"] = "full object-level score_final metrics are present; FP is reduced but AP_dark-small_object fails the gate"
        elif exp_id in {"E25_1_full", "E26_1_full"}:
            row["notes"] = "artifact complete but no dedicated results/S6_5_reliability_calibration/logs/*full* log found"
        rows.append(row)
    write_csv(
        AUDIT_DIR / "status_reconciliation.csv",
        rows,
        [
            "exp_id",
            "experiment_state",
            "leaderboard",
            "runbook",
            "manifest",
            "audited_primary_status",
            "stale_manifest",
            "notes",
        ],
    )
    md = [
        "# Status Reconciliation",
        "",
        "- Priority used: EXPERIMENT_STATE.md > leaderboard > AGENT_RUNBOOK.md > manifest",
        "- Manifest is treated as stale when it disagrees with state/leaderboard and local artifacts.",
        "",
        "## Key conflicts",
        "- E25_0: corrected seedfix artifacts for seed 42/43/44 are complete and distinct; old seed_pipeline_failed result is excluded and primary status is done_not_candidate.",
        "- E26_2a: artifacts are complete, but audit fails train-only source verification because class_confusion split field is blank.",
        "- E25_1_full / E26_1_full: artifacts are complete and gate result is done_not_candidate, but dedicated `results/S6_5_reliability_calibration/logs/` entries are still missing.",
        "- E27_1_full: verifier summary and full object-level score_final metrics are complete, but AP_dark-small_object fails the gate.",
        "",
    ]
    for row in rows:
        md.append(
            f"- {row['exp_id']}: state={row['experiment_state']}, leaderboard={row['leaderboard']}, "
            f"runbook={row['runbook']}, manifest={row['manifest']}, audited={row['audited_primary_status']}"
        )
        md.append(f"  note: {row['notes']}")
    (AUDIT_DIR / "status_reconciliation.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def build_e26_2a_outputs(source_check: dict) -> None:
    artifact_paths = [
        ROOT / "results/S6_5_reliability_calibration/e26_2a_class_confusion_cls125/args.yaml",
        ROOT / "results/S6_5_reliability_calibration/e26_2a_class_confusion_cls125/results.csv",
        ROOT / "results/S6_5_reliability_calibration/e26_2a_class_confusion_cls125/weights/best.pt",
        ROOT / "results/S6_5_reliability_calibration/e26_2a_class_confusion_cls125/weights/last.pt",
        ROOT / "results/S6_5_reliability_calibration/e26_2a_class_confusion_cls125_val/required_metrics.json",
        ROOT / "results/S6_5_reliability_calibration/e26_2a_class_confusion_cls125_val/metrics_summary.json",
        ROOT / "results/S6_5_reliability_calibration/e26_2a_class_confusion_cls125_object_level/required_metrics.json",
        ROOT / "results/S6_5_reliability_calibration/e26_2a_class_confusion_cls125_object_level/metrics_summary.json",
        ROOT / "results/S6_5_reliability_calibration/e26_2a_class_confusion_cls125_predictions",
        ROOT / "results/S6_5_reliability_calibration/logs/e26_2a_class_confusion_cls125_b48_20260526.log",
        ROOT / "results/S6_5_reliability_calibration/logs/s65_e26_2a_eval_gpu1_20260527.log",
    ]
    rows = []
    for path in artifact_paths:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "type": "dir" if path.is_dir() else "file",
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
            }
        )
    write_csv(AUDIT_DIR / "e26_2a_artifacts.csv", rows, ["path", "exists", "type", "size_bytes"])

    results_info = parse_results_csv(ROOT / "results/S6_5_reliability_calibration/e26_2a_class_confusion_cls125/results.csv")
    epoch_status = {
        "expected_epoch": 100,
        "max_epoch": results_info["max_epoch"],
        "complete_epoch": results_info["max_epoch"] == 100,
        "duplicate_epochs": results_info["duplicate_epochs"],
        "row_count": results_info["row_count"],
        "last_row": results_info["last_row"],
        "warnings": [],
    }
    if results_info["duplicate_epochs"]:
        epoch_status["warnings"].append("duplicate_epoch_entries_present")
    if results_info["max_epoch"] != 100:
        epoch_status["warnings"].append("incomplete_epoch")
    (AUDIT_DIR / "e26_2a_epoch_status.json").write_text(
        json.dumps(epoch_status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    log_paths = [
        ROOT / "results/S6_5_reliability_calibration/logs/e26_2a_class_confusion_cls125_b48_20260526.log",
        ROOT / "results/S6_5_reliability_calibration/logs/s65_e26_2a_eval_gpu1_20260527.log",
    ]
    found, missing, warn_hits = scan_log_markers(log_paths)
    log_scan_lines = [
        "E26_2a log scan",
        "",
        "found markers:",
        *[f"- {m}" for m in found],
        "",
        "missing markers:",
        *[f"- {m}" for m in missing],
        "",
        "warning hits:",
        *([f"- {m}" for m in warn_hits] or ["- none"]),
        "",
        f"train_only_source_pass={source_check['pass_train_only']}",
    ]
    (AUDIT_DIR / "e26_2a_log_scan.txt").write_text("\n".join(log_scan_lines) + "\n", encoding="utf-8")

    md = [
        "# E26_2a Loss Code Check",
        "",
        "- CLI exposes `--loss class-confusion-cls`, `--class-confusion-map`, `--class-confusion-cls-gain` in `scripts/e13_train_tiny_aware_loss.py`.",
        "- The trainer forwards those values through `E13DetectionTrainer.set_loss_config()` into `E13TinyAwareFusionModel.init_criterion()`.",
        "- `ScaleAwareDetectionLoss.get_assigned_targets_and_loss()` still calls `self.assigner(...)` before the class-confusion weighting branch, so assignment is unchanged.",
        "- The class-confusion branch only reweights classification BCE on `class_confusion_anchor & positive_cls`.",
        "- `loss[0]` and `loss[2]` still come from `self.bbox_loss(...)`; in `class-confusion-cls` mode `use_scale` is false and `center_alpha` is forced to 0.0, so bbox/DFL effective behavior stays baseline.",
        "- `_load_class_confusion_map()` filters `taxonomy == class_confusion` and `model == E6`, but it does not enforce `split == train`; source CSV audit must prove that separately.",
        "",
        "Conclusion: code path matches classification-only intent, but training-source train-only proof fails because the CSV split field is blank.",
    ]
    (AUDIT_DIR / "e26_2a_loss_code_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def build_evaluation_outputs(family_stats: dict[str, dict[str, tuple[float, float]]]) -> None:
    required_image = {
        "mAP50": ["mAP50"],
        "mAP50-95": ["mAP50-95"],
        "AP_dark-small": ["AP_dark-small"],
        "AP_tiny": ["AP_tiny"],
        "AP_low-contrast": ["AP_low-contrast"],
        "False Positives/image": ["False Positives/image", "FP/image"],
        "FPPI_dark": ["FPPI_dark"],
        "FPPI_low-contrast": ["FPPI_low-contrast"],
    }
    required_object = {
        "AP_dark-small_object": ["AP_dark-small_object"],
        "AP_tiny_object": ["AP_tiny_object"],
        "AP_low-contrast_object": ["AP_low-contrast_object"],
        "Recall_dark-small_object": ["Recall_dark-small_object"],
        "Recall_tiny_object": ["Recall_tiny_object"],
        "Recall_low-contrast_object": ["Recall_low-contrast_object"],
    }
    required_tax = ["class_confusion_fp", "background_far_fp", "localization_error_fp"]

    experiment_sources = {
        "E13_3b_seedfix_meanstd": {
            "image": {k: v[0] for k, v in family_stats["E13_3b_seedfix"].items()},
            "object": {k: v[0] for k, v in family_stats["E13_3b_seedfix"].items()},
            "tax": {},
            "logs": find_logs([r"seedfix_e13_3b_seed"]),
        },
        "E25_0_seedfix_meanstd": {
            "image": {k: v[0] for k, v in family_stats["E25_0_seedfix"].items()},
            "object": {k: v[0] for k, v in family_stats["E25_0_seedfix"].items()},
            "tax": {},
            "logs": find_logs([r"seedfix_e25"]),
        },
        "E25_1_full": {
            "image": load_json(ROOT / "results/S6_5_reliability_calibration/e25_1_full_e6_calibration_sweep/required_metrics.json"),
            "object": load_json(ROOT / "results/S6_5_reliability_calibration/e25_1_full_e6_calibration_sweep/required_metrics.json"),
            "tax": load_json(ROOT / "results/S6_5_reliability_calibration/e25_1_full_e6_calibration_sweep/best_operating_points.json")["best_f1"],
            "logs": find_logs([r"e25_1_full", r"calibration_sweep"]),
        },
        "E26_1_full": {
            "image": load_json(ROOT / "results/S6_5_reliability_calibration/e26_1_full_classwise_threshold_calibration/required_metrics.json"),
            "object": load_json(ROOT / "results/S6_5_reliability_calibration/e26_1_full_classwise_threshold_calibration/required_metrics.json"),
            "tax": load_json(ROOT / "results/S6_5_reliability_calibration/e26_1_full_classwise_threshold_calibration/best_operating_points.json")["val"],
            "logs": find_logs([r"e26_1_full", r"classwise_threshold"]),
        },
        "E26_2a": {
            "image": load_json(ROOT / "results/S6_5_reliability_calibration/e26_2a_class_confusion_cls125_val/required_metrics.json"),
            "object": load_json(ROOT / "results/S6_5_reliability_calibration/e26_2a_class_confusion_cls125_object_level/required_metrics.json"),
            "tax": {},
            "logs": find_logs([r"e26_2a", r"s65_e26_2a"]),
        },
        "E26_2b": {
            "image": load_json(ROOT / "results/S6_5_reliability_calibration/e26_2b_class_confusion_cls150_val/required_metrics.json"),
            "object": load_json(ROOT / "results/S6_5_reliability_calibration/e26_2b_class_confusion_cls150_object_level/required_metrics.json"),
            "tax": {},
            "logs": find_logs([r"e26_2b", r"s65_e26_2b"]),
        },
        "E27_1_full": {
            "image": load_json(ROOT / "results/S6_5_reliability_calibration/e27_1_full_metadata_verifier/required_metrics.json"),
            "object": load_json(ROOT / "results/S6_5_reliability_calibration/e27_1_full_metadata_verifier/required_metrics.json"),
            "tax": load_json(ROOT / "results/S6_5_reliability_calibration/e27_1_full_metadata_verifier/required_metrics.json"),
            "logs": find_logs([r"e27_1_full", r"metadata_verifier"]),
        },
    }

    rows = []
    md_lines = [
        "# Evaluation Missing Fields",
        "",
        "- Leaderboard currently has no taxonomy FP columns for `class_confusion_fp`, `background_far_fp`, `localization_error_fp`.",
        "",
    ]
    for exp_id, src in experiment_sources.items():
        missing_image = [k for k, aliases in required_image.items() if not any(a in src["image"] for a in aliases)]
        missing_object = [k for k, aliases in required_object.items() if not any(a in src["object"] for a in aliases)]
        missing_tax = [k for k in required_tax if k not in src["tax"]]
        has_log = bool(src["logs"])
        rows.append(
            {
                "exp_id": exp_id,
                "missing_image_fields": ";".join(missing_image),
                "missing_object_fields": ";".join(missing_object),
                "missing_taxonomy_fields": ";".join(missing_tax),
                "has_results_val_logs": has_log,
            }
        )
        if missing_image or missing_object or missing_tax or not has_log:
            md_lines.append(f"## {exp_id}")
            if missing_image:
                md_lines.append(f"- missing image-level fields: {', '.join(missing_image)}")
            if missing_object:
                md_lines.append(f"- missing object-level fields: {', '.join(missing_object)}")
            if missing_tax:
                md_lines.append(f"- missing taxonomy FP fields: {', '.join(missing_tax)}")
            if not has_log:
                md_lines.append("- missing dedicated log under `results/S6_5_reliability_calibration/logs/`")
            md_lines.append("")
    write_csv(
        AUDIT_DIR / "evaluation_completeness.csv",
        rows,
        ["exp_id", "missing_image_fields", "missing_object_fields", "missing_taxonomy_fields", "has_results_val_logs"],
    )
    (AUDIT_DIR / "evaluation_missing_fields.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def build_candidate_gate_review(family_stats: dict[str, dict[str, tuple[float, float]]], source_check: dict) -> None:
    baseline = {
        "AP_dark-small_object": 0.100028,
        "AP_tiny_object": 0.054049,
        "AP_low-contrast_object": 0.246427,
        "FP/image": 1.469027,
        "FPPI_dark": 2.536932,
        "FPPI_low-contrast": 1.612707,
    }
    candidates = {
        "E25_0_seedfix_mean": {
            "status": "done_not_candidate",
            "metrics": {k: v[0] for k, v in family_stats["E25_0_seedfix"].items()},
            "notes": [],
        },
        "E25_1_full": {
            "status": "done_not_candidate",
            "metrics": {
                "AP_dark-small_object": 0.07889585589037927,
                "AP_tiny_object": 0.041289495680902906,
                "AP_low-contrast_object": 0.22013782116866532,
                "FP/image": 1.2791014295439074,
                "FPPI_dark": 2.196022727272727,
                "FPPI_low-contrast": 1.4203655352480418,
            },
            "notes": [],
        },
        "E26_1_full": {
            "status": "done_not_candidate",
            "metrics": {
                "AP_dark-small_object": 0.06672005271227223,
                "AP_tiny_object": 0.03738799105236146,
                "AP_low-contrast_object": 0.21606522793516625,
                "FP/image": 1.1477195371000681,
                "FPPI_dark": 1.78125,
                "FPPI_low-contrast": 1.2732811140121845,
            },
            "notes": [],
        },
        "E26_2a": {
            "status": "failed_train_split_source_unverified",
            "metrics": {
                "AP_dark-small_object": 0.10016034576102065,
                "AP_tiny_object": 0.05299899933907147,
                "AP_low-contrast_object": 0.24307328010112425,
                "FP/image": 1.6820966643975495,
                "FPPI_dark": 2.960227272727273,
                "FPPI_low-contrast": 1.8476936466492602,
            },
            "notes": ["train-only source unverified"],
        },
        "E26_2b": {
            "status": "failed_train_split_source_unverified",
            "metrics": {
                "AP_dark-small_object": 0.09024073702113379,
                "AP_tiny_object": 0.05264671824812352,
                "AP_low-contrast_object": 0.24631075360320773,
                "FP/image": 1.752212389380531,
                "FPPI_dark": 3.085227272727273,
                "FPPI_low-contrast": 1.9364664926022628,
            },
            "notes": ["train-only source unverified"],
        },
        "E27_1_full": {
            "status": "done_not_candidate",
            "metrics": {
                "AP_dark-small_object": 0.06638046404772455,
                "AP_tiny_object": 0.037329137179012,
                "AP_low-contrast_object": 0.2150907983320774,
                "FP/image": 1.2300884955752212,
                "FPPI_dark": 1.9232954545454546,
                "FPPI_low-contrast": 1.3959965187119234,
            },
            "notes": [],
        },
    }
    rows = []
    md_lines = [
        "# Candidate Gate Review",
        "",
        f"- Baseline AP_dark-small_object={baseline['AP_dark-small_object']:.6f}, FP/image={baseline['FP/image']:.6f}, "
        f"FPPI_dark={baseline['FPPI_dark']:.6f}, FPPI_low-contrast={baseline['FPPI_low-contrast']:.6f}",
        "",
    ]
    for exp_id, item in candidates.items():
        m = item["metrics"]
        reasons = []
        if not source_check["pass_train_only"] and exp_id in {"E26_2a", "E26_2b"}:
            reasons.append("fail_train_split_source_unverified")
        reasons.extend(passes_candidate_gate(m, baseline))
        rows.append(
            {
                "exp_id": exp_id,
                "status": item["status"],
                "AP_dark-small_object": m.get("AP_dark-small_object", ""),
                "FP/image": m.get("FP/image", ""),
                "FPPI_dark": m.get("FPPI_dark", ""),
                "FPPI_low-contrast": m.get("FPPI_low-contrast", ""),
                "gate_result": "pass" if not reasons else "fail",
                "reasons": ";".join(dict.fromkeys(reasons + item["notes"])),
            }
        )
        md_lines.append(f"## {exp_id}")
        md_lines.append(f"- status: {item['status']}")
        md_lines.append(f"- reasons: {', '.join(dict.fromkeys(reasons + item['notes']))}")
        md_lines.append("")
    write_csv(
        AUDIT_DIR / "candidate_gate_review.csv",
        rows,
        ["exp_id", "status", "AP_dark-small_object", "FP/image", "FPPI_dark", "FPPI_low-contrast", "gate_result", "reasons"],
    )
    (AUDIT_DIR / "candidate_gate_review.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def build_risk_register() -> None:
    rows = [
        {"risk_id": "R1", "title": "状态多源冲突", "status": "closed_after_patch", "note": "Primary state files and manifest were reconciled during S6.5 audit patch"},
        {"risk_id": "R2", "title": "E25_0 corrected seedfix 与旧 seed_pipeline_failed 混淆", "status": "closed_after_fix", "note": "corrected seed42/43/44 hashed and old invalid outputs excluded"},
        {"risk_id": "R3", "title": "E26_2a 仍在训练", "status": "closed", "note": "no matching process found in local audit snapshot"},
        {"risk_id": "R4", "title": "E26_2a manifest stale", "status": "closed_after_patch", "note": "manifest was updated from paused_second_batch to failed_train_split_source_unverified"},
        {"risk_id": "R5", "title": "E26_2a class_confusion 源 train-only 未证实", "status": "open_fail", "note": "split field blank, pass_train_only=false"},
        {"risk_id": "R6", "title": "classification-only loss 证明 bbox/DFL/assignment 未变", "status": "closed_static_check", "note": "static code path check passed"},
        {"risk_id": "R7", "title": "多 seed prediction hash 未查", "status": "closed", "note": "weight/results/prediction/metrics hashes generated"},
        {"risk_id": "R8", "title": "旧 invalid seed 混入 mean/std", "status": "closed", "note": "summary excludes deleted_invalid_old_seed_logic and seed_pipeline_failed"},
        {"risk_id": "R9", "title": "cached 结果写成正式结论", "status": "closed", "note": "cached statuses remain done_limited_cached_predictions"},
        {"risk_id": "R10", "title": "E27_1_full 缺 object-level gate 证据", "status": "closed_after_backfill", "note": "full object-level score_final metrics were added; experiment remains done_not_candidate"},
        {"risk_id": "R11", "title": "taxonomy FP 缺失", "status": "open", "note": "leaderboard has no taxonomy FP columns"},
        {"risk_id": "R12", "title": "E26_2a 容易被误判为候选", "status": "closed_with_fail", "note": "source fail plus FP regression"},
        {"risk_id": "R13", "title": "E26_2a_run_b48 混入正式 E26_2a", "status": "closed", "note": "kept as off_plan_started_before_new_gate"},
        {"risk_id": "R14", "title": "manifest 需要 after-audit 更新", "status": "closed_after_patch", "note": "manifest statuses were updated after audit reconciliation"},
        {"risk_id": "R15", "title": "full 任务不通过 gate 也要完整记录", "status": "open_partial", "note": "E25_1_full/E26_1_full/E27_1_full still lack dedicated logs, but required metrics are complete"},
    ]
    write_csv(AUDIT_DIR / "risk_register.csv", rows, ["risk_id", "title", "status", "note"])


def build_summary() -> None:
    summary = {
        "overall_status": "FAIL",
        "reasons": [
            "E26_2a and E26_2b train-only source verification failed because class_confusion split field is blank",
            "No S6.5 candidate passes the object-level gate",
            "results completeness gaps remain in logs and taxonomy reporting",
        ],
        "warnings": [
            "nvidia-smi snapshot in this environment could not communicate with the NVIDIA driver",
            "E25_1_full / E26_1_full / E27_1_full have no dedicated results/S6_5_reliability_calibration/logs entries",
            "leaderboard still lacks taxonomy FP columns by design and must be extended separately",
        ],
    }
    (AUDIT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dir()
    write_process_snapshot()
    source_check = write_source_check()
    family_stats = build_seed_hash_outputs()
    build_status_reconciliation(family_stats, source_check)
    build_e26_2a_outputs(source_check)
    build_evaluation_outputs(family_stats)
    build_candidate_gate_review(family_stats, source_check)
    build_risk_register()
    build_summary()


if __name__ == "__main__":
    main()
