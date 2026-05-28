#!/usr/bin/env python3
"""Audit S6 experiment artifacts for correctness and reproducibility."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("/mnt/disk2/lhr/VSD")
MANIFEST = ROOT / "configs/experiments/dark_small_next.yaml"
LEADERBOARD_CSV = ROOT / "results/dark_small_experiment_leaderboard.csv"
OUT_DIR = ROOT / "results/S6_object_background_suppression/s6_repro_audit"

S6_IDS = [
    "E18_check",
    "E22_1",
    "E23",
    "E13_3b_light",
    "E22_2a",
    "E22_2b",
    "E14_1",
    "E14_2",
    "E14_3",
    "E14_4",
    "E24_0",
]

TRAINED_IDS = {"E13_3b_light", "E22_2a", "E22_2b", "E14_1", "E14_2", "E14_3"}

KEY_METRICS = [
    "mAP50",
    "mAP50-95",
    "AP_small",
    "Recall_small",
    "AP_dark",
    "Recall_dark",
    "AP_dark-small",
    "AP_tiny",
    "AP_low-contrast",
    "AP_dark-small_object",
    "AP_tiny_object",
    "AP_low-contrast_object",
    "False Positives/image",
    "FPPI_dark",
    "FPPI_low-contrast",
]


@dataclass
class Check:
    name: str
    status: str
    detail: str


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def as_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def add_check(checks: list[Check], name: str, ok: bool, detail: str, warn: bool = False) -> None:
    if ok:
        status = "pass"
    elif warn:
        status = "warn"
    else:
        status = "fail"
    checks.append(Check(name=name, status=status, detail=detail))


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    eff = metrics.get("efficiency")
    if isinstance(eff, dict):
        for k, v in eff.items():
            out[f"efficiency.{k}"] = v
    return out


def load_leaderboard(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[str(row.get("id", ""))] = row
    return rows


def float_equal(a: Any, b: Any, tol: float = 5e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    status = run(["git", "status", "--short"])
    return {
        "head": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
        "diff_stat": run(["git", "diff", "--stat"]).splitlines(),
    }


def collect_exp_metrics(exp: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metric_path = Path(exp["metrics"]) if exp.get("metrics") else None
    object_metric_path = Path(exp["object_metrics"]) if exp.get("object_metrics") else None
    if metric_path and metric_path.exists():
        metrics.update(flatten_metrics(read_json(metric_path)))
    if object_metric_path and object_metric_path.exists():
        metrics.update(flatten_metrics(read_json(object_metric_path)))
    return metrics


def train_dir(exp: dict[str, Any]) -> Path | None:
    if not exp.get("project") or not exp.get("name"):
        return None
    return Path(exp["project"]) / str(exp["name"])


def audit(args: argparse.Namespace) -> tuple[list[Check], list[dict[str, Any]], dict[str, Any]]:
    checks: list[Check] = []
    snapshots: list[dict[str, Any]] = []

    manifest = read_yaml(Path(args.manifest))
    exps = {str(exp["id"]): exp for exp in manifest.get("experiments", []) if isinstance(exp, dict) and "id" in exp}
    leaderboard = load_leaderboard(Path(args.leaderboard))

    missing_ids = [eid for eid in S6_IDS if eid not in exps]
    add_check(checks, "manifest_s6_ids_present", not missing_ids, f"missing={missing_ids}")

    for eid in S6_IDS:
        exp = exps.get(eid)
        if not exp:
            continue
        add_check(checks, f"{eid}.status_present", bool(exp.get("status")), str(exp.get("status", "")))

        result_dir = Path(exp["result_dir"]) if exp.get("result_dir") else None
        if result_dir:
            should_exist = exp.get("status") not in {"skipped_not_justified", "blocked_no_valid_candidate"}
            exists = result_dir.exists()
            add_check(checks, f"{eid}.result_dir", exists if should_exist else True, rel(result_dir))

        metric_path = Path(exp["metrics"]) if exp.get("metrics") else None
        object_metric_path = Path(exp["object_metrics"]) if exp.get("object_metrics") else None
        if metric_path:
            add_check(checks, f"{eid}.metrics_json", metric_path.exists(), rel(metric_path))
        if object_metric_path:
            add_check(checks, f"{eid}.object_metrics_json", object_metric_path.exists(), rel(object_metric_path))

        if eid in TRAINED_IDS:
            tdir = train_dir(exp)
            assert tdir is not None
            for suffix in ("args.yaml", "results.csv", "weights/best.pt", "weights/last.pt"):
                path = tdir / suffix
                add_check(checks, f"{eid}.{suffix}", path.exists(), rel(path))

            args_path = tdir / "args.yaml"
            if args_path.exists():
                train_args = read_yaml(args_path)
                for key in ("seed", "imgsz", "batch", "workers"):
                    if key in exp:
                        add_check(
                            checks,
                            f"{eid}.args.{key}",
                            str(train_args.get(key)) == str(exp.get(key)),
                            f"args={train_args.get(key)} manifest={exp.get(key)}",
                        )
                add_check(checks, f"{eid}.args.name", str(train_args.get("name")) == str(exp.get("name")), f"args={train_args.get('name')} manifest={exp.get('name')}")
                if eid in {"E22_2a", "E22_2b"}:
                    data_arg = str(train_args.get("data", ""))
                    data_yaml = as_path(data_arg)
                    add_check(checks, f"{eid}.args.hn_data_yaml", data_yaml.exists(), rel(data_yaml))
                    if data_yaml.exists():
                        data_cfg = read_yaml(data_yaml)
                        reweight = data_cfg.get("reweight_source")
                        add_check(checks, f"{eid}.args.hn_reweight_source", isinstance(reweight, dict), rel(data_yaml))
                        if isinstance(reweight, dict):
                            expected_reweight = exp.get("train_reweight", {})
                            subset = str(reweight.get("subset_yaml", ""))
                            expected_subset = str(expected_reweight.get("subset", ""))
                            add_check(
                                checks,
                                f"{eid}.args.hn_subset",
                                subset == expected_subset and subset.endswith("background_far_train.yaml"),
                                f"data={subset} manifest={expected_subset}",
                            )
                            add_check(
                                checks,
                                f"{eid}.args.hn_multiplier",
                                float_equal(reweight.get("multiplier"), expected_reweight.get("multiplier")),
                                f"data={reweight.get('multiplier')} manifest={expected_reweight.get('multiplier')}",
                            )
                            add_check(checks, f"{eid}.args.hn_paired_modalities", reweight.get("paired_modalities") is True, str(reweight.get("paired_modalities")))
                        add_check(checks, f"{eid}.args.hn_train_list", as_path(data_cfg.get("train", "")).exists(), str(data_cfg.get("train", "")))
                        ir_cfg = data_cfg.get("ir", {})
                        ir_train = ir_cfg.get("train", "") if isinstance(ir_cfg, dict) else ""
                        add_check(checks, f"{eid}.args.hn_ir_train_list", as_path(ir_train).exists(), str(ir_train))

            best = tdir / "weights/best.pt"
            last = tdir / "weights/last.pt"
            snapshots.append(
                {
                    "id": eid,
                    "artifact": "best.pt",
                    "path": rel(best),
                    "sha256": sha256(best) if best.exists() else "",
                    "bytes": best.stat().st_size if best.exists() else "",
                }
            )
            snapshots.append(
                {
                    "id": eid,
                    "artifact": "last.pt",
                    "path": rel(last),
                    "sha256": sha256(last) if last.exists() else "",
                    "bytes": last.stat().st_size if last.exists() else "",
                }
            )

        metrics = collect_exp_metrics(exp)
        if metrics:
            lb_row = leaderboard.get(eid, {})
            for key in KEY_METRICS:
                if key not in metrics:
                    continue
                lb_value = lb_row.get(key, "")
                add_check(checks, f"{eid}.leaderboard.{key}", float_equal(metrics[key], lb_value), f"metrics={metrics[key]} leaderboard={lb_value}")
            for key in KEY_METRICS:
                if key in metrics:
                    snapshots.append(
                        {
                            "id": eid,
                            "artifact": key,
                            "path": rel(metric_path) if metric_path else "",
                            "value": metrics[key],
                        }
                    )

    hnyaml = ROOT / "configs/generated/s6_hard_negative/background_far_train.yaml"
    add_check(checks, "hn_train_yaml_exists", hnyaml.exists(), rel(hnyaml))
    if hnyaml.exists():
        hn = read_yaml(hnyaml)
        add_check(checks, "hn_source_split_train", hn.get("source_split") == "train", str(hn.get("source_split")))
        add_check(checks, "hn_allowed_taxonomy", hn.get("allowed_taxonomy") == "background_far", str(hn.get("allowed_taxonomy")))
        add_check(checks, "hn_source_is_train_lists", "e22_1_train_hard_negative_lists" in str(hn.get("source", "")), str(hn.get("source", "")))
        add_check(checks, "hn_unique_images_positive", int(hn.get("unique_images", 0)) > 0, str(hn.get("unique_images")))
        train_list = Path(str(hn.get("train", "")))
        add_check(checks, "hn_train_image_list_exists", train_list.exists(), rel(train_list))

    e18_summary = ROOT / "results/S6_object_background_suppression/e18_check_e13_3b_seed_integrity/summary.json"
    add_check(checks, "e18_summary_exists", e18_summary.exists(), rel(e18_summary))
    if e18_summary.exists():
        e18 = read_json(e18_summary)
        e18_invalid_as_expected = e18.get("status") == "invalid_requires_seed2_rerun"
        add_check(checks, "e18_status_recorded", e18_invalid_as_expected, str(e18.get("status")))
        if e18_invalid_as_expected:
            checks.append(
                Check(
                    name="e18_seed_integrity_invalid",
                    status="warn",
                    detail="E13_3b seed1/seed2 metrics are bit-identical; do not use this multi-seed mean as a valid candidate.",
                )
            )

    for log in [
        ROOT / "results/S6_object_background_suppression/logs/e13_3b_light_target_center_loss_gpu0_20260524_143843.log",
        ROOT / "results/S6_object_background_suppression/logs/e22_2a_e6_background_far_hn15_gpu0_b48_20260525_1800.log",
        ROOT / "results/S6_object_background_suppression/logs/e22_2a_e6_background_far_hn15_val_gpu0_20260526_0008.log",
        ROOT / "results/S6_object_background_suppression/logs/e22_2a_hn15_object_eval_gpu0_20260526_0012.log",
        ROOT / "results/S6_object_background_suppression/logs/e22_2b_e6_background_far_hn2_gpu1_b48_20260525_1935.log",
        ROOT / "results/S6_object_background_suppression/logs/e22_2b_hn2_unified_val_gpu1_20260526.log",
        ROOT / "results/S6_object_background_suppression/logs/e22_2b_hn2_object_eval_gpu1_20260526.log",
        ROOT / "results/S6_object_background_suppression/logs/e14_1_e6_cebs_a005_object_eval_gpu0_20260525_1626.log",
        ROOT / "results/S6_object_background_suppression/logs/e14_2_e6_cebs_a010_object_eval_gpu0_20260525_1231.log",
        ROOT / "results/S6_object_background_suppression/logs/e14_3_cebs_a005_unified_val_gpu0_20260526.log",
        ROOT / "results/S6_object_background_suppression/logs/e14_3_cebs_a005_object_eval_gpu0_20260526.log",
    ]:
        add_check(checks, f"log.exists.{log.name}", log.exists(), rel(log))
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="ignore")
            completion = any(token in text for token in ("Saved:", "Optimizer stripped", "VALIDATE_EXIT 0", "metrics_summary.md", "Results saved"))
            bad_tokens = [token for token in ("Traceback (most recent call last)", "RuntimeError:", "CUDA out of memory") if token in text]
            if bad_tokens:
                detail = f"{rel(log)}; tokens={bad_tokens}"
                if completion:
                    detail += "; final completion marker present"
                add_check(checks, f"log.no_error.{log.name}", False, detail, warn=completion)
            else:
                add_check(checks, f"log.no_error.{log.name}", True, rel(log))
            add_check(checks, f"log.completion.{log.name}", completion, rel(log))

    demo = subprocess.run(
        [str(ROOT / "scripts/train/run_dark_small_experiment_demos.sh"), "S6"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    add_check(checks, "demo_s6_exit_zero", demo.returncode == 0, f"returncode={demo.returncode}")
    add_check(checks, "demo_s6_has_dry_run", "--dry-run" in demo.stdout, "manifest-backed entries print --dry-run")
    add_check(checks, "demo_s6_has_object_eval", "e23_object_level_subset_eval.py" in demo.stdout, "object-level commands present")

    ginfo = git_info()
    add_check(checks, "git_worktree_clean", not ginfo["dirty"], "dirty files present" if ginfo["dirty"] else "clean", warn=True)

    summary = {
        "stage": "S6",
        "status": "pass" if not any(c.status == "fail" for c in checks) else "fail",
        "warnings": [c.__dict__ for c in checks if c.status == "warn"],
        "failures": [c.__dict__ for c in checks if c.status == "fail"],
        "git": ginfo,
    }
    return checks, snapshots, summary


def write_outputs(out_dir: Path, checks: list[Check], snapshots: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "checks.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "status", "detail"])
        writer.writeheader()
        writer.writerows([c.__dict__ for c in checks])
    with (out_dir / "metrics_snapshot.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames: list[str] = []
        for row in snapshots:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["id"])
        writer.writeheader()
        writer.writerows(snapshots)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# S6 Reproducibility Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Failures: `{len(summary['failures'])}`",
        f"- Warnings: `{len(summary['warnings'])}`",
        f"- Git HEAD: `{summary['git'].get('head', '')}`",
        f"- Git dirty: `{summary['git'].get('dirty')}`",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        detail = check.detail.replace("|", "\\|")
        lines.append(f"| {check.name} | {check.status} | {detail} |")
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--leaderboard", default=str(LEADERBOARD_CSV))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--strict", action="store_true", help="Return non-zero on warnings as well as failures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks, snapshots, summary = audit(args)
    write_outputs(Path(args.out_dir), checks, snapshots, summary)
    print(Path(args.out_dir) / "metrics_summary.md")
    if summary["status"] == "fail" or (args.strict and summary["warnings"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
