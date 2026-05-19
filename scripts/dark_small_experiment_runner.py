#!/usr/bin/env python3
"""执行并汇总 DroneVehicle 暗弱小目标实验方案。

清单文件保持显式配置：每个实验都有稳定 ID、具体训练/验证命令、
固定输出目录，便于大规模实验断点继续和统一汇总。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_MANIFEST = Path("/mnt/disk2/lhr/VSD/configs/experiments/dark_small_next.yaml")


@dataclass(frozen=True)
class CommandStep:
    label: str
    argv: list[str]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"无效 YAML 文件：{path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _run(argv: list[str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(x) for x in argv)
    print(printable)
    if dry_run:
        return
    subprocess.run(argv, check=True)


def _batch_for(defaults: dict[str, Any], exp: dict[str, Any], family: str) -> int:
    if "batch" in exp:
        return int(exp["batch"])
    imgsz = str(exp.get("imgsz", 640))
    key = "batch_fusion" if family in {"e5", "e6"} else "batch_single"
    return int(defaults.get(key, {}).get(imgsz, 16))


def _resolve_train_data(manifest: dict[str, Any], exp: dict[str, Any], work_dir: Path) -> str:
    data = manifest["data"]
    kind = str(exp["kind"])
    modality = str(exp.get("modality", ""))
    if kind == "single":
        base_yaml = data[modality]
    elif kind in {"e5", "e6"}:
        mode = str(exp.get("mode", "rgb_ir"))
        base_yaml = data["rgb_ir"] if mode == "rgb_ir" else data[mode]
    else:
        raise ValueError(f"实验 {exp['id']} 没有训练数据配置：kind={kind}")

    if "train_reweight" not in exp:
        return str(base_yaml)

    reweight = exp["train_reweight"]
    out_yaml = work_dir / "generated_data" / f"{str(exp['id']).lower()}_train_reweighted.yaml"
    _make_reweighted_yaml(
        base_yaml=Path(base_yaml),
        subset_yaml=Path(reweight["subset"]),
        multiplier=int(reweight.get("multiplier", 2)),
        out_yaml=out_yaml,
    )
    return str(out_yaml)


def _iter_images_from_entry(root: Path, entry: Any) -> list[Path]:
    entries = entry if isinstance(entry, list) else [entry]
    images: list[Path] = []
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for item in entries:
        p = Path(str(item))
        if not p.is_absolute():
            p = root / p
        if p.is_dir():
            images.extend(x.resolve() for x in sorted(p.rglob("*")) if x.suffix.lower() in suffixes)
        elif p.is_file() and p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    q = Path(line)
                    images.append(q if q.is_absolute() else (root / q).resolve())
        elif p.is_file() and p.suffix.lower() in suffixes:
            images.append(p.resolve())
    return sorted(set(images))


def _make_reweighted_yaml(base_yaml: Path, subset_yaml: Path, multiplier: int, out_yaml: Path) -> None:
    if multiplier < 2:
        raise ValueError("train_reweight.multiplier 必须大于等于 2")
    base = _load_yaml(base_yaml)
    subset = _load_yaml(subset_yaml)
    base_root = Path(base.get("path", "."))
    subset_root = Path(subset.get("path", "."))

    train_images = _iter_images_from_entry(base_root, base["train"])
    subset_images = _iter_images_from_entry(subset_root, subset.get("train", subset.get("val")))
    subset_stems = {p.stem for p in subset_images}

    weighted = [str(p) for p in train_images]
    boosted = [str(p) for p in train_images if p.stem in subset_stems]
    for _ in range(multiplier - 1):
        weighted.extend(boosted)

    txt_path = out_yaml.with_suffix(".train.txt")
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(weighted) + "\n", encoding="utf-8")

    generated = dict(base)
    generated["path"] = str(base_root)
    generated["train"] = str(txt_path)
    generated["reweight_source"] = {
        "base_yaml": str(base_yaml),
        "subset_yaml": str(subset_yaml),
        "multiplier": multiplier,
        "base_train_images": len(train_images),
        "boosted_images": len(boosted),
        "weighted_train_entries": len(weighted),
    }
    _write_yaml(out_yaml, generated)


def _single_steps(manifest: dict[str, Any], exp: dict[str, Any], work_dir: Path) -> list[CommandStep]:
    defaults = manifest["defaults"]
    data_yaml = _resolve_train_data(manifest, exp, work_dir)
    imgsz = int(exp["imgsz"])
    model_path = str(Path(exp["project"]) / exp["name"] / "weights" / "best.pt")
    val_script = "e1_val_rgb_only.py" if exp["modality"] == "rgb" else "e2_val_ir_only.py"

    train = [
        str(Path(defaults["python"]).with_name("yolo")),
        "detect",
        "train",
        f"model={defaults['yolo_model']}",
        f"data={data_yaml}",
        f"imgsz={imgsz}",
        f"epochs={int(defaults['epochs'])}",
        f"batch={_batch_for(defaults, exp, 'single')}",
        f"workers={int(defaults['workers'])}",
        f"device={defaults['device']}",
        f"project={exp['project']}",
        f"name={exp['name']}",
        f"seed={int(defaults['seed'])}",
        "deterministic=True",
        f"close_mosaic={int(exp.get('close_mosaic', 10))}",
        "exist_ok=True",
    ]
    val = [
        defaults["python"],
        str(Path(defaults["root"]) / "scripts" / val_script),
        "--model",
        model_path,
        "--split",
        defaults.get("split", "val"),
        "--imgsz",
        str(imgsz),
        "--device",
        str(defaults["device"]),
        "--case-topk",
        str(defaults["case"]["topk"]),
        "--case-max-images",
        str(defaults["case"]["max_images"]),
        "--case-batch",
        str(defaults["case"]["batch"]),
        "--out-dir",
        str(exp["result_dir"]),
    ]
    if defaults.get("case", {}).get("skip_viz", True):
        val.append("--skip-case-viz")
    return [CommandStep("train", train), CommandStep("validate", val)]


def _late_fusion_steps(manifest: dict[str, Any], exp: dict[str, Any]) -> list[CommandStep]:
    defaults = manifest["defaults"]
    cmd = [
        defaults["python"],
        str(Path(defaults["root"]) / "scripts" / "e4_val_late_fusion_wbf.py"),
        "--method",
        str(exp.get("method", "late_wbf")),
        "--rgb-model",
        str(exp["rgb_model"]),
        "--ir-model",
        str(exp["ir_model"]),
        "--split",
        defaults.get("split", "val"),
        "--imgsz",
        str(int(exp.get("imgsz", 640))),
        "--device",
        str(defaults["device"]),
        "--batch",
        str(int(exp.get("batch", 16))),
        "--out-dir",
        str(exp["result_dir"]),
    ]
    if str(exp.get("method", "late_wbf")) == "late_wbf":
        cmd.extend(
            [
                "--wbf-weight-rgb",
                str(float(exp.get("wbf_weight_rgb", 1.0))),
                "--wbf-weight-ir",
                str(float(exp.get("wbf_weight_ir", 1.0))),
            ]
        )
    return [CommandStep("fusion_validate", cmd)]


def _fusion_model_steps(manifest: dict[str, Any], exp: dict[str, Any], work_dir: Path) -> list[CommandStep]:
    defaults = manifest["defaults"]
    kind = str(exp["kind"])
    train_script = "e5_train_feature_fusion_single.py" if kind == "e5" else "e6_train_feature_fusion_multiscale.py"
    val_script = "e5_val_feature_fusion_single.py" if kind == "e5" else "e6_val_feature_fusion_multiscale.py"
    data_yaml = _resolve_train_data(manifest, exp, work_dir)
    imgsz = int(exp["imgsz"])
    weights = str(Path(exp["project"]) / exp["name"] / "weights" / "best.pt")

    train = [
        defaults["python"],
        str(Path(defaults["root"]) / "scripts" / train_script),
        "--mode",
        str(exp.get("mode", "rgb_ir")),
        "--model",
        str(defaults["yolo_model"]),
        "--data-rgb-ir",
        data_yaml,
        "--data-ir",
        str(manifest["data"]["ir"]),
        "--epochs",
        str(int(defaults["epochs"])),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(_batch_for(defaults, exp, kind)),
        "--workers",
        str(int(defaults["workers"])),
        "--device",
        str(defaults["device"]),
        "--project",
        str(exp["project"]),
        "--name",
        str(exp["name"]),
        "--seed",
        str(int(defaults["seed"])),
        "--close-mosaic",
        str(int(exp.get("close_mosaic", 10))),
        "--exist-ok",
    ]
    val = [
        defaults["python"],
        str(Path(defaults["root"]) / "scripts" / val_script),
        "--weights",
        weights,
        "--mode",
        str(exp.get("mode", "rgb_ir")),
        "--split",
        defaults.get("split", "val"),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(_batch_for(defaults, exp, kind)),
        "--device",
        str(defaults["device"]),
        "--out-dir",
        str(exp["result_dir"]),
    ]
    return [CommandStep("train", train), CommandStep("validate", val)]


def _steps_for(manifest: dict[str, Any], exp: dict[str, Any], work_dir: Path) -> list[CommandStep]:
    kind = str(exp["kind"])
    if kind == "single":
        return _single_steps(manifest, exp, work_dir)
    if kind == "late_fusion":
        return _late_fusion_steps(manifest, exp)
    if kind in {"e5", "e6"}:
        return _fusion_model_steps(manifest, exp, work_dir)
    if kind == "planned":
        return []
    raise ValueError(f"Unsupported experiment kind for {exp.get('id')}: {kind}")


def _select_experiments(manifest: dict[str, Any], ids: set[str], stage: str | None) -> list[dict[str, Any]]:
    exps = manifest.get("experiments", [])
    selected = []
    for exp in exps:
        if ids and str(exp["id"]) not in ids:
            continue
        if stage and str(exp.get("stage")) != stage:
            continue
        selected.append(exp)
    return selected


def _read_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_metric_bundle(path: Path) -> dict[str, Any] | None:
    """读取 required_metrics，并尽量补充同目录完整摘要中的误报指标。"""
    metrics = _read_metrics(path)
    if metrics is None:
        return None

    summary_path = path.with_name("metrics_summary.json")
    summary = _read_metrics(summary_path) if summary_path.exists() else None
    if isinstance(summary, dict):
        for section in ("standard_metrics", "task_metrics", "required_metrics", "error_metrics"):
            values = summary.get(section)
            if isinstance(values, dict):
                metrics.update(values)
    return metrics


def _metric_path_for(exp: dict[str, Any]) -> Path | None:
    if "metrics" in exp:
        return Path(exp["metrics"])
    if "result_dir" not in exp:
        return None
    return Path(exp["result_dir"]) / "required_metrics.json"


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    if key in metrics:
        try:
            return float(metrics[key])
        except Exception:
            return None
    if key == "False Positives/image":
        for candidate in ("false_positives_per_image", "False Positives / image"):
            if candidate in metrics:
                try:
                    return float(metrics[candidate])
                except Exception:
                    return None
    return None


def _rank_tuple(row: dict[str, Any]) -> tuple[float, float, float, float]:
    ap_dark_small = float(row.get("AP_dark-small") or -1.0)
    recall_small = float(row.get("Recall_small") or -1.0)
    fp = row.get("False Positives/image")
    fp_score = -float(fp) if fp is not None else -1_000_000.0
    map95 = float(row.get("mAP50-95") or -1.0)
    return (ap_dark_small, recall_small, fp_score, map95)


def _experiment_order_tuple(row: dict[str, Any]) -> tuple[int, ...]:
    """按实验编号自然排序：E1, E2, E7_1, E7_2, E8_1。"""
    exp_id = str(row.get("id", ""))
    nums = [int(x) for x in re.findall(r"\d+", exp_id)]
    return tuple(nums) if nums else (10_000,)


def _collect_rows(manifest: dict[str, Any], include_baselines: bool = True) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if include_baselines:
        sources.extend(manifest.get("baseline_results", []))
    sources.extend(manifest.get("experiments", []))

    rows: list[dict[str, Any]] = []
    for src in sources:
        metric_path = _metric_path_for(src)
        metrics = _read_metric_bundle(metric_path) if metric_path else None
        row = {
            "id": src.get("id", ""),
            "name": src.get("title") or src.get("name", ""),
            "stage": src.get("stage", "baseline"),
            "kind": src.get("kind", "baseline"),
            "metrics_path": str(metric_path) if metric_path else "",
            "status": "done" if metrics else src.get("status", "pending"),
        }
        for key in (
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
            "AP_dark-small_object",
            "Recall_dark-small_object",
            "AP_tiny",
            "Recall_tiny",
            "AP_low-contrast",
            "Recall_low-contrast",
            "False Positives/image",
            "FPPI_dark",
            "FPPI_low-contrast",
        ):
            row[key] = _metric_value(metrics, key) if metrics else None
        if metrics and isinstance(metrics.get("efficiency"), dict):
            eff = metrics["efficiency"]
            row["Params"] = _metric_value(eff, "params")
            row["GFLOPs"] = _metric_value(eff, "gflops")
            row["FPS"] = _metric_value(eff, "fps_validator")
            row["GPU_mem_MB"] = _metric_value(eff, "gpu_memory_max_mb")
        if metrics and isinstance(metrics.get("per_class_AP"), dict):
            for cls_name, values in metrics["per_class_AP"].items():
                if isinstance(values, dict):
                    row[f"{cls_name}_AP50-95"] = _metric_value(values, "AP50-95")
                    row[f"{cls_name}_AP50"] = _metric_value(values, "AP50")
        rows.append(row)

    rows.sort(key=_experiment_order_tuple)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "序号",
        "ID",
        "实验",
        "状态",
        "mAP50",
        "mAP50-95",
        "AP_small",
        "Recall_small",
        "AP_dark",
        "Recall_dark",
        "AP_dark-small",
        "AP_tiny",
        "AP_low-contrast",
        "AP_dark-small_obj",
        "FP/image",
        "FPPI_dark",
        "FPPI_low-contrast",
        "Params",
        "GFLOPs",
        "FPS",
        "GPU MB",
    ]
    lines = [
        "# 暗弱小目标实验汇总表",
        "",
        "表格按实验编号顺序排列，便于对应完整实验流程；指标比较时仍优先关注 AP_dark-small、AP_tiny、AP_low-contrast、FP/image、FPPI_dark 和 mAP50-95。",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row["id"]),
                    str(row["name"]),
                    str(row["status"]),
                    _format_float(row.get("mAP50")),
                    _format_float(row.get("mAP50-95")),
                    _format_float(row.get("AP_small")),
                    _format_float(row.get("Recall_small")),
                    _format_float(row.get("AP_dark")),
                    _format_float(row.get("Recall_dark")),
                    _format_float(row.get("AP_dark-small")),
                    _format_float(row.get("AP_tiny")),
                    _format_float(row.get("AP_low-contrast")),
                    _format_float(row.get("AP_dark-small_object")),
                    _format_float(row.get("False Positives/image")),
                    _format_float(row.get("FPPI_dark")),
                    _format_float(row.get("FPPI_low-contrast")),
                    _format_float(row.get("Params")),
                    _format_float(row.get("GFLOPs")),
                    _format_float(row.get("FPS")),
                    _format_float(row.get("GPU_mem_MB")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_list(args: argparse.Namespace) -> None:
    manifest = _load_yaml(Path(args.manifest))
    for exp in _select_experiments(manifest, set(args.ids or []), args.stage):
        print(f"{exp['id']}\t{exp.get('stage', '')}\t{exp['kind']}\t{exp.get('title', exp.get('name', ''))}")


def cmd_run(args: argparse.Namespace) -> None:
    manifest = _load_yaml(Path(args.manifest))
    work_dir = Path(args.work_dir)
    selected = _select_experiments(manifest, set(args.ids or []), args.stage)
    if not selected:
        raise SystemExit("没有选中任何实验。")
    for exp in selected:
        print(f"\n## {exp['id']} {exp.get('title', '')}")
        steps = _steps_for(manifest, exp, work_dir)
        if not steps:
            print(f"跳过：{exp.get('status', 'not executable')} - {exp.get('note', '')}")
            continue
        for step in steps:
            if args.only and step.label not in args.only:
                continue
            print(f"# {step.label}")
            _run(step.argv, dry_run=args.dry_run)


def cmd_aggregate(args: argparse.Namespace) -> None:
    manifest = _load_yaml(Path(args.manifest))
    rows = _collect_rows(manifest, include_baselines=not args.no_baselines)
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "dark_small_experiment_leaderboard.csv", rows)
    _write_md(out_dir / "dark_small_experiment_leaderboard.md", rows)
    print(out_dir / "dark_small_experiment_leaderboard.md")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="列出清单中的实验。")
    p_list.add_argument("ids", nargs="*")
    p_list.add_argument("--stage", default=None)
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="执行或 dry-run 指定实验。")
    p_run.add_argument("ids", nargs="*")
    p_run.add_argument("--stage", default=None)
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--only", nargs="*", choices=["train", "validate", "fusion_validate"])
    p_run.add_argument("--work-dir", default="/mnt/disk2/lhr/VSD/results/val/work")
    p_run.set_defaults(func=cmd_run)

    p_ag = sub.add_parser("aggregate", help="汇总已有指标并写出排行榜。")
    p_ag.add_argument("--out-dir", default="/mnt/disk2/lhr/VSD/results/val")
    p_ag.add_argument("--no-baselines", action="store_true")
    p_ag.set_defaults(func=cmd_aggregate)

    return parser.parse_args(list(argv))


def main() -> None:
    args = parse_args(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
