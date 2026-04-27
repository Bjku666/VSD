#!/usr/bin/env python3
"""Validate E5 model and export standard evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

scripts_dir = os.environ.get("VSD_E5_SCRIPTS_DIR", "")
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from e5_feature_fusion_single_core import E5DetectionTrainer


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_metrics(metrics: Any) -> Dict[str, Any]:
    if isinstance(metrics, dict):
        return {
            "mAP50": _to_float(metrics.get("metrics/mAP50(B)", metrics.get("mAP50", 0.0))),
            "mAP50-95": _to_float(metrics.get("metrics/mAP50-95(B)", metrics.get("mAP50-95", 0.0))),
            "Precision": _to_float(metrics.get("metrics/precision(B)", metrics.get("Precision", 0.0))),
            "Recall": _to_float(metrics.get("metrics/recall(B)", metrics.get("Recall", 0.0))),
            "per_class_AP": {},
        }

    box = metrics.box
    result: Dict[str, Any] = {
        "mAP50": _to_float(getattr(box, "map50", 0.0)),
        "mAP50-95": _to_float(getattr(box, "map", 0.0)),
        "Precision": _to_float(getattr(box, "mp", 0.0)),
        "Recall": _to_float(getattr(box, "mr", 0.0)),
        "per_class_AP": {},
    }

    names = getattr(metrics, "names", None)
    if names is None:
        names = {}

    maps = getattr(box, "maps", None)
    all_ap = getattr(box, "all_ap", None)
    if maps is None:
        return result

    maps_arr = list(maps)
    ap50_arr = None
    if all_ap is not None:
        try:
            ap50_arr = [float(row[0]) for row in all_ap]
        except Exception:
            ap50_arr = None

    for idx, ap in enumerate(maps_arr):
        cls_name = str(names.get(idx, idx))
        cls_info: Dict[str, float] = {"AP50-95": _to_float(ap)}
        if ap50_arr is not None and idx < len(ap50_arr):
            cls_info["AP50"] = _to_float(ap50_arr[idx])
        result["per_class_AP"][cls_name] = cls_info

    return result


def _load_class_names(data_yaml: str) -> Dict[int, str]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names_cfg = cfg.get("names", {}) if isinstance(cfg, dict) else {}
    names: Dict[int, str] = {}
    if isinstance(names_cfg, dict):
        for k, v in names_cfg.items():
            try:
                names[int(k)] = str(v)
            except Exception:
                continue
    elif isinstance(names_cfg, list):
        for idx, v in enumerate(names_cfg):
            names[idx] = str(v)
    return names


def _load_yaml_dict(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid yaml config: {yaml_path}")
    return cfg


def _pick_split_entry(cfg: Dict[str, Any], split: str) -> str:
    if split in cfg:
        return str(cfg[split])
    if "val" in cfg:
        return str(cfg["val"])
    if "train" in cfg:
        return str(cfg["train"])
    raise ValueError(f"No split entry found for split={split}")


def _write_rgb_ir_subset_yaml(
    *,
    base_rgb_ir_yaml: str,
    rgb_subset_yaml: str,
    ir_subset_yaml: str,
    split: str,
    out_yaml: Path,
) -> str:
    base_cfg = _load_yaml_dict(base_rgb_ir_yaml)
    rgb_cfg = _load_yaml_dict(rgb_subset_yaml)
    ir_cfg = _load_yaml_dict(ir_subset_yaml)

    rgb_root = str(rgb_cfg.get("path", base_cfg.get("path", ".")))
    ir_root = str(ir_cfg.get("path", "."))
    rgb_entry = _pick_split_entry(rgb_cfg, split)
    ir_entry = _pick_split_entry(ir_cfg, split)

    merged = dict(base_cfg)
    merged["path"] = rgb_root
    merged["train"] = rgb_entry
    merged[split] = rgb_entry
    if "val" not in merged:
        merged["val"] = rgb_entry
    merged["channels"] = 6

    rgb_node = merged.get("rgb") if isinstance(merged.get("rgb"), dict) else {}
    rgb_node = dict(rgb_node)
    rgb_node["path"] = rgb_root
    rgb_node["train"] = rgb_entry
    rgb_node[split] = rgb_entry
    if "val" not in rgb_node:
        rgb_node["val"] = rgb_entry
    merged["rgb"] = rgb_node

    ir_node = merged.get("ir") if isinstance(merged.get("ir"), dict) else {}
    ir_node = dict(ir_node)
    ir_node["path"] = ir_root
    ir_node["train"] = ir_entry
    ir_node[split] = ir_entry
    if "val" not in ir_node:
        ir_node["val"] = ir_entry
    merged["ir"] = ir_node

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return str(out_yaml)


def _rename_per_class_ap_keys(metrics: Dict[str, Any], class_names: Dict[int, str]) -> Dict[str, Any]:
    per_class = metrics.get("per_class_AP", {})
    if not isinstance(per_class, dict):
        return metrics

    renamed: Dict[str, Any] = {}
    for raw_k, vals in per_class.items():
        try:
            idx = int(raw_k)
        except Exception:
            renamed[str(raw_k)] = vals
            continue
        renamed[class_names.get(idx, str(idx))] = vals

    out = dict(metrics)
    out["per_class_AP"] = renamed
    return out


def _evaluate_once(
    *,
    weights: str,
    mode: str,
    data_yaml: str,
    data_ir_yaml: str,
    split: str,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    project: Path,
    name: str,
    plots: bool,
    exist_ok: bool,
) -> Dict[str, Any]:
    eval_data_yaml = str(Path(data_yaml))
    temp_yaml_path: str | None = None
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and "train" not in cfg and "val" in cfg:
        cfg = dict(cfg)
        cfg["train"] = cfg["val"]
        fd, temp_yaml_path = tempfile.mkstemp(prefix="e5_eval_subset_", suffix=".yaml")
        os.close(fd)
        with open(temp_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        eval_data_yaml = temp_yaml_path

    overrides = {
        "task": "detect",
        "mode": "train",
        "model": str(Path(weights)),
        "data": eval_data_yaml,
        "epochs": 1,
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "device": device,
        "project": str(project),
        "name": name,
        "resume": False,
        "plots": bool(plots),
        "exist_ok": bool(exist_ok),
    }

    try:
        trainer = E5DetectionTrainer(overrides=overrides)
        trainer.set_fusion_mode(mode)
        trainer.set_ir_data(str(Path(data_ir_yaml)) if mode in {"ir", "rgb_ir"} else None)
        trainer.setup_model()
        trainer.model = trainer.model.to(trainer.device).float().eval()

        split_key = split if split in trainer.data else "val"
        trainer.test_loader = trainer.get_dataloader(trainer.data[split_key], batch_size=int(batch), rank=-1, mode="val")
        trainer.validator = trainer.get_validator()
        metrics = trainer.validator(model=trainer.model)
        metrics_source = getattr(trainer.validator, "metrics", None)
        return _extract_metrics(metrics_source if metrics_source is not None else metrics)
    finally:
        if temp_yaml_path is not None:
            Path(temp_yaml_path).unlink(missing_ok=True)


def _remove_eval_weights_dirs(project_dir: Path) -> None:
    if not project_dir.exists():
        return
    for run_dir in project_dir.iterdir():
        if not run_dir.is_dir():
            continue
        weights_dir = run_dir / "weights"
        if weights_dir.exists() and weights_dir.is_dir():
            shutil.rmtree(weights_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO11n E5 single-layer fusion model")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--mode", type=str, default="rgb_ir", choices=["rgb", "ir", "rgb_ir"])
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument(
        "--data-rgb",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb.yaml",
    )
    parser.add_argument(
        "--data-rgb-ir",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml",
    )
    parser.add_argument(
        "--data-ir",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml",
    )
    parser.add_argument(
        "--data-small-rgb",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/rgb_small.yaml",
    )
    parser.add_argument(
        "--data-dark-rgb",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/rgb_dark.yaml",
    )
    parser.add_argument(
        "--data-dark-small-rgb",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/rgb_dark-small.yaml",
    )
    parser.add_argument(
        "--data-small-ir",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/ir_small.yaml",
    )
    parser.add_argument(
        "--data-dark-ir",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/ir_dark.yaml",
    )
    parser.add_argument(
        "--data-dark-small-ir",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/ir_dark-small.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/disk2/lhr/VSD/results/val/e5_feature_fusion_single_val",
    )
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    ultra_project = out_dir / "ultralytics_val"
    ultra_name_full = f"{args.split}_full"
    ultra_name_small = f"{args.split}_small"
    ultra_name_dark = f"{args.split}_dark"
    ultra_name_dark_small = f"{args.split}_dark-small"

    if args.mode == "rgb":
        data_yaml = args.data_rgb
        data_small = args.data_small_rgb
        data_dark = args.data_dark_rgb
        data_dark_small = args.data_dark_small_rgb
    elif args.mode == "ir":
        data_yaml = args.data_ir
        data_small = args.data_small_ir
        data_dark = args.data_dark_ir
        data_dark_small = args.data_dark_small_ir
    else:
        data_yaml = args.data_rgb_ir
        generated_dir = out_dir / "_generated_eval_data"
        data_small = _write_rgb_ir_subset_yaml(
            base_rgb_ir_yaml=args.data_rgb_ir,
            rgb_subset_yaml=args.data_small_rgb,
            ir_subset_yaml=args.data_small_ir,
            split=args.split,
            out_yaml=generated_dir / f"{args.split}_small_rgb_ir.yaml",
        )
        data_dark = _write_rgb_ir_subset_yaml(
            base_rgb_ir_yaml=args.data_rgb_ir,
            rgb_subset_yaml=args.data_dark_rgb,
            ir_subset_yaml=args.data_dark_ir,
            split=args.split,
            out_yaml=generated_dir / f"{args.split}_dark_rgb_ir.yaml",
        )
        data_dark_small = _write_rgb_ir_subset_yaml(
            base_rgb_ir_yaml=args.data_rgb_ir,
            rgb_subset_yaml=args.data_dark_small_rgb,
            ir_subset_yaml=args.data_dark_small_ir,
            split=args.split,
            out_yaml=generated_dir / f"{args.split}_dark-small_rgb_ir.yaml",
        )

    standard = _evaluate_once(
        weights=args.weights,
        mode=args.mode,
        data_yaml=data_yaml,
        data_ir_yaml=args.data_ir,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=ultra_project,
        name=ultra_name_full,
        plots=True,
        exist_ok=bool(args.exist_ok),
    )
    small_metrics = _evaluate_once(
        weights=args.weights,
        mode=args.mode,
        data_yaml=data_small,
        data_ir_yaml=args.data_ir,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=ultra_project,
        name=ultra_name_small,
        plots=False,
        exist_ok=True,
    )
    dark_metrics = _evaluate_once(
        weights=args.weights,
        mode=args.mode,
        data_yaml=data_dark,
        data_ir_yaml=args.data_ir,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=ultra_project,
        name=ultra_name_dark,
        plots=False,
        exist_ok=True,
    )
    dark_small_metrics = _evaluate_once(
        weights=args.weights,
        mode=args.mode,
        data_yaml=data_dark_small,
        data_ir_yaml=args.data_ir,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=ultra_project,
        name=ultra_name_dark_small,
        plots=False,
        exist_ok=True,
    )

    class_names = _load_class_names(data_yaml)
    standard = _rename_per_class_ap_keys(standard, class_names)
    required = {
        "mAP50": standard["mAP50"],
        "mAP50-95": standard["mAP50-95"],
        "Precision": standard["Precision"],
        "Recall": standard["Recall"],
        "per_class_AP": standard["per_class_AP"],
        "AP_small": small_metrics["mAP50-95"],
        "Recall_small": small_metrics["Recall"],
        "AP_dark": dark_metrics["mAP50-95"],
        "Recall_dark": dark_metrics["Recall"],
        "AP_dark-small": dark_small_metrics["mAP50-95"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary = {
        "model": str(Path(args.weights)),
        "mode": args.mode,
        "imgsz": int(args.imgsz),
        "split": args.split,
        "device": args.device,
        "data": {
            "full": str(Path(data_yaml)),
            "small": str(Path(data_small)),
            "dark": str(Path(data_dark)),
            "dark_small": str(Path(data_dark_small)),
        },
        "standard_metrics": standard,
        "custom_metrics": {
            "AP_small": required["AP_small"],
            "Recall_small": required["Recall_small"],
            "AP_dark": required["AP_dark"],
            "Recall_dark": required["Recall_dark"],
            "AP_dark-small": required["AP_dark-small"],
        },
        "required_metrics": required,
        "ultralytics_project_dir": str(ultra_project),
        "ultralytics_run_name": ultra_name_full,
        "ultralytics_run_names": {
            "full": ultra_name_full,
            "small": ultra_name_small,
            "dark": ultra_name_dark,
            "dark_small": ultra_name_dark_small,
        },
    }

    metrics_json = out_dir / "metrics_summary.json"
    required_json = out_dir / "required_metrics.json"
    required_csv = out_dir / "required_metrics.csv"
    metrics_md = out_dir / "metrics_summary.md"

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
    with open(required_json, "w", encoding="utf-8") as f:
        json.dump(required, f, ensure_ascii=False, indent=2)

    with open(required_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["mAP50", f"{required['mAP50']:.6f}"])
        writer.writerow(["mAP50-95", f"{required['mAP50-95']:.6f}"])
        writer.writerow(["Precision", f"{required['Precision']:.6f}"])
        writer.writerow(["Recall", f"{required['Recall']:.6f}"])
        writer.writerow(["AP_small", f"{required['AP_small']:.6f}"])
        writer.writerow(["Recall_small", f"{required['Recall_small']:.6f}"])
        writer.writerow(["AP_dark", f"{required['AP_dark']:.6f}"])
        writer.writerow(["Recall_dark", f"{required['Recall_dark']:.6f}"])
        writer.writerow(["AP_dark-small", f"{required['AP_dark-small']:.6f}"])
        writer.writerow([])
        writer.writerow(["class", "AP50", "AP50-95"])
        for cls_name, vals in required["per_class_AP"].items():
            writer.writerow([
                cls_name,
                f"{_to_float(vals.get('AP50', 0.0)):.6f}",
                f"{_to_float(vals.get('AP50-95', 0.0)):.6f}",
            ])

    md_lines = [
        "# E5 Validation Summary",
        "",
        f"- mAP50: {required['mAP50']:.6f}",
        f"- mAP50-95: {required['mAP50-95']:.6f}",
        f"- Precision: {required['Precision']:.6f}",
        f"- Recall: {required['Recall']:.6f}",
        "",
        f"- AP_small: {required['AP_small']:.6f}",
        f"- Recall_small: {required['Recall_small']:.6f}",
        f"- AP_dark: {required['AP_dark']:.6f}",
        f"- Recall_dark: {required['Recall_dark']:.6f}",
        f"- AP_dark-small: {required['AP_dark-small']:.6f}",
        "",
    ]
    with open(metrics_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    run_dir = ultra_project / ultra_name_full
    cm = run_dir / "confusion_matrix.png"
    cmn = run_dir / "confusion_matrix_normalized.png"
    if cm.exists():
        shutil.copy2(cm, out_dir / "confusion_matrix.png")
    if cmn.exists():
        shutil.copy2(cmn, out_dir / "confusion_matrix_normalized.png")
    elif cm.exists():
        shutil.copy2(cm, out_dir / "confusion_matrix_normalized.png")

    metrics_summary["required_metric_files"] = {
        "json": str(required_json),
        "csv": str(required_csv),
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    # Keep evaluation output clean: remove auto-created, unused empty weights dirs.
    _remove_eval_weights_dirs(ultra_project)

    print(f"Saved: {metrics_json}")
    print(f"Saved: {required_json}")
    print(f"Saved: {required_csv}")
    print(f"Saved: {metrics_md}")


if __name__ == "__main__":
    main()
