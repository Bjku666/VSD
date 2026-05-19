#!/usr/bin/env python3
"""Validate E6 model and export standard evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from ultralytics.utils.torch_utils import get_flops

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "")
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from e6_feature_fusion_multiscale_core import E6DetectionTrainer


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


def _extract_confusion_error_metrics(validator: Any) -> Dict[str, Any]:
    cm = getattr(validator, "confusion_matrix", None)
    matrix = getattr(cm, "matrix", None)
    nc = int(getattr(cm, "nc", 0) or 0)
    n_images = int(getattr(validator, "seen", 0) or 0)
    if matrix is None or nc <= 0:
        return {"image_count": n_images}

    bg_fp = _to_float(matrix[:nc, nc].sum())
    bg_fn = _to_float(matrix[nc, :nc].sum())
    diag = _to_float(matrix[:nc, :nc].diagonal().sum())
    offdiag = _to_float(matrix[:nc, :nc].sum()) - diag
    gt_total = _to_float(matrix[:, :nc].sum())
    return {
        "image_count": n_images,
        "false_positives": bg_fp,
        "false_positives_per_image": bg_fp / n_images if n_images > 0 else 0.0,
        "background_false_positive": bg_fp,
        "background_false_negative": bg_fn,
        "confusion_offdiag": offdiag,
        "background_confusion_rate": (bg_fp + bg_fn) / gt_total if gt_total > 0 else 0.0,
    }


def _extract_speed_metrics(validator: Any, elapsed_s: float, image_count: int) -> Dict[str, Any]:
    speed = getattr(validator, "speed", {}) or {}
    preprocess = _to_float(speed.get("preprocess", 0.0))
    inference = _to_float(speed.get("inference", 0.0))
    loss = _to_float(speed.get("loss", 0.0))
    postprocess = _to_float(speed.get("postprocess", 0.0))
    total_ms = preprocess + inference + loss + postprocess
    return {
        "speed_ms_per_image": {
            "preprocess": preprocess,
            "inference": inference,
            "loss": loss,
            "postprocess": postprocess,
            "total": total_ms,
        },
        "fps_validator": 1000.0 / total_ms if total_ms > 0 else 0.0,
        "fps_wall": image_count / elapsed_s if elapsed_s > 0 and image_count > 0 else 0.0,
        "elapsed_seconds": elapsed_s,
    }


def _profile_model(model: Any, imgsz: int) -> Dict[str, Any]:
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        flops = float(get_flops(model, imgsz=imgsz))
    except Exception:
        flops = 0.0
    if flops <= 0.0:
        try:
            from copy import deepcopy
            import thop

            p = next(model.parameters())
            ch = 6 if getattr(model, "fusion_mode", "") == "rgb_ir" else 3
            im = torch.empty((1, ch, int(imgsz), int(imgsz)), device=p.device)
            flops = float(thop.profile(deepcopy(model), inputs=[im], verbose=False)[0]) / 1e9 * 2.0
        except Exception:
            flops = 0.0
    return {
        "params": int(params),
        "trainable_params": int(trainable),
        "gflops": flops,
    }


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
    include_efficiency: bool = False,
) -> Dict[str, Any]:
    eval_data_yaml = str(Path(data_yaml))
    temp_yaml_path: str | None = None
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and "train" not in cfg and "val" in cfg:
        cfg = dict(cfg)
        cfg["train"] = cfg["val"]
        fd, temp_yaml_path = tempfile.mkstemp(prefix="e6_eval_subset_", suffix=".yaml")
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
        # Ultralytics updates confusion_matrix only when plots=True. Keep it enabled
        # for all subset validations so FP/image and FPPI_* use the same matcher.
        "plots": True,
        "exist_ok": bool(exist_ok),
    }

    try:
        trainer = E6DetectionTrainer(overrides=overrides)
        trainer.set_fusion_mode(mode)
        trainer.set_ir_data(str(Path(data_ir_yaml)) if mode in {"ir", "rgb_ir"} else None)
        trainer.setup_model()
        trainer.model = trainer.model.to(trainer.device).float().eval()

        profile = _profile_model(trainer.model, int(imgsz)) if include_efficiency else {}
        if torch.cuda.is_available() and str(device).lower() != "cpu":
            torch.cuda.reset_peak_memory_stats(trainer.device)

        split_key = split if split in trainer.data else "val"
        trainer.test_loader = trainer.get_dataloader(trainer.data[split_key], batch_size=int(batch), rank=-1, mode="val")
        trainer.validator = trainer.get_validator()
        started = time.perf_counter()
        metrics = trainer.validator(model=trainer.model)
        elapsed_s = time.perf_counter() - started
        metrics_source = getattr(trainer.validator, "metrics", None)
        out = _extract_metrics(metrics_source if metrics_source is not None else metrics)
        error_metrics = _extract_confusion_error_metrics(trainer.validator)
        out["error_metrics"] = error_metrics
        if include_efficiency:
            image_count = int(error_metrics.get("image_count", 0) or getattr(trainer.validator, "seen", 0) or 0)
            max_mem_mb = 0.0
            if torch.cuda.is_available() and str(device).lower() != "cpu":
                max_mem_mb = torch.cuda.max_memory_allocated(trainer.device) / (1024.0 ** 2)
            out["efficiency"] = {
                **profile,
                **_extract_speed_metrics(trainer.validator, elapsed_s, image_count),
                "gpu_memory_max_mb": max_mem_mb,
            }
        return out
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
    parser = argparse.ArgumentParser(description="Validate YOLO11n E6 multi-scale fusion model")
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
        "--data-tiny-rgb",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/rgb_tiny.yaml",
    )
    parser.add_argument(
        "--data-low-contrast-rgb",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/rgb_low-contrast.yaml",
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
    parser.add_argument(
        "--data-tiny-ir",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/ir_tiny.yaml",
    )
    parser.add_argument(
        "--data-low-contrast-ir",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/ir_low-contrast.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/disk2/lhr/VSD/results/val/e6_feature_fusion_multiscale_val",
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
    ultra_name_tiny = f"{args.split}_tiny"
    ultra_name_low_contrast = f"{args.split}_low-contrast"

    if args.mode == "rgb":
        data_yaml = args.data_rgb
        data_small = args.data_small_rgb
        data_dark = args.data_dark_rgb
        data_dark_small = args.data_dark_small_rgb
        data_tiny = args.data_tiny_rgb
        data_low_contrast = args.data_low_contrast_rgb
    elif args.mode == "ir":
        data_yaml = args.data_ir
        data_small = args.data_small_ir
        data_dark = args.data_dark_ir
        data_dark_small = args.data_dark_small_ir
        data_tiny = args.data_tiny_ir
        data_low_contrast = args.data_low_contrast_ir
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
        data_tiny = _write_rgb_ir_subset_yaml(
            base_rgb_ir_yaml=args.data_rgb_ir,
            rgb_subset_yaml=args.data_tiny_rgb,
            ir_subset_yaml=args.data_tiny_ir,
            split=args.split,
            out_yaml=generated_dir / f"{args.split}_tiny_rgb_ir.yaml",
        )
        data_low_contrast = _write_rgb_ir_subset_yaml(
            base_rgb_ir_yaml=args.data_rgb_ir,
            rgb_subset_yaml=args.data_low_contrast_rgb,
            ir_subset_yaml=args.data_low_contrast_ir,
            split=args.split,
            out_yaml=generated_dir / f"{args.split}_low-contrast_rgb_ir.yaml",
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
        include_efficiency=True,
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
    tiny_metrics = _evaluate_once(
        weights=args.weights,
        mode=args.mode,
        data_yaml=data_tiny,
        data_ir_yaml=args.data_ir,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=ultra_project,
        name=ultra_name_tiny,
        plots=False,
        exist_ok=True,
    )
    low_contrast_metrics = _evaluate_once(
        weights=args.weights,
        mode=args.mode,
        data_yaml=data_low_contrast,
        data_ir_yaml=args.data_ir,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=ultra_project,
        name=ultra_name_low_contrast,
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
        "Recall_dark-small": dark_small_metrics["Recall"],
        "AP_tiny": tiny_metrics["mAP50-95"],
        "Recall_tiny": tiny_metrics["Recall"],
        "AP_low-contrast": low_contrast_metrics["mAP50-95"],
        "Recall_low-contrast": low_contrast_metrics["Recall"],
        "False Positives/image": standard.get("error_metrics", {}).get("false_positives_per_image", 0.0),
        "FPPI_dark": dark_metrics.get("error_metrics", {}).get("false_positives_per_image", 0.0),
        "FPPI_low-contrast": low_contrast_metrics.get("error_metrics", {}).get("false_positives_per_image", 0.0),
        "efficiency": standard.get("efficiency", {}),
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
            "tiny": str(Path(data_tiny)),
            "low_contrast": str(Path(data_low_contrast)),
        },
        "standard_metrics": standard,
        "custom_metrics": {
            "AP_small": required["AP_small"],
            "Recall_small": required["Recall_small"],
            "AP_dark": required["AP_dark"],
            "Recall_dark": required["Recall_dark"],
            "AP_dark-small": required["AP_dark-small"],
            "Recall_dark-small": required["Recall_dark-small"],
            "AP_tiny": required["AP_tiny"],
            "Recall_tiny": required["Recall_tiny"],
            "AP_low-contrast": required["AP_low-contrast"],
            "Recall_low-contrast": required["Recall_low-contrast"],
        },
        "error_metrics": {
            "full": standard.get("error_metrics", {}),
            "dark": dark_metrics.get("error_metrics", {}),
            "dark_small": dark_small_metrics.get("error_metrics", {}),
            "tiny": tiny_metrics.get("error_metrics", {}),
            "low_contrast": low_contrast_metrics.get("error_metrics", {}),
        },
        "efficiency": required.get("efficiency", {}),
        "required_metrics": required,
        "ultralytics_project_dir": str(ultra_project),
        "ultralytics_run_name": ultra_name_full,
        "ultralytics_run_names": {
            "full": ultra_name_full,
            "small": ultra_name_small,
            "dark": ultra_name_dark,
            "dark_small": ultra_name_dark_small,
            "tiny": ultra_name_tiny,
            "low_contrast": ultra_name_low_contrast,
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
        writer.writerow(["Recall_dark-small", f"{required['Recall_dark-small']:.6f}"])
        writer.writerow(["AP_tiny", f"{required['AP_tiny']:.6f}"])
        writer.writerow(["Recall_tiny", f"{required['Recall_tiny']:.6f}"])
        writer.writerow(["AP_low-contrast", f"{required['AP_low-contrast']:.6f}"])
        writer.writerow(["Recall_low-contrast", f"{required['Recall_low-contrast']:.6f}"])
        writer.writerow(["False Positives/image", f"{required['False Positives/image']:.6f}"])
        writer.writerow(["FPPI_dark", f"{required['FPPI_dark']:.6f}"])
        writer.writerow(["FPPI_low-contrast", f"{required['FPPI_low-contrast']:.6f}"])
        for metric_name, metric_value in required.get("efficiency", {}).items():
            if isinstance(metric_value, dict):
                for sub_name, sub_value in metric_value.items():
                    writer.writerow([f"efficiency/{metric_name}/{sub_name}", f"{_to_float(sub_value):.6f}"])
            else:
                writer.writerow([f"efficiency/{metric_name}", f"{_to_float(metric_value):.6f}"])
        writer.writerow([])
        writer.writerow(["class", "AP50", "AP50-95"])
        for cls_name, vals in required["per_class_AP"].items():
            writer.writerow([
                cls_name,
                f"{_to_float(vals.get('AP50', 0.0)):.6f}",
                f"{_to_float(vals.get('AP50-95', 0.0)):.6f}",
            ])

    md_lines = [
        "# E6 Validation Summary",
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
        f"- Recall_dark-small: {required['Recall_dark-small']:.6f}",
        f"- AP_tiny: {required['AP_tiny']:.6f}",
        f"- Recall_tiny: {required['Recall_tiny']:.6f}",
        f"- AP_low-contrast: {required['AP_low-contrast']:.6f}",
        f"- Recall_low-contrast: {required['Recall_low-contrast']:.6f}",
        f"- False Positives/image: {required['False Positives/image']:.6f}",
        f"- FPPI_dark: {required['FPPI_dark']:.6f}",
        f"- FPPI_low-contrast: {required['FPPI_low-contrast']:.6f}",
        "",
        f"- Params: {int(required.get('efficiency', {}).get('params', 0))}",
        f"- GFLOPs: {required.get('efficiency', {}).get('gflops', 0.0):.6f}",
        f"- FPS_validator: {required.get('efficiency', {}).get('fps_validator', 0.0):.6f}",
        f"- GPU memory max MB: {required.get('efficiency', {}).get('gpu_memory_max_mb', 0.0):.6f}",
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

    _remove_eval_weights_dirs(ultra_project)

    print(f"Saved: {metrics_json}")
    print(f"Saved: {required_json}")
    print(f"Saved: {required_csv}")
    print(f"Saved: {metrics_md}")


if __name__ == "__main__":
    main()
