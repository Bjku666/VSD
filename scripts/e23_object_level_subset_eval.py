#!/usr/bin/env python3
"""E23: build object-level subsets and evaluate object-scoped metrics.

Image-level subsets keep all labels from selected images. Object-level subsets
keep the same images only when at least one target object qualifies, then filter
labels down to qualifying objects. This prevents AP_dark-small from silently
measuring non-dark-small objects in dark-small images.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "") or str(Path(__file__).resolve().parent)
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import e6_val_feature_fusion_multiscale as e6val
from e13_val_tiny_aware_loss import _evaluate_once as e13_evaluate_once
from e14_val_cebs import _ConfiguredE14CEBSTrainer


e6_evaluate_once = e6val._evaluate_once


CLASS_NAMES = ["car", "truck", "bus", "van", "freight_car"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SCOPES = ("dark-small", "tiny", "low-contrast")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON: {path}")
    return data


def _split_dir(data_yaml: Path, split: str, modality: str | None = None) -> Path:
    cfg = _load_yaml(data_yaml)
    if modality and isinstance(cfg.get(modality), dict):
        node = cfg[modality]
        root = Path(str(node.get("path", cfg.get("path", "."))))
        entry = Path(str(node[split]))
    else:
        root = Path(str(cfg.get("path", ".")))
        entry = Path(str(cfg[split]))
    return entry if entry.is_absolute() else root / entry


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            rows.append((int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    return rows


def _xywhn_to_xyxy(
    row: tuple[int, float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    _cls, x, y, w, h = row
    return (
        (x - w / 2.0) * width,
        (y - h / 2.0) * height,
        (x + w / 2.0) * width,
        (y + h / 2.0) * height,
    )


def _box_contrast(gray: np.ndarray, xyxy: tuple[float, float, float, float], ring_scale: float) -> float | None:
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1_i = int(max(0, min(width - 1, np.floor(x1))))
    y1_i = int(max(0, min(height - 1, np.floor(y1))))
    x2_i = int(max(x1_i + 1, min(width, np.ceil(x2))))
    y2_i = int(max(y1_i + 1, min(height, np.ceil(y2))))
    inner = gray[y1_i:y2_i, x1_i:x2_i]
    if inner.size == 0:
        return None

    box_w = max(1, x2_i - x1_i)
    box_h = max(1, y2_i - y1_i)
    exp_w = max(box_w + 2, int(round(box_w * ring_scale)))
    exp_h = max(box_h + 2, int(round(box_h * ring_scale)))
    cx = (x1_i + x2_i) / 2.0
    cy = (y1_i + y2_i) / 2.0
    ex1 = int(max(0, round(cx - exp_w / 2.0)))
    ey1 = int(max(0, round(cy - exp_h / 2.0)))
    ex2 = int(min(width, round(cx + exp_w / 2.0)))
    ey2 = int(min(height, round(cy + exp_h / 2.0)))
    expanded = gray[ey1:ey2, ex1:ex2]
    if expanded.size == 0:
        return None

    mask = np.ones(expanded.shape[:2], dtype=bool)
    rx1 = int(max(0, min(expanded.shape[1], x1_i - ex1)))
    ry1 = int(max(0, min(expanded.shape[0], y1_i - ey1)))
    rx2 = int(max(rx1 + 1, min(expanded.shape[1], x2_i - ex1)))
    ry2 = int(max(ry1 + 1, min(expanded.shape[0], y2_i - ey1)))
    mask[ry1:ry2, rx1:rx2] = False
    ring = expanded[mask]
    if ring.size == 0:
        return None
    return abs(float(inner.mean()) - float(ring.mean())) / (float(ring.mean()) + 1e-6)


def _qualifies(
    *,
    scope: str,
    brightness: float,
    area: float,
    contrast: float | None,
    small_thr: float,
    tiny_thr: float,
    dark_thr: float,
    low_contrast_thr: float,
) -> bool:
    if scope == "dark-small":
        return brightness <= dark_thr and area <= small_thr
    if scope == "tiny":
        return area <= tiny_thr
    if scope == "low-contrast":
        return contrast is not None and contrast <= low_contrast_thr
    raise ValueError(f"Unsupported object scope: {scope}")


def _safe_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _label_line(row: tuple[int, float, float, float, float]) -> str:
    cls, x, y, w, h = row
    return f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def _write_dataset_yaml(path: Path, root: Path) -> None:
    data = {
        "path": str(root),
        "train": "images/val",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    _write_yaml(path, data)


def _write_rgb_ir_yaml(path: Path, rgb_root: Path, ir_root: Path) -> None:
    data = {
        "path": str(rgb_root),
        "train": "images/val",
        "val": "images/val",
        "rgb": {"path": str(rgb_root), "train": "images/val", "val": "images/val"},
        "ir": {"path": str(ir_root), "train": "images/val", "val": "images/val"},
        "channels": 6,
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    _write_yaml(path, data)


def build_object_subsets(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = _read_json(Path(args.thresholds))
    dark_thr = float(thresholds["brightness_dark_threshold"])
    small_thr = float(thresholds["object_area_small_threshold"])
    tiny_thr = float(thresholds["object_area_tiny_threshold"])
    low_contrast_thr = float(thresholds["low_contrast_threshold"])
    ring_scale = float(thresholds.get("contrast_ring_scale", 1.6))

    rgb_dir = _split_dir(Path(args.data_rgb), args.split)
    ir_dir = _split_dir(Path(args.data_ir), args.split)
    rgb_images = {p.stem: p for p in sorted(rgb_dir.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES}
    ir_images = {p.stem: p for p in sorted(ir_dir.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES}
    stems = sorted(set(rgb_images) & set(ir_images))

    subset_root = Path(args.out_dir) / "object_level_subsets"
    config_dir = Path(args.out_dir) / "_generated_object_level_configs"
    shutil.rmtree(subset_root, ignore_errors=True)
    shutil.rmtree(config_dir, ignore_errors=True)

    summary: dict[str, Any] = {
        "experiment": "E23",
        "split": args.split,
        "selection_basis": "RGB image brightness/contrast and RGB labels; labels are mirrored to IR for paired evaluation",
        "thresholds": {
            "brightness_dark_threshold": dark_thr,
            "object_area_small_threshold": small_thr,
            "object_area_tiny_threshold": tiny_thr,
            "low_contrast_threshold": low_contrast_thr,
            "contrast_ring_scale": ring_scale,
        },
        "scopes": {},
        "configs": {},
    }

    for scope in SCOPES:
        rgb_root = subset_root / f"{scope}_object" / "rgb"
        ir_root = subset_root / f"{scope}_object" / "ir"
        written_images = 0
        written_objects = 0
        class_counts = {name: 0 for name in CLASS_NAMES}

        for stem in stems:
            rgb_image = rgb_images[stem]
            ir_image = ir_images[stem]
            labels = _read_yolo_labels(_label_path_for_image(rgb_image))
            if not labels:
                continue
            with Image.open(rgb_image) as im:
                gray = np.asarray(im.convert("L"), dtype=np.float32)
                width, height = im.size
            brightness = float(gray.mean()) if gray.size else 0.0

            kept: list[tuple[int, float, float, float, float]] = []
            for row in labels:
                xyxy = _xywhn_to_xyxy(row, width, height)
                area = max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])
                contrast = _box_contrast(gray, xyxy, ring_scale)
                if _qualifies(
                    scope=scope,
                    brightness=brightness,
                    area=area,
                    contrast=contrast,
                    small_thr=small_thr,
                    tiny_thr=tiny_thr,
                    dark_thr=dark_thr,
                    low_contrast_thr=low_contrast_thr,
                ):
                    kept.append(row)

            if not kept:
                continue

            rgb_img_dst = rgb_root / "images" / args.split / rgb_image.name
            ir_img_dst = ir_root / "images" / args.split / ir_image.name
            rgb_label_dst = rgb_root / "labels" / args.split / f"{stem}.txt"
            ir_label_dst = ir_root / "labels" / args.split / f"{stem}.txt"
            _safe_link(rgb_image, rgb_img_dst)
            _safe_link(ir_image, ir_img_dst)
            label_text = "\n".join(_label_line(row) for row in kept) + "\n"
            rgb_label_dst.parent.mkdir(parents=True, exist_ok=True)
            ir_label_dst.parent.mkdir(parents=True, exist_ok=True)
            rgb_label_dst.write_text(label_text, encoding="utf-8")
            ir_label_dst.write_text(label_text, encoding="utf-8")

            written_images += 1
            written_objects += len(kept)
            for cls, *_rest in kept:
                if 0 <= cls < len(CLASS_NAMES):
                    class_counts[CLASS_NAMES[cls]] += 1

        rgb_yaml = config_dir / f"rgb_{scope}_object.yaml"
        ir_yaml = config_dir / f"ir_{scope}_object.yaml"
        rgb_ir_yaml = config_dir / f"rgb_ir_{scope}_object.yaml"
        _write_dataset_yaml(rgb_yaml, rgb_root)
        _write_dataset_yaml(ir_yaml, ir_root)
        _write_rgb_ir_yaml(rgb_ir_yaml, rgb_root, ir_root)
        summary["scopes"][scope] = {
            "images": written_images,
            "objects": written_objects,
            "class_counts": class_counts,
        }
        summary["configs"][scope] = {
            "rgb": str(rgb_yaml),
            "ir": str(ir_yaml),
            "rgb_ir": str(rgb_ir_yaml),
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "object_level_subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _metric(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics[key])
    except Exception:
        return 0.0


def evaluate_object_scopes(args: argparse.Namespace, subset_summary: dict[str, Any]) -> dict[str, Any]:
    if args.validator == "e13":
        evaluator = e13_evaluate_once
    elif args.validator == "e14":
        _ConfiguredE14CEBSTrainer.cebs_args = args
        e6val.E6DetectionTrainer = _ConfiguredE14CEBSTrainer
        evaluator = e6val._evaluate_once
    else:
        evaluator = e6_evaluate_once
    out_dir = Path(args.out_dir)
    project = out_dir / "ultralytics_val"
    object_metrics: dict[str, Any] = {}
    required: dict[str, Any] = {}

    for scope in SCOPES:
        cfgs = subset_summary["configs"][scope]
        if args.mode == "rgb":
            data_yaml = cfgs["rgb"]
        elif args.mode == "ir":
            data_yaml = cfgs["ir"]
        else:
            data_yaml = cfgs["rgb_ir"]
        metrics = evaluator(
            weights=args.weights,
            mode=args.mode,
            data_yaml=data_yaml,
            data_ir_yaml=cfgs["ir"],
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=project,
            name=f"{args.split}_{scope}_object",
            plots=False,
            exist_ok=True,
        )
        object_metrics[scope] = metrics
        metric_prefix = scope.replace("-", "_")
        required[f"AP_{scope}_object"] = _metric(metrics, "mAP50-95")
        required[f"Recall_{scope}_object"] = _metric(metrics, "Recall")
        required[f"FPPI_{scope}_object"] = float(
            metrics.get("error_metrics", {}).get("false_positives_per_image", 0.0)
        )
        required[f"images_{metric_prefix}_object"] = int(subset_summary["scopes"][scope]["images"])
        required[f"objects_{metric_prefix}_object"] = int(subset_summary["scopes"][scope]["objects"])

    return {"object_metrics": object_metrics, "required_metrics": required}


def _load_image_metrics(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return _read_json(p)


def write_reports(args: argparse.Namespace, subset_summary: dict[str, Any], eval_summary: dict[str, Any]) -> None:
    out_dir = Path(args.out_dir)
    required = eval_summary.get("required_metrics", {})
    image_metrics = _load_image_metrics(args.image_metrics)

    comparison_rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        image_key = f"AP_{scope}"
        image_recall_key = f"Recall_{scope}"
        object_key = f"AP_{scope}_object"
        object_recall_key = f"Recall_{scope}_object"
        row = {
            "scope": scope,
            "AP_image": image_metrics.get(image_key, ""),
            "Recall_image": image_metrics.get(image_recall_key, ""),
            "AP_object": required.get(object_key, ""),
            "Recall_object": required.get(object_recall_key, ""),
            "image_count_object_subset": subset_summary["scopes"][scope]["images"],
            "object_count_object_subset": subset_summary["scopes"][scope]["objects"],
        }
        try:
            row["AP_object_minus_image"] = float(row["AP_object"]) - float(row["AP_image"])
        except Exception:
            row["AP_object_minus_image"] = ""
        comparison_rows.append(row)

    metrics_summary = {
        "experiment": "E23",
        "weights": args.weights,
        "validator": args.validator,
        "mode": args.mode,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "split": args.split,
        "image_metrics_source": args.image_metrics,
        "object_subset_summary": subset_summary,
        "object_metrics": eval_summary.get("object_metrics", {}),
        "required_metrics": required,
        "scope_comparison": comparison_rows,
    }

    (out_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "required_metrics.json").write_text(
        json.dumps(required, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "required_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in required.items():
            writer.writerow([key, value])
    with (out_dir / "metric_scope_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(comparison_rows[0].keys()) if comparison_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    lines = [
        "# E23 Object-Level Evaluator",
        "",
        f"- Weights: {args.weights}",
        f"- Validator: {args.validator}",
        f"- Mode: {args.mode}",
        "",
        "| Scope | AP image-level | AP object-level | Recall image-level | Recall object-level | Obj images | Obj count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            "| {scope} | {ap_img} | {ap_obj} | {rec_img} | {rec_obj} | {n_img} | {n_obj} |".format(
                scope=row["scope"],
                ap_img=_fmt(row["AP_image"]),
                ap_obj=_fmt(row["AP_object"]),
                rec_img=_fmt(row["Recall_image"]),
                rec_obj=_fmt(row["Recall_object"]),
                n_img=row["image_count_object_subset"],
                n_obj=row["object_count_object_subset"],
            )
        )
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--validator", default="e6", choices=["e6", "e13", "e14"])
    parser.add_argument("--mode", default="rgb_ir", choices=["rgb", "ir", "rgb_ir"])
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--data-rgb", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb.yaml")
    parser.add_argument("--data-ir", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml")
    parser.add_argument("--thresholds", default="/mnt/disk2/lhr/VSD/results/dataset_audit/train_thresholds.json")
    parser.add_argument("--image-metrics", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out-dir", default="/mnt/disk2/lhr/VSD/results/val/e23_object_level_evaluator")
    parser.add_argument("--cebs-alpha", type=float, default=0.05)
    parser.add_argument("--dark-threshold", type=float, default=33.50320816040039)
    parser.add_argument("--low-contrast-threshold", type=float, default=0.08425217866897583)
    parser.add_argument("--contrast-kernel", type=int, default=7)
    parser.add_argument("--suppression-temperature", type=float, default=0.08)
    parser.add_argument("--build-only", action="store_true", help="Only build object-level subsets and configs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset_summary = build_object_subsets(args)
    eval_summary: dict[str, Any] = {"object_metrics": {}, "required_metrics": {}}
    if not args.build_only:
        eval_summary = evaluate_object_scopes(args, subset_summary)
    write_reports(args, subset_summary, eval_summary)
    print(Path(args.out_dir) / "metrics_summary.md")


if __name__ == "__main__":
    main()
