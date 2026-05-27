#!/usr/bin/env python3
"""Offline calibration for S6.5 E25/E26 using cached train/val predictions.

This script intentionally uses train/val only.  It consumes cached E20 E6
prediction labels exported at conf=0.25 and NMS IoU=0.70, so the NMS sweep is
recorded as the cache condition rather than re-running inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from PIL import Image, ImageDraw


ROOT = Path("/mnt/disk2/lhr/VSD")
DATA_RGB_IR = ROOT / "configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml"
TRAIN_PRED_DIR = ROOT / "results/val/e20_train_for_e22_hn/predictions/E6/labels"
VAL_PRED_DIR = ROOT / "results/val/e20_0_error_delta_analysis/predictions/E6/labels"
TRAIN_E22 = ROOT / "results/val/e22_0_train_hard_negative_taxonomy/hard_negative_list.csv"
VAL_E22 = ROOT / "results/val/e22_0_hard_negative_taxonomy/hard_negative_list.csv"
E6_WEIGHTS = ROOT / "results/val/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt"
E6_REQUIRED = ROOT / "results/val/e6_feature_fusion_multiscale_val/required_metrics.json"
E6_OBJECT = ROOT / "results/val/e23_object_level_evaluator/required_metrics.json"
OBJECT_SUBSET_ROOT = ROOT / "results/val/e23_object_level_evaluator/object_level_subsets"


@dataclass(frozen=True)
class Box:
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float = 1.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class SplitData:
    name: str
    images: list[Path]
    names: dict[int, str]
    gt: dict[str, list[Box]]
    pred: dict[str, list[Box]]
    tags: dict[str, set[str]]
    image_count: int
    gt_count: int


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return data


def split_path(data_yaml: Path, split: str) -> Path:
    cfg = load_yaml(data_yaml)
    root = Path(str(cfg.get("path", ".")))
    entry = Path(str(cfg[split]))
    return entry if entry.is_absolute() else root / entry


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def iter_images(root: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in suffixes)


def class_names(data_yaml: Path) -> dict[int, str]:
    raw = load_yaml(data_yaml).get("names", {})
    if isinstance(raw, list):
        return {i: str(v) for i, v in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    return {}


def xywhn_to_xyxy(vals: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = vals
    return (
        (x - w / 2.0) * width,
        (y - h / 2.0) * height,
        (x + w / 2.0) * width,
        (y + h / 2.0) * height,
    )


def read_boxes(path: Path, width: int, height: int, with_conf: bool) -> list[Box]:
    if not path.exists():
        return []
    boxes: list[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        conf = float(parts[5]) if with_conf and len(parts) >= 6 else 1.0
        boxes.append(
            Box(
                cls=int(float(parts[0])),
                xyxy=xywhn_to_xyxy([float(x) for x in parts[1:5]], width, height),
                conf=conf,
            )
        )
    return boxes


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def stem_set_from_subset(path: Path, split: str) -> set[str]:
    if not path.exists():
        return set()
    return {p.stem for p in iter_images(split_path(path, split))}


def load_split(split: str, pred_dir: Path) -> SplitData:
    image_dir = split_path(DATA_RGB_IR, split)
    images = iter_images(image_dir)
    names = class_names(DATA_RGB_IR)
    subset_root = ROOT / "configs/dronevehicle_resplit/subsets"
    subset_stems = {
        "dark": stem_set_from_subset(subset_root / "rgb_ir_dark.yaml", split),
        "small": stem_set_from_subset(subset_root / "rgb_ir_small.yaml", split),
        "dark-small": stem_set_from_subset(subset_root / "rgb_ir_dark-small.yaml", split),
        "tiny": stem_set_from_subset(subset_root / "rgb_ir_tiny.yaml", split),
        "low-contrast": stem_set_from_subset(subset_root / "rgb_ir_low-contrast.yaml", split),
    }

    gt: dict[str, list[Box]] = {}
    pred: dict[str, list[Box]] = {}
    tags: dict[str, set[str]] = {}
    for image in images:
        with Image.open(image) as im:
            width, height = im.size
        stem = image.stem
        gt[stem] = read_boxes(label_path_for_image(image), width, height, with_conf=False)
        pred[stem] = read_boxes(pred_dir / f"{stem}.txt", width, height, with_conf=True)
        tags[stem] = {name for name, stems in subset_stems.items() if stem in stems}
    return SplitData(
        name=split,
        images=images,
        names=names,
        gt=gt,
        pred=pred,
        tags=tags,
        image_count=len(images),
        gt_count=sum(len(v) for v in gt.values()),
    )


def taxonomy(fp: Box, gt: list[Box]) -> tuple[str, float, int | None]:
    best_any = 0.0
    best_any_cls: int | None = None
    best_same = 0.0
    for g in gt:
        val = iou(fp.xyxy, g.xyxy)
        if val > best_any:
            best_any = val
            best_any_cls = g.cls
        if g.cls == fp.cls and val > best_same:
            best_same = val
    if best_same >= 0.50:
        return "duplicate_or_conf_threshold", best_same, fp.cls
    if best_any >= 0.30 and best_any_cls is not None and best_any_cls != fp.cls:
        return "class_confusion", best_any, best_any_cls
    if best_same >= 0.10:
        return "localization_error", best_same, fp.cls
    if best_any >= 0.10:
        return "near_object_background", best_any, best_any_cls
    return "background_far", best_any, best_any_cls


def match_predictions(
    data: SplitData,
    keep: Callable[[Box, set[str]], bool],
    stems: set[str] | None = None,
    gt_override: dict[str, list[Box]] | None = None,
) -> dict[str, Any]:
    selected_stems = stems if stems is not None else set(data.gt)
    gt_map = gt_override if gt_override is not None else data.gt
    total_gt = sum(len(gt_map.get(stem, [])) for stem in selected_stems)
    total_pred = 0
    tp = 0
    fp = 0
    matched_by_class: Counter[int] = Counter()
    gt_by_class: Counter[int] = Counter()
    pred_by_class: Counter[int] = Counter()
    fp_by_class: Counter[int] = Counter()
    fp_taxonomy: Counter[str] = Counter()
    dark_fp = 0
    low_contrast_fp = 0
    dark_images = 0
    low_contrast_images = 0

    for stem in selected_stems:
        gt = gt_map.get(stem, [])
        tags = data.tags.get(stem, set())
        if "dark" in tags:
            dark_images += 1
        if "low-contrast" in tags:
            low_contrast_images += 1
        for g in gt:
            gt_by_class[g.cls] += 1
        preds = [p for p in data.pred.get(stem, []) if keep(p, tags)]
        preds.sort(key=lambda b: b.conf, reverse=True)
        total_pred += len(preds)
        for p in preds:
            pred_by_class[p.cls] += 1
        used: set[int] = set()
        for p in preds:
            best_i = -1
            best_iou = 0.0
            for idx, g in enumerate(gt):
                if idx in used or g.cls != p.cls:
                    continue
                val = iou(p.xyxy, g.xyxy)
                if val >= 0.50 and val > best_iou:
                    best_iou = val
                    best_i = idx
            if best_i >= 0:
                used.add(best_i)
                tp += 1
                matched_by_class[p.cls] += 1
            else:
                fp += 1
                fp_by_class[p.cls] += 1
                label, _, _ = taxonomy(p, gt)
                fp_taxonomy[label] += 1
                if "dark" in tags:
                    dark_fp += 1
                if "low-contrast" in tags:
                    low_contrast_fp += 1

    fn = total_gt - tp
    precision = tp / total_pred if total_pred else 0.0
    recall = tp / total_gt if total_gt else 0.0
    rows: dict[str, Any] = {
        "images": len(selected_stems),
        "gt": total_gt,
        "pred": total_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fp_per_image": fp / len(selected_stems) if selected_stems else 0.0,
        "fppi_dark": dark_fp / dark_images if dark_images else 0.0,
        "fppi_low_contrast": low_contrast_fp / low_contrast_images if low_contrast_images else 0.0,
        "class_confusion_fp": fp_taxonomy["class_confusion"],
        "background_far_fp": fp_taxonomy["background_far"],
        "localization_error_fp": fp_taxonomy["localization_error"],
        "duplicate_or_conf_threshold_fp": fp_taxonomy["duplicate_or_conf_threshold"],
        "near_object_background_fp": fp_taxonomy["near_object_background"],
        "per_class": {},
    }
    for cls_id in sorted(set(gt_by_class) | set(pred_by_class)):
        c_tp = matched_by_class[cls_id]
        c_pred = pred_by_class[cls_id]
        c_gt = gt_by_class[cls_id]
        c_prec = c_tp / c_pred if c_pred else 0.0
        c_rec = c_tp / c_gt if c_gt else 0.0
        rows["per_class"][str(cls_id)] = {
            "class": data.names.get(cls_id, str(cls_id)),
            "gt": c_gt,
            "pred": c_pred,
            "tp": c_tp,
            "fp": fp_by_class[cls_id],
            "precision": c_prec,
            "recall": c_rec,
        }
    return rows


def ap50_for_class(
    data: SplitData,
    cls_id: int,
    keep: Callable[[Box, set[str]], bool],
    stems: set[str] | None = None,
    gt_override: dict[str, list[Box]] | None = None,
) -> float:
    selected_stems = stems if stems is not None else set(data.gt)
    gt_map = gt_override if gt_override is not None else data.gt
    total_gt = sum(1 for stem in selected_stems for g in gt_map.get(stem, []) if g.cls == cls_id)
    if total_gt == 0:
        return 0.0
    candidates: list[tuple[float, str, Box]] = []
    for stem in selected_stems:
        tags = data.tags.get(stem, set())
        for p in data.pred.get(stem, []):
            if p.cls == cls_id and keep(p, tags):
                candidates.append((p.conf, stem, p))
    candidates.sort(key=lambda x: x[0], reverse=True)
    used: dict[str, set[int]] = defaultdict(set)
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    for _, stem, p in candidates:
        best_i = -1
        best_iou = 0.0
        gt = gt_map.get(stem, [])
        for idx, g in enumerate(gt):
            if idx in used[stem] or g.cls != cls_id:
                continue
            val = iou(p.xyxy, g.xyxy)
            if val >= 0.50 and val > best_iou:
                best_iou = val
                best_i = idx
        if best_i >= 0:
            used[stem].add(best_i)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    if not tp_flags:
        return 0.0
    cum_tp = 0
    cum_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for t, f in zip(tp_flags, fp_flags):
        cum_tp += t
        cum_fp += f
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / total_gt)
    ap = 0.0
    for threshold in [x / 100 for x in range(0, 101)]:
        prec = max((p for p, r in zip(precisions, recalls) if r >= threshold), default=0.0)
        ap += prec / 101.0
    return ap


def map50(data: SplitData, keep: Callable[[Box, set[str]], bool], stems: set[str] | None = None, gt_override: dict[str, list[Box]] | None = None) -> float:
    values = []
    gt_map = gt_override if gt_override is not None else data.gt
    selected_stems = stems if stems is not None else set(gt_map)
    for cls_id in data.names:
        if any(g.cls == cls_id for stem in selected_stems for g in gt_map.get(stem, [])):
            values.append(ap50_for_class(data, cls_id, keep, stems=selected_stems, gt_override=gt_map))
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def flatten_metric(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "per_class":
            continue
        out[f"{prefix}_{key}"] = value
    return out


def make_global_keep(threshold: float) -> Callable[[Box, set[str]], bool]:
    return lambda p, tags: p.conf >= threshold


def size_bucket(box: Box, tiny_area: float, small_area: float) -> str:
    if box.area <= tiny_area:
        return "tiny"
    if box.area <= small_area:
        return "small"
    return "other"


def pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = sorted(rows, key=lambda r: (float(r["val_fp_per_image"]), -float(r["val_recall"])))
    out: list[dict[str, Any]] = []
    best_recall = -1.0
    for row in candidates:
        rec = float(row["val_recall"])
        if rec > best_recall:
            out.append(row)
            best_recall = rec
    return out


def render_pareto_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    width, height = 720, 460
    margin = 60
    xs = [float(r["val_fp_per_image"]) for r in rows]
    ys = [float(r["val_recall"]) for r in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        max_y = min_y + 0.01

    def sx(x: float) -> float:
        return margin + (x - min_x) / (max_x - min_x) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - (y - min_y) / (max_y - min_y) * (height - 2 * margin)

    circles = []
    for row in rows:
        x = sx(float(row["val_fp_per_image"]))
        y = sy(float(row["val_recall"]))
        label = row["strategy"]
        circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb"><title>{label}</title></circle>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#111827"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#111827"/>
<text x="{width/2}" y="{height-16}" text-anchor="middle" font-family="sans-serif" font-size="14">FP/image</text>
<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">Recall@0.50</text>
<text x="{margin}" y="{height-margin+22}" font-family="sans-serif" font-size="11">{min_x:.3f}</text>
<text x="{width-margin}" y="{height-margin+22}" text-anchor="end" font-family="sans-serif" font-size="11">{max_x:.3f}</text>
<text x="{margin-8}" y="{height-margin}" text-anchor="end" font-family="sans-serif" font-size="11">{min_y:.3f}</text>
<text x="{margin-8}" y="{margin+4}" text-anchor="end" font-family="sans-serif" font-size="11">{max_y:.3f}</text>
{''.join(circles)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def render_pareto_png(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, x_label: str, y_label: str) -> None:
    if not rows:
        return
    width, height = 720, 460
    margin = 64
    xs = [float(r[x_key]) for r in rows]
    ys = [float(r[y_key]) for r in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        max_y = min_y + 0.01

    def sx(x: float) -> float:
        return margin + (x - min_x) / (max_x - min_x) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - (y - min_y) / (max_y - min_y) * (height - 2 * margin)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    axis = (17, 24, 39)
    blue = (37, 99, 235)
    draw.line((margin, height - margin, width - margin, height - margin), fill=axis, width=2)
    draw.line((margin, margin, margin, height - margin), fill=axis, width=2)
    draw.text((width // 2 - 40, height - 34), x_label, fill=axis)
    draw.text((12, height // 2 - 8), y_label, fill=axis)
    draw.text((margin, height - margin + 10), f"{min_x:.3f}", fill=axis)
    draw.text((width - margin - 50, height - margin + 10), f"{max_x:.3f}", fill=axis)
    draw.text((margin - 54, height - margin - 6), f"{min_y:.3f}", fill=axis)
    draw.text((margin - 54, margin - 6), f"{max_y:.3f}", fill=axis)
    for row in rows:
        x = sx(float(row[x_key]))
        y = sy(float(row[y_key]))
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=blue)
    image.save(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_sha256(path: Path) -> str:
    if not path.exists() or not path.is_dir():
        return ""
    h = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(file_path.relative_to(path)).encode("utf-8"))
        h.update(b"\0")
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def run_e25_0(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [42, 43, 44]
    cuda_available = torch_cuda_available()
    rows: list[dict[str, Any]] = []
    key_metric_sets: list[dict[str, Any]] = []
    for seed in seeds:
        train_dir = ROOT / f"results/val/e25_0_e13_3b_seed{seed}"
        val_dir = ROOT / f"results/val/e25_0_e13_3b_seed{seed}_val"
        best_weights = train_dir / "weights/best.pt"
        last_weights = train_dir / "weights/last.pt"
        args_yaml = train_dir / "args.yaml"
        required = val_dir / "required_metrics.json"
        object_required = ROOT / f"results/val/e25_0_e13_3b_seed{seed}_object_level/required_metrics.json"
        prediction_label_dirs = [
            ROOT / f"results/val/e25_0_e13_3b_seed{seed}_predictions/labels",
            val_dir / "labels",
            val_dir / "predictions/labels",
        ]
        prediction_label_dir = next((p for p in prediction_label_dirs if p.exists()), prediction_label_dirs[0])
        metrics = load_json(required)
        object_metrics = load_json(object_required)
        selected = {
            k: metrics.get(k)
            for k in (
                "mAP50-95",
                "AP_dark-small",
                "AP_tiny",
                "AP_low-contrast",
                "False Positives/image",
                "FPPI_dark",
            )
        }
        selected.update({k: object_metrics.get(k) for k in ("AP_dark-small_object", "AP_tiny_object", "AP_low-contrast_object")})
        if metrics:
            key_metric_sets.append(selected)
        rows.append(
            {
                "seed": seed,
                "train_dir": str(train_dir),
                "val_dir": str(val_dir),
                "args_yaml_exists": args_yaml.exists(),
                "best_pt_exists": best_weights.exists(),
                "best_pt_sha256": file_sha256(best_weights) if best_weights.exists() else "",
                "last_pt_exists": last_weights.exists(),
                "last_pt_sha256": file_sha256(last_weights) if last_weights.exists() else "",
                "prediction_labels_dir": str(prediction_label_dir),
                "prediction_labels_exists": prediction_label_dir.exists(),
                "prediction_labels_sha256": dir_sha256(prediction_label_dir),
                "required_metrics_exists": required.exists(),
                "object_metrics_exists": object_required.exists(),
                **selected,
            }
        )
    identical = bool(key_metric_sets) and all(m == key_metric_sets[0] for m in key_metric_sets)
    partial_outputs = any(
        row["args_yaml_exists"] or row["best_pt_exists"] or row["last_pt_exists"] or row["prediction_labels_exists"]
        for row in rows
    )
    status = "blocked_partial_seed_outputs_no_cuda" if partial_outputs and not cuda_available else "blocked_no_cuda_no_seed_outputs"
    if len(key_metric_sets) == 3:
        status = "seed_pipeline_failed" if identical else "done"
    if len(key_metric_sets) == 3 and identical:
        note = (
            "Complete E25_0 seed=42/43/44 outputs are present, but the selected image-level and "
            "object-level metrics are identical across all three seeds. Treat this as suspicious and "
            "do not use it as a valid multi-seed conclusion until the validation path is audited."
        )
    elif len(key_metric_sets) == 3:
        note = "Complete E25_0 seed=42/43/44 outputs are present and selected metrics differ across seeds."
    elif partial_outputs and not cuda_available:
        note = (
            "Partial E25_0 outputs exist, but the seed=42/43/44 pipeline is incomplete and CUDA is not "
            "currently available."
        )
    else:
        note = (
            "No complete E25_0 seed=42/43/44 outputs are present; valid rerun remains blocked until GPU "
            "training is available."
        )
    summary = {
        "experiment": "E25_0",
        "requested_seeds": seeds,
        "status": status,
        "cuda_available": cuda_available,
        "note": note,
        "rows": rows,
    }
    write_csv(out_dir / "seed_audit.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E25_0 E13_3b Multi-Seed Audit",
        "",
        f"- Status: {status}",
        "- Requested seeds: 42, 43, 44",
        f"- Current torch CUDA available: {cuda_available}",
        "- This offline audit did not launch training.",
        f"- Note: {note}",
        "",
        "| Seed | args.yaml | best.pt | last.pt | predictions | required_metrics | object_metrics | best.pt SHA256 | last.pt SHA256 | prediction SHA256 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['args_yaml_exists']} | {row['best_pt_exists']} | "
            f"{row['last_pt_exists']} | {row['prediction_labels_exists']} | "
            f"{row['required_metrics_exists']} | {row['object_metrics_exists']} | "
            f"{row['best_pt_sha256']} | {row['last_pt_sha256']} | {row['prediction_labels_sha256']} |"
        )
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "metrics_summary.md")


def run_e25_1(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_split("train", TRAIN_PRED_DIR)
    val = load_split("val", VAL_PRED_DIR)
    thresholds = load_json(ROOT / "results/dataset_audit/train_thresholds.json")
    tiny_area = float(thresholds.get("object_area_tiny_threshold", 880.0))
    small_area = float(thresholds.get("object_area_small_threshold", 1288.0))
    dark_small_stems, dark_small_gt = read_object_subset("dark-small_object")

    rules: list[tuple[str, str, Callable[[Box, set[str]], bool], dict[str, Any]]] = []
    for t in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80):
        rules.append(("global", f"conf_{t:.2f}", make_global_keep(t), {"global_conf": t, "nms_iou": 0.70}))
    for tiny_t in (0.25, 0.30, 0.35, 0.40, 0.45):
        for small_t in (0.25, 0.30, 0.35, 0.40):
            other_t = 0.30
            def keep(p: Box, tags: set[str], tiny_t: float = tiny_t, small_t: float = small_t, other_t: float = other_t) -> bool:
                bucket = size_bucket(p, tiny_area, small_area)
                return p.conf >= {"tiny": tiny_t, "small": small_t, "other": other_t}[bucket]

            rules.append(
                (
                    "size_wise",
                    f"tiny{tiny_t:.2f}_small{small_t:.2f}_other{other_t:.2f}",
                    keep,
                    {"tiny_conf": tiny_t, "small_conf": small_t, "other_conf": other_t, "nms_iou": 0.70},
                )
            )
    for dark_t in (0.25, 0.30, 0.35, 0.40, 0.45):
        for other_t in (0.25, 0.30, 0.35):
            def keep(p: Box, tags: set[str], dark_t: float = dark_t, other_t: float = other_t) -> bool:
                return p.conf >= (dark_t if "dark" in tags else other_t)

            rules.append(
                (
                    "illumination_wise",
                    f"dark{dark_t:.2f}_other{other_t:.2f}",
                    keep,
                    {"dark_conf": dark_t, "other_conf": other_t, "nms_iou": 0.70},
                )
            )
    for cls_id, cls_name in train.names.items():
        for cls_t in (0.30, 0.35, 0.40, 0.45, 0.50):
            def keep(p: Box, tags: set[str], cls_id: int = cls_id, cls_t: float = cls_t) -> bool:
                return p.conf >= (cls_t if p.cls == cls_id else 0.25)

            rules.append(("class_wise_single", f"{cls_name}_{cls_t:.2f}", keep, {"raised_class": cls_name, "raised_conf": cls_t, "nms_iou": 0.70}))

    rows: list[dict[str, Any]] = []
    for strategy, rule_id, keep, params in rules:
        train_metrics = match_predictions(train, keep)
        val_metrics = match_predictions(val, keep)
        object_dark_small_map50 = map50(val, keep, stems=dark_small_stems, gt_override=dark_small_gt)
        row: dict[str, Any] = {"strategy": strategy, "rule_id": rule_id, **params}
        row.update(flatten_metric("train", train_metrics))
        row.update(flatten_metric("val", val_metrics))
        row["val_dark_small_object_mAP50_cached"] = object_dark_small_map50
        rows.append(row)

    rows.sort(key=lambda r: (float(r["val_fp_per_image"]), -float(r["val_recall"])))
    write_csv(out_dir / "calibration_grid.csv", rows)
    pareto = pareto_rows(rows)
    write_csv(out_dir / "pareto_curve.csv", pareto)
    render_pareto_svg(out_dir / "pareto_curve.svg", pareto)
    render_pareto_png(
        out_dir / "pareto_ap_obj_vs_fppi_dark.png",
        rows,
        x_key="val_fppi_dark",
        y_key="val_dark_small_object_mAP50_cached",
        x_label="FPPI_dark",
        y_label="dark-small obj AP50",
    )

    def score(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(row["val_f1"]),
            -max(0.0, float(row["val_fp_per_image"]) - 1.469027),
            float(row["val_recall"]),
        )

    best_f1 = max(rows, key=score)
    best_low_fp = min((r for r in rows if float(r["val_recall"]) >= 0.90), key=lambda r: float(r["val_fp_per_image"]), default=min(rows, key=lambda r: float(r["val_fp_per_image"])))
    best_high_recall = max(rows, key=lambda r: (float(r["val_recall"]), -float(r["val_fp_per_image"])))
    classwise_suggestions: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["strategy"] != "class_wise_single":
            continue
        cls_name = str(row.get("raised_class", ""))
        if not cls_name:
            continue
        current = classwise_suggestions.get(cls_name)
        if current is None or score(row) > score(current):
            classwise_suggestions[cls_name] = row
    best = {
        "experiment": "E25_1",
        "cache_source": {
            "train_predictions": str(TRAIN_PRED_DIR),
            "val_predictions": str(VAL_PRED_DIR),
            "minimum_conf_in_cache": 0.25,
            "nms_iou_in_cache": 0.70,
            "nms_sweep_status": "limited_to_cached_nms_iou_0.70_no_reinference",
        },
        "E25_balanced": best_f1,
        "E25_low_fp": best_low_fp,
        "E25_high_recall": best_high_recall,
        "best_f1": best_f1,
        "best_low_fp_recall90": best_low_fp,
    }
    (out_dir / "best_operating_points.json").write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "classwise_thresholds.json").write_text(
        json.dumps(
            {
                "note": "Single-class threshold suggestions from E25_1 cache sweep; E26_1 performs sequential class-wise calibration.",
                "thresholds": {
                    cls_name: {
                        "threshold": row.get("raised_conf"),
                        "rule_id": row.get("rule_id"),
                        "val_fp_per_image": row.get("val_fp_per_image"),
                        "val_fppi_dark": row.get("val_fppi_dark"),
                        "val_class_confusion_fp": row.get("val_class_confusion_fp"),
                    }
                    for cls_name, row in sorted(classwise_suggestions.items())
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    required = {
        "best_strategy": best_f1["strategy"],
        "best_rule_id": best_f1["rule_id"],
        "Precision": best_f1["val_precision"],
        "Recall": best_f1["val_recall"],
        "False Positives/image": best_f1["val_fp_per_image"],
        "FPPI_dark": best_f1["val_fppi_dark"],
        "FPPI_low-contrast": best_f1["val_fppi_low_contrast"],
        "class_confusion_fp": best_f1["val_class_confusion_fp"],
        "background_far_fp": best_f1["val_background_far_fp"],
    }
    (out_dir / "required_metrics.json").write_text(json.dumps(required, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E25_1 E6 Calibration Sweep",
        "",
        "- Source: cached E6 train/val predictions, conf>=0.25, NMS IoU=0.70.",
        "- NMS sweep is limited to the cached 0.70 operating point because no new inference was run.",
        "- Object AP in the Pareto PNG is cached AP50 on the dark-small object subset, not full COCO AP50-95.",
        f"- Grid rows: {len(rows)}",
        f"- Best F1: {best_f1['strategy']} / {best_f1['rule_id']} | val F1={float(best_f1['val_f1']):.6f}, FP/image={float(best_f1['val_fp_per_image']):.6f}, FPPI_dark={float(best_f1['val_fppi_dark']):.6f}",
        f"- Low-FP recall>=0.90: {best_low_fp['strategy']} / {best_low_fp['rule_id']} | val recall={float(best_low_fp['val_recall']):.6f}, FP/image={float(best_low_fp['val_fp_per_image']):.6f}",
        f"- High recall: {best_high_recall['strategy']} / {best_high_recall['rule_id']} | val recall={float(best_high_recall['val_recall']):.6f}, FP/image={float(best_high_recall['val_fp_per_image']):.6f}",
    ]
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "metrics_summary.md")


def read_object_subset(scope: str) -> tuple[set[str], dict[str, list[Box]]]:
    root = OBJECT_SUBSET_ROOT / scope / "rgb"
    image_dir = root / "images/val"
    label_dir = root / "labels/val"
    images = iter_images(image_dir)
    gt: dict[str, list[Box]] = {}
    for image in images:
        with Image.open(image) as im:
            width, height = im.size
        gt[image.stem] = read_boxes(label_dir / f"{image.stem}.txt", width, height, with_conf=False)
    return {p.stem for p in images}, gt


def run_e26_1(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_split("train", TRAIN_PRED_DIR)
    val = load_split("val", VAL_PRED_DIR)

    class_thresholds = {cls_id: 0.25 for cls_id in train.names}
    search_rows: list[dict[str, Any]] = []
    for cls_id, cls_name in train.names.items():
        best_t = 0.25
        best_score: tuple[float, float, float] | None = None
        for t in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70):
            candidate = dict(class_thresholds)
            candidate[cls_id] = t

            def keep(p: Box, tags: set[str], candidate: dict[int, float] = candidate) -> bool:
                return p.conf >= candidate.get(p.cls, 0.25)

            metrics = match_predictions(train, keep)
            cls_metrics = metrics["per_class"].get(str(cls_id), {})
            cls_confusion_penalty = metrics["class_confusion_fp"]
            score = (
                float(metrics["f1"]),
                -float(cls_confusion_penalty),
                float(cls_metrics.get("recall", 0.0)),
            )
            search_rows.append(
                {
                    "class_id": cls_id,
                    "class": cls_name,
                    "threshold": t,
                    "train_f1": metrics["f1"],
                    "train_fp_per_image": metrics["fp_per_image"],
                    "train_class_confusion_fp": metrics["class_confusion_fp"],
                    "train_class_recall": cls_metrics.get("recall", 0.0),
                    "train_class_precision": cls_metrics.get("precision", 0.0),
                }
            )
            if best_score is None or score > best_score:
                best_score = score
                best_t = t
        class_thresholds[cls_id] = best_t

    def final_keep(p: Box, tags: set[str]) -> bool:
        return p.conf >= class_thresholds.get(p.cls, 0.25)

    train_final = match_predictions(train, final_keep)
    val_final = match_predictions(val, final_keep)
    dark_small_stems, dark_small_gt = read_object_subset("dark-small_object")
    object_dark_small = match_predictions(val, final_keep, stems=dark_small_stems, gt_override=dark_small_gt)
    object_dark_small_map50 = map50(val, final_keep, stems=dark_small_stems, gt_override=dark_small_gt)

    per_class_rows: list[dict[str, Any]] = []
    for cls_id, cls_name in val.names.items():
        ap50 = ap50_for_class(val, cls_id, final_keep)
        cls_metrics = val_final["per_class"].get(str(cls_id), {})
        per_class_rows.append(
            {
                "class_id": cls_id,
                "class": cls_name,
                "threshold": class_thresholds.get(cls_id, 0.25),
                "AP50_cached": ap50,
                "precision": cls_metrics.get("precision", 0.0),
                "recall": cls_metrics.get("recall", 0.0),
                "gt": cls_metrics.get("gt", 0),
                "pred": cls_metrics.get("pred", 0),
                "fp": cls_metrics.get("fp", 0),
            }
        )

    write_csv(out_dir / "class_threshold_search.csv", search_rows)
    write_csv(out_dir / "per_class_metrics.csv", per_class_rows)
    thresholds_by_name = {val.names[k]: v for k, v in class_thresholds.items()}
    best = {
        "experiment": "E26_1",
        "class_thresholds": thresholds_by_name,
        "train": {k: v for k, v in train_final.items() if k != "per_class"},
        "val": {k: v for k, v in val_final.items() if k != "per_class"},
        "object_dark_small": {k: v for k, v in object_dark_small.items() if k != "per_class"},
        "object_dark_small_mAP50_cached": object_dark_small_map50,
        "source": {
            "train_predictions": str(TRAIN_PRED_DIR),
            "val_predictions": str(VAL_PRED_DIR),
            "minimum_conf_in_cache": 0.25,
            "nms_iou_in_cache": 0.70,
        },
    }
    (out_dir / "best_operating_points.json").write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    required = {
        "Precision": val_final["precision"],
        "Recall": val_final["recall"],
        "False Positives/image": val_final["fp_per_image"],
        "FPPI_dark": val_final["fppi_dark"],
        "class_confusion_fp": val_final["class_confusion_fp"],
        "AP50_dark-small_object_cached": object_dark_small_map50,
        "Precision_dark-small_object_cached": object_dark_small["precision"],
        "Recall_dark-small_object_cached": object_dark_small["recall"],
    }
    (out_dir / "required_metrics.json").write_text(json.dumps(required, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E26_1 Class-Wise Threshold Calibration",
        "",
        "- Source: cached E6 train/val predictions, conf>=0.25, NMS IoU=0.70.",
        f"- Selected class thresholds: {json.dumps(thresholds_by_name, ensure_ascii=False)}",
        f"- Val FP/image: {val_final['fp_per_image']:.6f}",
        f"- Val FPPI_dark: {val_final['fppi_dark']:.6f}",
        f"- Val class_confusion FP: {val_final['class_confusion_fp']}",
        f"- Dark-small object cached mAP50/precision/recall: {object_dark_small_map50:.6f}/{object_dark_small['precision']:.6f}/{object_dark_small['recall']:.6f}",
    ]
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "metrics_summary.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    p0 = sub.add_parser("e25_0")
    p0.add_argument("--out-dir", default=str(ROOT / "results/val/e25_0_e13_3b_multiseed_rerun"))
    p0.set_defaults(func=run_e25_0)
    p1 = sub.add_parser("e25_1")
    p1.add_argument("--out-dir", default=str(ROOT / "results/val/e25_1_e6_calibration_sweep"))
    p1.set_defaults(func=run_e25_1)
    p2 = sub.add_parser("e26_1")
    p2.add_argument("--out-dir", default=str(ROOT / "results/val/e26_1_classwise_threshold_calibration"))
    p2.set_defaults(func=run_e26_1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
