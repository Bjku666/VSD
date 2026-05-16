#!/usr/bin/env python3
"""DroneVehicle 数据集重划分流水线：审计、转换与子集生成。

该脚本用于构建规范化的 RGB/IR YOLO 数据集，并生成面向 dark/small 分析的
重划分评测子集，同时将报告输出与训练日志目录分离。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SplitSpec:
    split: str
    rgb_img_dir: str
    ir_img_dir: str
    rgb_xml_dir: str
    ir_xml_dir: str


SPLIT_SPECS: Tuple[SplitSpec, ...] = (
    SplitSpec("train", "trainimg", "trainimgr", "trainlabel", "trainlabelr"),
    SplitSpec("val", "valimg", "valimgr", "vallabel", "vallabelr"),
    SplitSpec("test", "testimg", "testimgr", "testlabel", "testlabelr"),
)

CLASS_NAMES: List[str] = ["car", "truck", "bus", "van", "freight_car"]
CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

CLASS_ALIASES: Dict[str, Optional[str]] = {
    "car": "car",
    "truck": "truck",
    "truvk": "truck",
    "bus": "bus",
    "van": "van",
    "freight car": "freight_car",
    "freight_car": "freight_car",
    "feright car": "freight_car",
    "feright_car": "freight_car",
    "feright": "freight_car",
    "*": None,
    "": None,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CROP_BORDER_PX = 100
CONTRAST_RING_SCALE = 1.6


def normalize_class_name(raw: str) -> Optional[str]:
    text = (raw or "").strip().lower().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    canonical = CLASS_ALIASES.get(text)
    if canonical is not None:
        return canonical
    return None


def collect_stem_to_file(dir_path: Path, suffixes: Optional[set] = None) -> Dict[str, Path]:
    if not dir_path.exists():
        return {}
    mapping: Dict[str, Path] = {}
    for path in sorted(dir_path.iterdir()):
        if not path.is_file():
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        mapping[path.stem] = path
    return mapping


def parse_polygon(poly_node: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for child in list(poly_node):
        tag = child.tag.lower()
        text = (child.text or "").strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        if tag.startswith("x"):
            xs.append(value)
        elif tag.startswith("y"):
            ys.append(value)
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def parse_bndbox(box_node: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    def get_float(name: str) -> Optional[float]:
        text = box_node.findtext(name)
        if text is None:
            return None
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    xmin = get_float("xmin")
    ymin = get_float("ymin")
    xmax = get_float("xmax")
    ymax = get_float("ymax")
    if xmin is None or ymin is None or xmax is None or ymax is None:
        return None
    return xmin, ymin, xmax, ymax


def clamp_bbox(
    bbox: Tuple[float, float, float, float], width: float, height: float
) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def parse_xml(
    xml_path: Path,
) -> Tuple[int, int, List[Tuple[str, Tuple[float, float, float, float]]], Dict[str, int]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width_text = root.findtext("size/width")
    height_text = root.findtext("size/height")
    if width_text is None or height_text is None:
        raise ValueError(f"Missing size in {xml_path}")

    width = int(float(width_text))
    height = int(float(height_text))

    stats = {
        "objects_total": 0,
        "objects_kept": 0,
        "objects_unknown_class": 0,
        "objects_no_box": 0,
        "objects_invalid_box": 0,
    }

    boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        stats["objects_total"] += 1
        cls_name = normalize_class_name(obj.findtext("name") or "")
        if cls_name is None:
            stats["objects_unknown_class"] += 1
            continue

        bbox: Optional[Tuple[float, float, float, float]] = None
        poly_node = obj.find("polygon")
        if poly_node is not None:
            bbox = parse_polygon(poly_node)
        if bbox is None:
            box_node = obj.find("bndbox")
            if box_node is not None:
                bbox = parse_bndbox(box_node)
        if bbox is None:
            stats["objects_no_box"] += 1
            continue

        bbox = clamp_bbox(bbox, width, height)
        if bbox is None:
            stats["objects_invalid_box"] += 1
            continue

        boxes.append((cls_name, bbox))
        stats["objects_kept"] += 1

    return width, height, boxes, stats


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def greedy_match_iou(
    boxes_a: List[Tuple[float, float, float, float]],
    boxes_b: List[Tuple[float, float, float, float]],
) -> Tuple[int, float]:
    if not boxes_a or not boxes_b:
        return 0, 0.0

    used_a = set()
    used_b = set()
    matched = 0
    iou_sum = 0.0

    while True:
        best_i = -1
        best_j = -1
        best_iou = -1.0
        for i, box_a in enumerate(boxes_a):
            if i in used_a:
                continue
            for j, box_b in enumerate(boxes_b):
                if j in used_b:
                    continue
                iou = bbox_iou(box_a, box_b)
                if iou > best_iou:
                    best_iou = iou
                    best_i = i
                    best_j = j
        if best_i < 0 or best_j < 0:
            break
        used_a.add(best_i)
        used_b.add(best_j)
        matched += 1
        iou_sum += max(0.0, best_iou)

    return matched, iou_sum


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool = False) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def save_cropped_image(src: Path, dst: Path, crop_border: int, overwrite: bool = True) -> Tuple[int, int]:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if overwrite:
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        else:
            with Image.open(dst) as existing:
                return existing.size

    with Image.open(src) as img:
        width, height = img.size
        if crop_border < 0:
            raise ValueError("crop_border must be non-negative")
        if crop_border == 0:
            cropped = img.copy()
        else:
            if width <= crop_border * 2 or height <= crop_border * 2:
                raise ValueError(f"Image too small for crop_border={crop_border}: {src}")
            cropped = img.crop((crop_border, crop_border, width - crop_border, height - crop_border))
        cropped.save(dst)
        return cropped.size


def crop_bbox_to_image(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
    crop_border: int,
) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = bbox
    if crop_border:
        x1 -= crop_border
        y1 -= crop_border
        x2 -= crop_border
        y2 -= crop_border
        width -= crop_border * 2
        height -= crop_border * 2
        if width <= 0 or height <= 0:
            return None
    return clamp_bbox((x1, y1, x2, y2), width, height)


def compute_box_contrast(
    gray_image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    ring_scale: float = CONTRAST_RING_SCALE,
) -> Optional[float]:
    height, width = gray_image.shape[:2]
    x1, y1, x2, y2 = bbox

    x1_i = int(max(0, min(width - 1, np.floor(x1))))
    y1_i = int(max(0, min(height - 1, np.floor(y1))))
    x2_i = int(max(x1_i + 1, min(width, np.ceil(x2))))
    y2_i = int(max(y1_i + 1, min(height, np.ceil(y2))))

    inner = gray_image[y1_i:y2_i, x1_i:x2_i]
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

    if ex2 <= ex1 or ey2 <= ey1:
        return None

    expanded = gray_image[ey1:ey2, ex1:ex2]
    if expanded.size == 0:
        return None

    mask = np.ones(expanded.shape[:2], dtype=bool)
    inner_rel_x1 = int(max(0, min(expanded.shape[1], x1_i - ex1)))
    inner_rel_y1 = int(max(0, min(expanded.shape[0], y1_i - ey1)))
    inner_rel_x2 = int(max(inner_rel_x1 + 1, min(expanded.shape[1], x2_i - ex1)))
    inner_rel_y2 = int(max(inner_rel_y1 + 1, min(expanded.shape[0], y2_i - ey1)))
    mask[inner_rel_y1:inner_rel_y2, inner_rel_x1:inner_rel_x2] = False

    ring_pixels = expanded[mask]
    if ring_pixels.size == 0:
        return None

    inner_mean = float(np.mean(inner))
    ring_mean = float(np.mean(ring_pixels))
    return abs(inner_mean - ring_mean) / (ring_mean + 1e-6)


def analyze_image_record(
    img_path: Path,
    label_records: List[Tuple[int, float, float, float, float]],
    contrast_ring_scale: float = CONTRAST_RING_SCALE,
) -> Dict[str, object]:
    with Image.open(img_path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        width, height = img.size

    brightness = float(np.mean(gray)) if gray.size else 0.0
    areas: List[float] = []
    contrasts: List[float] = []
    for _cls_id, xc, yc, bw, bh in label_records:
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        areas.append(max(0.0, (x2 - x1) * (y2 - y1)))
        contrast = compute_box_contrast(gray, (x1, y1, x2, y2), ring_scale=contrast_ring_scale)
        if contrast is not None:
            contrasts.append(float(contrast))

    return {
        "brightness": brightness,
        "width": width,
        "height": height,
        "object_count": len(label_records),
        "min_area": float(min(areas)) if areas else 0.0,
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "min_contrast": float(min(contrasts)) if contrasts else 0.0,
        "mean_contrast": float(np.mean(contrasts)) if contrasts else 0.0,
        "area_values": areas,
        "contrast_values": contrasts,
    }


def yolo_line_from_bbox(
    cls_id: int,
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> str:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    xc /= width
    yc /= height
    bw /= width
    bh /= height

    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)

    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def split_maps(raw_root: Path, spec: SplitSpec) -> Dict[str, Dict[str, Path]]:
    split_root = raw_root / spec.split
    return {
        "rgb_imgs": collect_stem_to_file(split_root / spec.rgb_img_dir, IMAGE_SUFFIXES),
        "ir_imgs": collect_stem_to_file(split_root / spec.ir_img_dir, IMAGE_SUFFIXES),
        "rgb_xml": collect_stem_to_file(split_root / spec.rgb_xml_dir, {".xml"}),
        "ir_xml": collect_stem_to_file(split_root / spec.ir_xml_dir, {".xml"}),
    }


def size_bucket(area_px: float, small_thr: float, medium_thr: float) -> str:
    if area_px < small_thr:
        return "small"
    if area_px < medium_thr:
        return "medium"
    return "large"


def get_original_split_root(prepared_root: Path) -> Path:
    return prepared_root / "original_split"


def get_resplit_subset_root(prepared_root: Path) -> Path:
    return prepared_root / "resplit_subsets"


def render_report_figures(
    report_root: Path,
    class_total: Counter,
    objs_per_image_all: List[int],
    bbox_area_all: List[float],
    brightness_all: List[float],
    dark_thr: float,
    bright_thr: float,
    val_bucket_counts: Counter,
    dark_size_counter: Counter,
    split_stats: Dict[str, Dict[str, object]],
) -> List[str]:
    fig_dir = report_root / "figures"
    ensure_dir(fig_dir)

    generated: List[str] = []

    labels = CLASS_NAMES
    values = [int(class_total.get(name, 0)) for name in labels]
    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, values)
    plt.title("Class Distribution")
    plt.ylabel("Object Count")
    plt.xticks(rotation=20)
    plt.tight_layout()
    path = fig_dir / "class_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()
    generated.append(str(path))

    plt.figure(figsize=(8, 4.5))
    plt.hist(objs_per_image_all, bins=50)
    plt.title("Objects Per Image")
    plt.xlabel("Objects")
    plt.ylabel("Image Count")
    plt.tight_layout()
    path = fig_dir / "objects_per_image_hist.png"
    plt.savefig(path, dpi=160)
    plt.close()
    generated.append(str(path))

    log_area = np.log10(np.maximum(np.asarray(bbox_area_all, dtype=np.float32), 1.0))
    plt.figure(figsize=(8, 4.5))
    plt.hist(log_area, bins=60)
    plt.title("BBox Area Distribution (log10 px^2)")
    plt.xlabel("log10(area)")
    plt.ylabel("BBox Count")
    plt.tight_layout()
    path = fig_dir / "bbox_area_log_hist.png"
    plt.savefig(path, dpi=160)
    plt.close()
    generated.append(str(path))

    plt.figure(figsize=(8, 4.5))
    plt.hist(brightness_all, bins=60)
    plt.axvline(dark_thr, linestyle="--", linewidth=1.5)
    plt.axvline(bright_thr, linestyle="--", linewidth=1.5)
    plt.title("RGB Brightness Distribution")
    plt.xlabel("Mean Gray Value")
    plt.ylabel("Image Count")
    plt.tight_layout()
    path = fig_dir / "brightness_hist.png"
    plt.savefig(path, dpi=160)
    plt.close()
    generated.append(str(path))

    bucket_labels = ["dark", "normal", "bright"]
    bucket_values = [int(val_bucket_counts.get(k, 0)) for k in bucket_labels]
    plt.figure(figsize=(6, 4))
    plt.bar(bucket_labels, bucket_values)
    plt.title("Train Brightness Buckets")
    plt.ylabel("Image Count")
    plt.tight_layout()
    path = fig_dir / "val_brightness_buckets.png"
    plt.savefig(path, dpi=160)
    plt.close()
    generated.append(str(path))

    size_labels = ["small", "medium", "large"]
    size_values = [int(dark_size_counter.get(k, 0)) for k in size_labels]
    plt.figure(figsize=(6, 4))
    plt.bar(size_labels, size_values)
    plt.title("Train Dark Subset Size Distribution")
    plt.ylabel("Object Count")
    plt.tight_layout()
    path = fig_dir / "dark_subset_size_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()
    generated.append(str(path))

    split_labels = ["train", "val", "test"]
    split_values = [
        float(split_stats.get(split, {}).get("avg_objects_per_image", 0.0))
        for split in split_labels
    ]
    plt.figure(figsize=(6, 4))
    plt.bar(split_labels, split_values)
    plt.title("Avg Objects Per Image by Split")
    plt.ylabel("Average Objects")
    plt.tight_layout()
    path = fig_dir / "split_avg_objects.png"
    plt.savefig(path, dpi=160)
    plt.close()
    generated.append(str(path))

    return generated


def run_audit(raw_root: Path, log_root: Path, iou_warn_threshold: float) -> Dict[str, object]:
    ensure_dir(log_root)

    summary = {
        "dataset": "DroneVehicle",
        "raw_root": str(raw_root),
        "iou_warn_threshold": iou_warn_threshold,
        "splits": {},
    }

    for spec in SPLIT_SPECS:
        maps = split_maps(raw_root, spec)
        rgb_img_stems = set(maps["rgb_imgs"].keys())
        ir_img_stems = set(maps["ir_imgs"].keys())
        rgb_xml_stems = set(maps["rgb_xml"].keys())
        ir_xml_stems = set(maps["ir_xml"].keys())

        paired_stems = rgb_img_stems & ir_img_stems & rgb_xml_stems & ir_xml_stems
        all_stems = rgb_img_stems | ir_img_stems | rgb_xml_stems | ir_xml_stems

        split_result = {
            "total_unique_stems": len(all_stems),
            "paired_success": len(paired_stems),
            "paired_failed": len(all_stems - paired_stems),
            "missing_ir_image": sorted(rgb_img_stems - ir_img_stems)[:20],
            "missing_rgb_image": sorted(ir_img_stems - rgb_img_stems)[:20],
            "missing_ir_label": sorted(rgb_xml_stems - ir_xml_stems)[:20],
            "missing_rgb_label": sorted(ir_xml_stems - rgb_xml_stems)[:20],
        }

        size_mismatch_count = 0
        xml_size_mismatch_count = 0
        object_count_equal = 0
        class_hist_equal = 0
        low_iou_count = 0
        count_drift_examples: List[str] = []
        low_iou_examples: List[Dict[str, object]] = []

        rgb_obj_counts: List[int] = []
        ir_obj_counts: List[int] = []
        matched_iou_means: List[float] = []

        aggregate_parse_stats_rgb = Counter()
        aggregate_parse_stats_ir = Counter()

        for idx, stem in enumerate(sorted(paired_stems), start=1):
            rgb_img_path = maps["rgb_imgs"][stem]
            ir_img_path = maps["ir_imgs"][stem]
            rgb_xml_path = maps["rgb_xml"][stem]
            ir_xml_path = maps["ir_xml"][stem]

            with Image.open(rgb_img_path) as img_rgb:
                rgb_size = img_rgb.size
            with Image.open(ir_img_path) as img_ir:
                ir_size = img_ir.size
            if rgb_size != ir_size:
                size_mismatch_count += 1

            rgb_w, rgb_h, rgb_boxes, rgb_stats = parse_xml(rgb_xml_path)
            ir_w, ir_h, ir_boxes, ir_stats = parse_xml(ir_xml_path)
            aggregate_parse_stats_rgb.update(rgb_stats)
            aggregate_parse_stats_ir.update(ir_stats)

            if rgb_w != ir_w or rgb_h != ir_h:
                xml_size_mismatch_count += 1

            rgb_obj_counts.append(len(rgb_boxes))
            ir_obj_counts.append(len(ir_boxes))

            if len(rgb_boxes) == len(ir_boxes):
                object_count_equal += 1
            else:
                if len(count_drift_examples) < 20:
                    count_drift_examples.append(stem)

            rgb_hist = Counter([cls_name for cls_name, _ in rgb_boxes])
            ir_hist = Counter([cls_name for cls_name, _ in ir_boxes])
            if rgb_hist == ir_hist:
                class_hist_equal += 1

            matched_total = 0
            iou_sum_total = 0.0
            for cls_name in CLASS_NAMES:
                a_boxes = [bbox for c, bbox in rgb_boxes if c == cls_name]
                b_boxes = [bbox for c, bbox in ir_boxes if c == cls_name]
                matched, iou_sum = greedy_match_iou(a_boxes, b_boxes)
                matched_total += matched
                iou_sum_total += iou_sum

            if matched_total > 0:
                mean_iou = iou_sum_total / matched_total
            else:
                mean_iou = 0.0
            matched_iou_means.append(mean_iou)

            if mean_iou < iou_warn_threshold:
                low_iou_count += 1
                if len(low_iou_examples) < 20:
                    low_iou_examples.append({"stem": stem, "mean_iou": round(mean_iou, 4)})

            if idx % 2000 == 0:
                print(f"[audit:{spec.split}] processed {idx}/{len(paired_stems)}")

        split_result.update(
            {
                "image_size_mismatch": size_mismatch_count,
                "xml_size_mismatch": xml_size_mismatch_count,
                "object_count_equal_images": object_count_equal,
                "class_hist_equal_images": class_hist_equal,
                "low_iou_images": low_iou_count,
                "mean_bbox_iou": float(np.mean(matched_iou_means)) if matched_iou_means else 0.0,
                "median_bbox_iou": float(np.median(matched_iou_means)) if matched_iou_means else 0.0,
                "rgb_bbox_count": {
                    "mean": float(np.mean(rgb_obj_counts)) if rgb_obj_counts else 0.0,
                    "median": float(np.median(rgb_obj_counts)) if rgb_obj_counts else 0.0,
                    "min": int(np.min(rgb_obj_counts)) if rgb_obj_counts else 0,
                    "max": int(np.max(rgb_obj_counts)) if rgb_obj_counts else 0,
                    "sum": int(np.sum(rgb_obj_counts)) if rgb_obj_counts else 0,
                },
                "ir_bbox_count": {
                    "mean": float(np.mean(ir_obj_counts)) if ir_obj_counts else 0.0,
                    "median": float(np.median(ir_obj_counts)) if ir_obj_counts else 0.0,
                    "min": int(np.min(ir_obj_counts)) if ir_obj_counts else 0,
                    "max": int(np.max(ir_obj_counts)) if ir_obj_counts else 0,
                    "sum": int(np.sum(ir_obj_counts)) if ir_obj_counts else 0,
                },
                "count_drift_examples": count_drift_examples,
                "low_iou_examples": low_iou_examples,
                "rgb_parse_stats": dict(aggregate_parse_stats_rgb),
                "ir_parse_stats": dict(aggregate_parse_stats_ir),
            }
        )

        summary["splits"][spec.split] = split_result

    write_json(log_root / "pair_audit.json", summary)
    return summary


def write_dataset_yaml(config_path: Path, root_path: Path) -> None:
    text_lines = [
        f"path: {root_path}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        text_lines.append(f"  {idx}: {name}")
    write_text(config_path, "\n".join(text_lines) + "\n")


def write_rgb_ir_dataset_yaml(config_path: Path, original_root: Path) -> None:
    """写出双模态训练配置，供 E5/E6 等 RGB-IR 融合实验使用。"""
    text_lines = [
        f"path: {original_root}",
        "train: rgb/images/train",
        "val: rgb/images/val",
        "test: rgb/images/test",
        "",
        "rgb:",
        f"  path: {original_root / 'rgb'}",
        "  train: images/train",
        "  val: images/val",
        "  test: images/test",
        "",
        "ir:",
        f"  path: {original_root / 'ir'}",
        "  train: images/train",
        "  val: images/val",
        "  test: images/test",
        "",
        "channels: 6",
        "",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        text_lines.append(f"  {idx}: {name}")
    write_text(config_path, "\n".join(text_lines) + "\n")


def run_convert(
    raw_root: Path,
    prepared_root: Path,
    config_root: Path,
    link_mode: str,
    crop_border: int,
) -> Dict[str, object]:
    original_root = get_original_split_root(prepared_root)
    ensure_dir(original_root)
    ensure_dir(config_root)

    summary = {
        "dataset": "DroneVehicle",
        "raw_root": str(raw_root),
        "prepared_root": str(prepared_root),
        "original_split_root": str(original_root),
        "link_mode": link_mode,
        "crop_border": crop_border,
        "splits": {},
    }

    for modality in ("rgb", "ir"):
        for split in ("train", "val", "test"):
            shutil.rmtree(original_root / modality / "images" / split, ignore_errors=True)
            shutil.rmtree(original_root / modality / "labels" / split, ignore_errors=True)
            ensure_dir(original_root / modality / "images" / split)
            ensure_dir(original_root / modality / "labels" / split)

    ensure_dir(original_root / "manifests")

    for spec in SPLIT_SPECS:
        maps = split_maps(raw_root, spec)
        # 二次划分必须保证 RGB/IR 图像与两侧 XML 标注同时存在。
        # 当前 YOLO 标签采用 RGB XML 生成并同步给 IR，IR XML 用于配对完整性约束。
        paired_stems = (
            set(maps["rgb_imgs"].keys())
            & set(maps["ir_imgs"].keys())
            & set(maps["rgb_xml"].keys())
            & set(maps["ir_xml"].keys())
        )

        split_stats = {
            "paired_stems": len(paired_stems),
            "converted_images": 0,
            "converted_labels": 0,
            "skipped_parse_error": 0,
            "class_counts": Counter(),
            "unknown_class_dropped": 0,
            "no_box_dropped": 0,
            "invalid_box_dropped": 0,
        }

        manifest_lines: List[str] = []
        for idx, stem in enumerate(sorted(paired_stems), start=1):
            rgb_img = maps["rgb_imgs"][stem]
            ir_img = maps["ir_imgs"][stem]
            rgb_xml = maps["rgb_xml"][stem]

            try:
                width, height, boxes, parse_stats = parse_xml(rgb_xml)
            except Exception:
                split_stats["skipped_parse_error"] += 1
                continue

            split_stats["unknown_class_dropped"] += parse_stats["objects_unknown_class"]
            split_stats["no_box_dropped"] += parse_stats["objects_no_box"]
            split_stats["invalid_box_dropped"] += parse_stats["objects_invalid_box"]

            lines = []
            for cls_name, bbox in boxes:
                cls_id = CLASS_TO_ID[cls_name]
                cropped_bbox = crop_bbox_to_image(bbox, width, height, crop_border)
                if cropped_bbox is None:
                    continue
                split_stats["class_counts"][cls_name] += 1
                cropped_width = width - crop_border * 2 if crop_border else width
                cropped_height = height - crop_border * 2 if crop_border else height
                lines.append(yolo_line_from_bbox(cls_id, cropped_bbox, cropped_width, cropped_height))

            rgb_img_dst = original_root / "rgb" / "images" / spec.split / f"{stem}{rgb_img.suffix.lower()}"
            ir_img_dst = original_root / "ir" / "images" / spec.split / f"{stem}{ir_img.suffix.lower()}"
            rgb_lbl_dst = original_root / "rgb" / "labels" / spec.split / f"{stem}.txt"
            ir_lbl_dst = original_root / "ir" / "labels" / spec.split / f"{stem}.txt"

            save_cropped_image(rgb_img, rgb_img_dst, crop_border, overwrite=True)
            save_cropped_image(ir_img, ir_img_dst, crop_border, overwrite=True)

            write_text(rgb_lbl_dst, "\n".join(lines) + ("\n" if lines else ""))
            write_text(ir_lbl_dst, "\n".join(lines) + ("\n" if lines else ""))

            split_stats["converted_images"] += 2
            split_stats["converted_labels"] += 2
            manifest_lines.append(stem)

            if idx % 2000 == 0:
                print(f"[convert:{spec.split}] processed {idx}/{len(paired_stems)}")

        write_text(original_root / "manifests" / f"{spec.split}.txt", "\n".join(manifest_lines) + "\n")
        split_stats["class_counts"] = dict(split_stats["class_counts"])
        summary["splits"][spec.split] = split_stats

    write_json(original_root / "conversion_summary.json", summary)

    write_dataset_yaml(config_root / "dronevehicle_resplit_rgb.yaml", original_root / "rgb")
    write_dataset_yaml(config_root / "dronevehicle_resplit_ir.yaml", original_root / "ir")
    write_rgb_ir_dataset_yaml(config_root / "dronevehicle_resplit_rgb_ir.yaml", original_root)

    return summary


def load_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    records = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
            except ValueError:
                continue
            records.append((cls_id, xc, yc, bw, bh))
    return records


def run_report(
    prepared_root: Path,
    report_root: Path,
    brightness_q_low: float,
    brightness_q_high: float,
    area_q_small: float,
    area_q_tiny: float,
    contrast_q_low: float,
    contrast_ring_scale: float,
) -> Dict[str, object]:
    ensure_dir(report_root)
    original_root = get_original_split_root(prepared_root)

    split_stats: Dict[str, Dict[str, object]] = {}
    class_total = Counter()
    bbox_w_all: List[float] = []
    bbox_h_all: List[float] = []
    bbox_area_all: List[float] = []
    objs_per_image_all: List[int] = []
    brightness_all: List[float] = []
    contrast_all: List[float] = []

    train_brightness: List[float] = []
    train_object_areas: List[float] = []
    train_object_contrasts: List[float] = []
    image_rows: List[Dict[str, object]] = []

    for split in ("train", "val", "test"):
        img_dir = original_root / "rgb" / "images" / split
        lab_dir = original_root / "rgb" / "labels" / split
        img_map = collect_stem_to_file(img_dir, IMAGE_SUFFIXES)
        stems = sorted(img_map.keys())

        split_class_counter = Counter()
        split_bbox_w: List[float] = []
        split_bbox_h: List[float] = []
        split_bbox_area: List[float] = []
        split_object_areas: List[float] = []
        split_object_contrasts: List[float] = []
        split_brightness: List[float] = []
        split_objs_per_image: List[int] = []

        for idx, stem in enumerate(stems, start=1):
            img_path = img_map[stem]
            labels = load_yolo_labels(lab_dir / f"{stem}.txt")
            analysis = analyze_image_record(img_path, labels, contrast_ring_scale)
            width = int(analysis["width"])
            height = int(analysis["height"])
            brightness = float(analysis["brightness"])
            areas = list(analysis["area_values"])
            contrasts = list(analysis["contrast_values"])

            split_objs_per_image.append(len(labels))
            objs_per_image_all.append(len(labels))
            split_brightness.append(brightness)
            brightness_all.append(brightness)
            split_object_areas.extend(areas)
            split_object_contrasts.extend(contrasts)
            contrast_all.extend(contrasts)

            if split == "train":
                train_brightness.append(brightness)
                train_object_areas.extend(areas)
                train_object_contrasts.extend(contrasts)

            for cls_id, xc, yc, bw, bh in labels:
                cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
                split_class_counter[cls_name] += 1
                class_total[cls_name] += 1

                bw_px = bw * width
                bh_px = bh * height
                area_px = bw_px * bh_px
                split_bbox_w.append(bw_px)
                split_bbox_h.append(bh_px)
                split_bbox_area.append(area_px)
                bbox_w_all.append(bw_px)
                bbox_h_all.append(bh_px)
                bbox_area_all.append(area_px)

            image_rows.append(
                {
                    "split": split,
                    "stem": stem,
                    "brightness": brightness,
                    "object_count": len(labels),
                    "min_object_area": float(min(areas)) if areas else 0.0,
                    "mean_object_area": float(np.mean(areas)) if areas else 0.0,
                    "min_object_contrast": float(min(contrasts)) if contrasts else 0.0,
                    "mean_object_contrast": float(np.mean(contrasts)) if contrasts else 0.0,
                    "width": width,
                    "height": height,
                    "_areas": areas,
                    "_contrasts": contrasts,
                }
            )

            if idx % 2000 == 0:
                print(f"[report:{split}] processed {idx}/{len(stems)}")

        split_stats[split] = {
            "images": len(stems),
            "objects": int(sum(split_objs_per_image)),
            "class_counts": dict(split_class_counter),
            "avg_objects_per_image": float(np.mean(split_objs_per_image)) if split_objs_per_image else 0.0,
            "median_objects_per_image": float(np.median(split_objs_per_image)) if split_objs_per_image else 0.0,
            "bbox_w_px": {
                "mean": float(np.mean(split_bbox_w)) if split_bbox_w else 0.0,
                "median": float(np.median(split_bbox_w)) if split_bbox_w else 0.0,
            },
            "bbox_h_px": {
                "mean": float(np.mean(split_bbox_h)) if split_bbox_h else 0.0,
                "median": float(np.median(split_bbox_h)) if split_bbox_h else 0.0,
            },
            "bbox_area_px": {
                "mean": float(np.mean(split_bbox_area)) if split_bbox_area else 0.0,
                "median": float(np.median(split_bbox_area)) if split_bbox_area else 0.0,
            },
            "brightness": {
                "mean": float(np.mean(split_brightness)) if split_brightness else 0.0,
                "median": float(np.median(split_brightness)) if split_brightness else 0.0,
                "q10": float(np.quantile(split_brightness, 0.10)) if split_brightness else 0.0,
                "q90": float(np.quantile(split_brightness, 0.90)) if split_brightness else 0.0,
            },
            "object_area": {
                "mean": float(np.mean(split_object_areas)) if split_object_areas else 0.0,
                "median": float(np.median(split_object_areas)) if split_object_areas else 0.0,
                "q25": float(np.quantile(split_object_areas, 0.25)) if split_object_areas else 0.0,
                "q75": float(np.quantile(split_object_areas, 0.75)) if split_object_areas else 0.0,
            },
            "object_contrast": {
                "mean": float(np.mean(split_object_contrasts)) if split_object_contrasts else 0.0,
                "median": float(np.median(split_object_contrasts)) if split_object_contrasts else 0.0,
                "q25": float(np.quantile(split_object_contrasts, 0.25)) if split_object_contrasts else 0.0,
            },
        }

    train_brightness_values = np.array(train_brightness, dtype=np.float32)
    train_area_values = np.array(train_object_areas, dtype=np.float32)
    train_contrast_values = np.array(train_object_contrasts, dtype=np.float32)

    dark_thr = float(np.quantile(train_brightness_values, brightness_q_low)) if len(train_brightness_values) else 0.0
    bright_thr = float(np.quantile(train_brightness_values, brightness_q_high)) if len(train_brightness_values) else 255.0
    small_thr = float(np.quantile(train_area_values, area_q_small)) if len(train_area_values) else 0.0
    tiny_thr = float(np.quantile(train_area_values, area_q_tiny)) if len(train_area_values) else 0.0
    medium_thr = float(np.quantile(train_area_values, 0.75)) if len(train_area_values) else 0.0
    low_contrast_thr = float(np.quantile(train_contrast_values, contrast_q_low)) if len(train_contrast_values) else 0.0

    train_bucket_counts = Counter()
    train_dark_size_counter = Counter()
    train_dark_image_count = 0
    train_dark_object_count = 0

    for row in image_rows:
        if row["split"] != "train":
            continue
        brightness = float(row["brightness"])
        areas = list(row["_areas"])
        if brightness <= dark_thr:
            bucket = "dark"
            train_dark_image_count += 1
            train_dark_object_count += len(areas)
            for area in areas:
                train_dark_size_counter[size_bucket(float(area), small_thr, medium_thr)] += 1
        elif brightness >= bright_thr:
            bucket = "bright"
        else:
            bucket = "normal"
        train_bucket_counts[bucket] += 1

    for row in image_rows:
        row.pop("_areas", None)
        row.pop("_contrasts", None)

    crop_width = None
    crop_height = None
    if image_rows:
        sample_width = int(image_rows[0]["width"])
        sample_height = int(image_rows[0]["height"])
        crop_width = sample_width
        crop_height = sample_height

    train_thresholds = {
        "threshold_source": "train split only",
        "crop_border": CROP_BORDER_PX,
        "cropped_image_size": {
            "width": crop_width,
            "height": crop_height,
        },
        "brightness_dark_threshold": dark_thr,
        "brightness_bright_threshold": bright_thr,
        "object_area_small_threshold": small_thr,
        "object_area_tiny_threshold": tiny_thr,
        "object_area_medium_threshold": medium_thr,
        "low_contrast_threshold": low_contrast_thr,
        "brightness_quantiles": {
            "dark_q": brightness_q_low,
            "bright_q": brightness_q_high,
        },
        "object_area_quantiles": {
            "small_q": area_q_small,
            "tiny_q": area_q_tiny,
            "medium_q": 0.75,
        },
        "contrast_quantiles": {
            "low_q": contrast_q_low,
        },
        "contrast_ring_scale": contrast_ring_scale,
        "summary": {
            "train_images": int(len(train_brightness_values)),
            "train_objects": int(len(train_area_values)),
            "train_dark_images": int(train_dark_image_count),
            "train_dark_objects": int(train_dark_object_count),
        },
    }

    visualizations = render_report_figures(
        report_root=report_root,
        class_total=class_total,
        objs_per_image_all=objs_per_image_all,
        bbox_area_all=bbox_area_all,
        brightness_all=brightness_all,
        dark_thr=dark_thr,
        bright_thr=bright_thr,
        val_bucket_counts=train_bucket_counts,
        dark_size_counter=train_dark_size_counter,
        split_stats=split_stats,
    )

    report = {
        "dataset": "DroneVehicle",
        "prepared_root": str(prepared_root),
        "original_split_root": str(original_root),
        "report_root": str(report_root),
        "protocol": "DroneVehicle-DarkSmall-v1",
        "class_names": CLASS_NAMES,
        "train_thresholds": train_thresholds,
        "overall": {
            "objects_total": int(sum(class_total.values())),
            "class_counts": dict(class_total),
            "avg_objects_per_image": float(np.mean(objs_per_image_all)) if objs_per_image_all else 0.0,
            "bbox_w_px": {
                "mean": float(np.mean(bbox_w_all)) if bbox_w_all else 0.0,
                "median": float(np.median(bbox_w_all)) if bbox_w_all else 0.0,
            },
            "bbox_h_px": {
                "mean": float(np.mean(bbox_h_all)) if bbox_h_all else 0.0,
                "median": float(np.median(bbox_h_all)) if bbox_h_all else 0.0,
            },
            "bbox_area_px": {
                "mean": float(np.mean(bbox_area_all)) if bbox_area_all else 0.0,
                "median": float(np.median(bbox_area_all)) if bbox_area_all else 0.0,
                "q10": float(np.quantile(bbox_area_all, 0.1)) if bbox_area_all else 0.0,
                "q90": float(np.quantile(bbox_area_all, 0.9)) if bbox_area_all else 0.0,
            },
            "contrast": {
                "mean": float(np.mean(contrast_all)) if contrast_all else 0.0,
                "median": float(np.median(contrast_all)) if contrast_all else 0.0,
            },
        },
        "splits": split_stats,
        "visualizations": visualizations,
    }

    write_json(report_root / "data_report.json", report)
    write_json(report_root / "train_thresholds.json", train_thresholds)
    write_csv_rows(
        report_root / "image_metadata.csv",
        [
            "split",
            "stem",
            "brightness",
            "object_count",
            "min_object_area",
            "mean_object_area",
            "min_object_contrast",
            "mean_object_contrast",
            "width",
            "height",
        ],
        image_rows,
    )

    md = [
        "# DroneVehicle Data Report",
        "",
        "## Protocol",
        "",
        "- protocol: DroneVehicle-DarkSmall-v1",
        f"- crop border: {CROP_BORDER_PX}px",
        "- thresholds are learned from train split only",
        "",
        "## Train Thresholds",
        "",
        f"- brightness dark threshold: {dark_thr:.6f}",
        f"- object area small threshold: {small_thr:.6f}",
        f"- object area tiny threshold: {tiny_thr:.6f}",
        f"- low-contrast threshold: {low_contrast_thr:.6f}",
        "",
        "## Summary",
        "",
        f"- total objects: {report['overall']['objects_total']}",
        f"- avg objects per image: {report['overall']['avg_objects_per_image']:.4f}",
        f"- train dark images: {train_dark_image_count}",
        f"- train dark objects: {train_dark_object_count}",
        "",
        "## Metrics Protocol",
        "",
        "Report these metrics in every experiment:",
        "- mAP50",
        "- mAP50-95",
        "- Recall",
        "- Precision",
        "- AP_small",
        "- Recall_small",
        "- AP_dark",
        "- Recall_dark",
        "- AP_dark-small",
        "- FPS / Params / FLOPs",
        "",
        "Priority metrics for method direction:",
        "- AP_dark",
        "- Recall_small",
        "- AP_dark-small",
        "",
    ]
    write_text(report_root / "data_report.md", "\n".join(md) + "\n")
    write_text(report_root / "protocol_summary.md", "\n".join(md) + "\n")
    return report


def filtered_lines_by_size(
    lines: List[Tuple[int, float, float, float, float]],
    width: int,
    height: int,
    target: str,
    small_thr: float,
    medium_thr: float,
) -> List[str]:
    kept = []
    for cls_id, xc, yc, bw, bh in lines:
        area_px = (bw * width) * (bh * height)
        bucket = size_bucket(area_px, small_thr, medium_thr)
        if bucket == target:
            kept.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return kept


def load_image_metadata_csv(metadata_csv: Path) -> Dict[str, Dict[str, Dict[str, object]]]:
    rows: Dict[str, Dict[str, Dict[str, object]]] = {"train": {}, "val": {}, "test": {}}
    if not metadata_csv.exists():
        return rows

    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = str(row.get("split", "")).strip()
            stem = str(row.get("stem", "")).strip()
            if split not in rows or not stem:
                continue
            rows[split][stem] = {
                "brightness": float(row.get("brightness", 0.0) or 0.0),
                "object_count": int(float(row.get("object_count", 0) or 0)),
                "min_object_area": float(row.get("min_object_area", 0.0) or 0.0),
                "mean_object_area": float(row.get("mean_object_area", 0.0) or 0.0),
                "min_object_contrast": float(row.get("min_object_contrast", 0.0) or 0.0),
                "mean_object_contrast": float(row.get("mean_object_contrast", 0.0) or 0.0),
                "width": int(float(row.get("width", 0) or 0)),
                "height": int(float(row.get("height", 0) or 0)),
            }
    return rows


def run_subsets(
    prepared_root: Path,
    report_root: Path,
    config_root: Path,
    keep_empty: bool,
    link_mode: str,
) -> Dict[str, object]:
    original_root = get_original_split_root(prepared_root)
    resplit_root = get_resplit_subset_root(prepared_root)
    metadata_csv = report_root / "image_metadata.csv"
    thresholds_path = report_root / "train_thresholds.json"
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Missing {metadata_csv}. Run 'report' before 'subsets' or use command 'all'."
        )
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"Missing {thresholds_path}. Run 'report' before 'subsets' or use command 'all'."
        )

    with thresholds_path.open("r", encoding="utf-8") as f:
        thresholds = json.load(f)
    if not isinstance(thresholds, dict):
        raise RuntimeError("train_thresholds.json is invalid")

    meta = load_image_metadata_csv(metadata_csv)
    if not meta["train"]:
        raise RuntimeError("image metadata is empty")

    dark_thr = float(thresholds.get("brightness_dark_threshold", 0.0))
    small_thr = float(thresholds.get("object_area_small_threshold", 0.0))
    tiny_thr = float(thresholds.get("object_area_tiny_threshold", 0.0))
    low_contrast_thr = float(thresholds.get("low_contrast_threshold", 0.0))
    crop_border = int(thresholds.get("crop_border", CROP_BORDER_PX))

    subset_defs = {
        "dark": lambda row: float(row["brightness"]) <= dark_thr,
        "small": lambda row: float(row["min_object_area"]) <= small_thr,
        "tiny": lambda row: float(row["min_object_area"]) <= tiny_thr,
        "dark-small": lambda row: float(row["brightness"]) <= dark_thr and float(row["min_object_area"]) <= small_thr,
        "low-contrast": lambda row: float(row["min_object_contrast"]) <= low_contrast_thr,
    }

    subset_summary = {
        "protocol": "DroneVehicle-DarkSmall-v1",
        "prepared_root": str(prepared_root),
        "original_split_root": str(original_root),
        "resplit_subset_root": str(resplit_root),
        "keep_empty_images": keep_empty,
        "link_mode": link_mode,
        "crop_border": crop_border,
        "thresholds": thresholds,
        "subsets": {},
    }
    subset_counts_rows: List[Dict[str, object]] = []

    subset_config_dir = config_root / "subsets"
    ensure_dir(subset_config_dir)

    def _reset_root(root: Path) -> None:
        shutil.rmtree(root, ignore_errors=True)
        ensure_dir(root)

    for subset_name, predicate in subset_defs.items():
        subset_summary["subsets"][subset_name] = {}

        rgb_root = resplit_root / "rgb" / subset_name
        ir_root = resplit_root / "ir" / subset_name
        rgb_ir_root = resplit_root / "rgb_ir" / subset_name
        _reset_root(rgb_root)
        _reset_root(ir_root)
        _reset_root(rgb_ir_root)

        for split in ("train", "val", "test"):
            rgb_src_img_dir = original_root / "rgb" / "images" / split
            rgb_src_lab_dir = original_root / "rgb" / "labels" / split
            ir_src_img_dir = original_root / "ir" / "images" / split
            ir_src_lab_dir = original_root / "ir" / "labels" / split

            rgb_src_imgs = collect_stem_to_file(rgb_src_img_dir, IMAGE_SUFFIXES)
            ir_src_imgs = collect_stem_to_file(ir_src_img_dir, IMAGE_SUFFIXES)

            selected_stems = [stem for stem, row in meta[split].items() if predicate(row)]
            selected_stems = sorted(selected_stems)

            rgb_img_written = 0
            rgb_obj_written = 0
            rgb_label_written = 0
            ir_img_written = 0
            ir_obj_written = 0
            ir_label_written = 0
            paired_img_written = 0
            paired_obj_written = 0
            paired_label_written = 0

            for stem in selected_stems:
                rgb_img_path = rgb_src_imgs.get(stem)
                ir_img_path = ir_src_imgs.get(stem)
                if rgb_img_path is None or ir_img_path is None:
                    continue

                rgb_labels = load_yolo_labels(rgb_src_lab_dir / f"{stem}.txt")
                ir_labels = load_yolo_labels(ir_src_lab_dir / f"{stem}.txt")
                if not keep_empty and (len(rgb_labels) == 0 or len(ir_labels) == 0):
                    continue

                rgb_img_dst = rgb_root / "images" / split / f"{stem}{rgb_img_path.suffix.lower()}"
                rgb_lab_dst = rgb_root / "labels" / split / f"{stem}.txt"
                ir_img_dst = ir_root / "images" / split / f"{stem}{ir_img_path.suffix.lower()}"
                ir_lab_dst = ir_root / "labels" / split / f"{stem}.txt"

                safe_link_or_copy(rgb_img_path, rgb_img_dst, link_mode, overwrite=True)
                safe_link_or_copy(ir_img_path, ir_img_dst, link_mode, overwrite=True)
                write_text(rgb_lab_dst, "\n".join(
                    f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for cls_id, xc, yc, bw, bh in rgb_labels
                ) + ("\n" if rgb_labels else ""))
                write_text(ir_lab_dst, "\n".join(
                    f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for cls_id, xc, yc, bw, bh in ir_labels
                ) + ("\n" if ir_labels else ""))

                rgb_img_written += 1
                rgb_label_written += 1
                rgb_obj_written += len(rgb_labels)
                ir_img_written += 1
                ir_label_written += 1
                ir_obj_written += len(ir_labels)

                rgb_ir_rgb_img_dst = rgb_ir_root / "rgb" / "images" / split / f"{stem}{rgb_img_path.suffix.lower()}"
                rgb_ir_rgb_lab_dst = rgb_ir_root / "rgb" / "labels" / split / f"{stem}.txt"
                rgb_ir_ir_img_dst = rgb_ir_root / "ir" / "images" / split / f"{stem}{ir_img_path.suffix.lower()}"
                rgb_ir_ir_lab_dst = rgb_ir_root / "ir" / "labels" / split / f"{stem}.txt"
                safe_link_or_copy(rgb_img_path, rgb_ir_rgb_img_dst, link_mode, overwrite=True)
                safe_link_or_copy(ir_img_path, rgb_ir_ir_img_dst, link_mode, overwrite=True)
                write_text(rgb_ir_rgb_lab_dst, "\n".join(
                    f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for cls_id, xc, yc, bw, bh in rgb_labels
                ) + ("\n" if rgb_labels else ""))
                write_text(rgb_ir_ir_lab_dst, "\n".join(
                    f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for cls_id, xc, yc, bw, bh in ir_labels
                ) + ("\n" if ir_labels else ""))

                paired_img_written += 2
                paired_label_written += 2
                paired_obj_written += len(rgb_labels) + len(ir_labels)

            subset_summary["subsets"][subset_name][split] = {
                "rgb": {
                    "candidate_images": len(meta[split]),
                    "selected_images": len(selected_stems),
                    "images_written": rgb_img_written,
                    "labels_written": rgb_label_written,
                    "objects_written": rgb_obj_written,
                },
                "ir": {
                    "candidate_images": len(meta[split]),
                    "selected_images": len(selected_stems),
                    "images_written": ir_img_written,
                    "labels_written": ir_label_written,
                    "objects_written": ir_obj_written,
                },
                "rgb_ir": {
                    "candidate_images": len(meta[split]),
                    "selected_images": len(selected_stems),
                    "images_written": paired_img_written,
                    "labels_written": paired_label_written,
                    "objects_written": paired_obj_written,
                },
            }

            subset_counts_rows.extend(
                [
                    {
                        "subset": subset_name,
                        "split": split,
                        "modality": "rgb",
                        "candidate_images": len(meta[split]),
                        "selected_images": len(selected_stems),
                        "images_written": rgb_img_written,
                        "labels_written": rgb_label_written,
                        "objects_written": rgb_obj_written,
                        "dark_threshold": dark_thr,
                        "small_threshold": small_thr,
                        "tiny_threshold": tiny_thr,
                        "low_contrast_threshold": low_contrast_thr,
                    },
                    {
                        "subset": subset_name,
                        "split": split,
                        "modality": "ir",
                        "candidate_images": len(meta[split]),
                        "selected_images": len(selected_stems),
                        "images_written": ir_img_written,
                        "labels_written": ir_label_written,
                        "objects_written": ir_obj_written,
                        "dark_threshold": dark_thr,
                        "small_threshold": small_thr,
                        "tiny_threshold": tiny_thr,
                        "low_contrast_threshold": low_contrast_thr,
                    },
                    {
                        "subset": subset_name,
                        "split": split,
                        "modality": "rgb_ir",
                        "candidate_images": len(meta[split]),
                        "selected_images": len(selected_stems),
                        "images_written": paired_img_written,
                        "labels_written": paired_label_written,
                        "objects_written": paired_obj_written,
                        "dark_threshold": dark_thr,
                        "small_threshold": small_thr,
                        "tiny_threshold": tiny_thr,
                        "low_contrast_threshold": low_contrast_thr,
                    },
                ]
            )

        write_dataset_yaml(subset_config_dir / f"rgb_{subset_name}.yaml", rgb_root)
        write_dataset_yaml(subset_config_dir / f"ir_{subset_name}.yaml", ir_root)
        write_rgb_ir_dataset_yaml(subset_config_dir / f"rgb_ir_{subset_name}.yaml", rgb_ir_root)

    write_json(report_root / "subset_summary.json", subset_summary)
    write_csv_rows(
        report_root / "subset_counts.csv",
        [
            "subset",
            "split",
            "modality",
            "candidate_images",
            "selected_images",
            "images_written",
            "labels_written",
            "objects_written",
            "dark_threshold",
            "small_threshold",
            "tiny_threshold",
            "low_contrast_threshold",
        ],
        subset_counts_rows,
    )

    protocol_md = [
        "# DroneVehicle-DarkSmall-v1 Protocol Summary",
        "",
        "## Contract",
        "",
        "- keep original DroneVehicle train/val/test splits",
        "- thresholds are learned from train only",
        "- crop a 100px border before all statistics and label conversion",
        "- AP_dark-small is image-level subset evaluation",
        "- low-contrast is derived from train object contrast statistics",
        "",
        "## Train Thresholds",
        "",
        f"- brightness_dark_threshold: {dark_thr:.6f}",
        f"- object_area_small_threshold: {small_thr:.6f}",
        f"- object_area_tiny_threshold: {tiny_thr:.6f}",
        f"- low_contrast_threshold: {low_contrast_thr:.6f}",
        "",
        "## Generated Subsets",
        "",
        "- train / val / test",
        "- dark",
        "- small",
        "- tiny",
        "- dark-small",
        "- low-contrast",
        "",
        "## YAML Outputs",
        "",
        f"- {subset_config_dir / 'rgb_dark.yaml'}",
        f"- {subset_config_dir / 'ir_dark.yaml'}",
        f"- {subset_config_dir / 'rgb_ir_dark.yaml'}",
        "",
        "## Metrics",
        "",
        "Report these metrics in every experiment:",
        "- mAP50",
        "- mAP50-95",
        "- Recall",
        "- Precision",
        "- AP_small",
        "- Recall_small",
        "- AP_dark",
        "- Recall_dark",
        "- AP_dark-small",
        "- FPS / Params / FLOPs",
        "",
        "Priority for method decisions:",
        "- AP_dark",
        "- Recall_small",
        "- AP_dark-small",
        "",
    ]
    write_text(report_root / "protocol_summary.md", "\n".join(protocol_md) + "\n")
    write_text(report_root / "evaluation_protocol.md", "\n".join(protocol_md) + "\n")

    return subset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DroneVehicle dataset-resplit tooling for audit, conversion, reports, and subsets."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["audit", "convert", "report", "subsets", "all"],
        help="Which pipeline stage to run. Defaults to all when omitted.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/mnt/disk2/lhr/VSD/data/DroneVehicle/raw"),
    )
    parser.add_argument(
        "--prepared-root",
        "--out-root",
        dest="prepared_root",
        type=Path,
        default=Path("/mnt/disk2/lhr/VSD/prepared/dronevehicle_resplit"),
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit"),
    )
    parser.add_argument(
        "--report-root",
        "--log-root",
        dest="report_root",
        type=Path,
        default=Path("/mnt/disk2/lhr/VSD/results/dataset_audit"),
    )
    parser.add_argument("--iou-warn-threshold", type=float, default=0.90)
    parser.add_argument("--crop-border", type=int, default=CROP_BORDER_PX)
    parser.add_argument("--brightness-q-low", type=float, default=0.25)
    parser.add_argument("--brightness-q-high", type=float, default=0.75)
    parser.add_argument("--area-q-small", type=float, default=0.25)
    parser.add_argument("--area-q-tiny", type=float, default=0.10)
    parser.add_argument("--contrast-q-low", type=float, default=0.25)
    parser.add_argument("--contrast-ring-scale", type=float, default=CONTRAST_RING_SCALE)
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="If set, subset datasets will skip images whose filtered labels are empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command in {"audit", "all"}:
        print("[stage] audit")
        run_audit(args.raw_root, args.report_root, args.iou_warn_threshold)

    if args.command in {"convert", "all"}:
        print("[stage] convert")
        run_convert(args.raw_root, args.prepared_root, args.config_root, args.link_mode, args.crop_border)

    if args.command in {"report", "all"}:
        print("[stage] report")
        run_report(
            args.prepared_root,
            args.report_root,
            args.brightness_q_low,
            args.brightness_q_high,
            args.area_q_small,
            args.area_q_tiny,
            args.contrast_q_low,
            args.contrast_ring_scale,
        )

    if args.command in {"subsets", "all"}:
        print("[stage] subsets")
        run_subsets(
            args.prepared_root,
            args.report_root,
            args.config_root,
            keep_empty=not args.drop_empty,
            link_mode=args.link_mode,
        )

    print("Done.")


if __name__ == "__main__":
    main()
