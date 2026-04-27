#!/usr/bin/env python3
"""DroneVehicle 数据集重划分流水线：审计、转换与子集生成。

该脚本用于构建规范化的 RGB/IR YOLO 数据集，并生成面向 dark/small 分析的
重划分评测子集，同时将报告输出与训练日志目录分离。
"""
from __future__ import annotations

import argparse
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
from PIL import Image, ImageStat

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


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


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
    plt.title("Validation Brightness Buckets")
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
    plt.title("Dark Subset Size Distribution")
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


def run_convert(
    raw_root: Path,
    prepared_root: Path,
    config_root: Path,
    link_mode: str,
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
        "splits": {},
    }

    for modality in ("rgb", "ir"):
        for split in ("train", "val", "test"):
            ensure_dir(original_root / modality / "images" / split)
            ensure_dir(original_root / modality / "labels" / split)

    ensure_dir(original_root / "manifests")

    for spec in SPLIT_SPECS:
        maps = split_maps(raw_root, spec)
        paired_stems = set(maps["rgb_imgs"].keys()) & set(maps["ir_imgs"].keys()) & set(maps["rgb_xml"].keys())

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
                split_stats["class_counts"][cls_name] += 1
                lines.append(yolo_line_from_bbox(cls_id, bbox, width, height))

            rgb_img_dst = original_root / "rgb" / "images" / spec.split / f"{stem}{rgb_img.suffix.lower()}"
            ir_img_dst = original_root / "ir" / "images" / spec.split / f"{stem}{ir_img.suffix.lower()}"
            rgb_lbl_dst = original_root / "rgb" / "labels" / spec.split / f"{stem}.txt"
            ir_lbl_dst = original_root / "ir" / "labels" / spec.split / f"{stem}.txt"

            safe_link_or_copy(rgb_img, rgb_img_dst, link_mode)
            safe_link_or_copy(ir_img, ir_img_dst, link_mode)

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


def calc_brightness(img_path: Path) -> float:
    with Image.open(img_path) as img:
        gray = img.convert("L")
        return float(ImageStat.Stat(gray).mean[0])


def run_report(
    prepared_root: Path,
    report_root: Path,
    brightness_q_low: float,
    brightness_q_high: float,
    small_px: float,
    medium_px: float,
) -> Dict[str, object]:
    ensure_dir(report_root)
    original_root = get_original_split_root(prepared_root)

    small_thr = float(small_px * small_px)
    medium_thr = float(medium_px * medium_px)

    split_stats = {}
    class_total = Counter()
    bbox_w_all: List[float] = []
    bbox_h_all: List[float] = []
    bbox_area_all: List[float] = []
    objs_per_image_all: List[int] = []
    brightness_all: List[float] = []

    val_brightness_by_stem: Dict[str, float] = {}
    val_obj_by_stem: Dict[str, Dict[str, int]] = {}

    for split in ("train", "val", "test"):
        img_dir = original_root / "rgb" / "images" / split
        lab_dir = original_root / "rgb" / "labels" / split
        img_map = collect_stem_to_file(img_dir, IMAGE_SUFFIXES)
        stems = sorted(img_map.keys())

        split_class_counter = Counter()
        split_sizes = Counter()
        split_bbox_w: List[float] = []
        split_bbox_h: List[float] = []
        split_bbox_area: List[float] = []
        split_objs_per_image: List[int] = []
        split_brightness: List[float] = []

        for idx, stem in enumerate(stems, start=1):
            img_path = img_map[stem]
            with Image.open(img_path) as img:
                w, h = img.size

            labels = load_yolo_labels(lab_dir / f"{stem}.txt")
            split_objs_per_image.append(len(labels))
            objs_per_image_all.append(len(labels))

            size_counter_for_image = Counter()
            for cls_id, _xc, _yc, bw, bh in labels:
                cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
                split_class_counter[cls_name] += 1
                class_total[cls_name] += 1

                bw_px = bw * w
                bh_px = bh * h
                area_px = bw_px * bh_px
                bucket = size_bucket(area_px, small_thr, medium_thr)
                split_sizes[bucket] += 1
                size_counter_for_image[bucket] += 1

                split_bbox_w.append(bw_px)
                split_bbox_h.append(bh_px)
                split_bbox_area.append(area_px)
                bbox_w_all.append(bw_px)
                bbox_h_all.append(bh_px)
                bbox_area_all.append(area_px)

            brightness = calc_brightness(img_path)
            split_brightness.append(brightness)
            brightness_all.append(brightness)
            if split == "val":
                val_brightness_by_stem[stem] = brightness
                val_obj_by_stem[stem] = {
                    "small": size_counter_for_image["small"],
                    "medium": size_counter_for_image["medium"],
                    "large": size_counter_for_image["large"],
                    "total": len(labels),
                }

            if idx % 2000 == 0:
                print(f"[report:{split}] processed {idx}/{len(stems)}")

        split_stats[split] = {
            "images": len(stems),
            "objects": int(sum(split_objs_per_image)),
            "class_counts": dict(split_class_counter),
            "avg_objects_per_image": float(np.mean(split_objs_per_image)) if split_objs_per_image else 0.0,
            "median_objects_per_image": float(np.median(split_objs_per_image)) if split_objs_per_image else 0.0,
            "size_buckets": dict(split_sizes),
            "brightness": {
                "mean": float(np.mean(split_brightness)) if split_brightness else 0.0,
                "median": float(np.median(split_brightness)) if split_brightness else 0.0,
                "q10": float(np.quantile(split_brightness, 0.10)) if split_brightness else 0.0,
                "q90": float(np.quantile(split_brightness, 0.90)) if split_brightness else 0.0,
            },
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
        }

    val_brightness_values = np.array(list(val_brightness_by_stem.values()), dtype=np.float32)
    dark_thr = float(np.quantile(val_brightness_values, brightness_q_low)) if len(val_brightness_values) else 0.0
    bright_thr = float(np.quantile(val_brightness_values, brightness_q_high)) if len(val_brightness_values) else 255.0

    val_bucket_counts = Counter()
    dark_size_counter = Counter()
    dark_image_count = 0
    dark_obj_total = 0

    metadata_lines = [
        "stem,brightness,bucket,objects_total,objects_small,objects_medium,objects_large"
    ]

    for stem in sorted(val_brightness_by_stem.keys()):
        b = val_brightness_by_stem[stem]
        if b <= dark_thr:
            bucket = "dark"
        elif b >= bright_thr:
            bucket = "bright"
        else:
            bucket = "normal"
        val_bucket_counts[bucket] += 1

        obj_info = val_obj_by_stem[stem]
        metadata_lines.append(
            f"{stem},{b:.4f},{bucket},{obj_info['total']},{obj_info['small']},{obj_info['medium']},{obj_info['large']}"
        )

        if bucket == "dark":
            dark_image_count += 1
            dark_obj_total += obj_info["total"]
            dark_size_counter["small"] += obj_info["small"]
            dark_size_counter["medium"] += obj_info["medium"]
            dark_size_counter["large"] += obj_info["large"]

    dark_avg_obj = (dark_obj_total / dark_image_count) if dark_image_count > 0 else 0.0

    visualizations = render_report_figures(
        report_root=report_root,
        class_total=class_total,
        objs_per_image_all=objs_per_image_all,
        bbox_area_all=bbox_area_all,
        brightness_all=brightness_all,
        dark_thr=dark_thr,
        bright_thr=bright_thr,
        val_bucket_counts=val_bucket_counts,
        dark_size_counter=dark_size_counter,
        split_stats=split_stats,
    )

    report = {
        "dataset": "DroneVehicle",
        "prepared_root": str(prepared_root),
        "original_split_root": str(original_root),
        "report_root": str(report_root),
        "class_names": CLASS_NAMES,
        "size_thresholds_px": {
            "small_lt": small_px,
            "medium_lt": medium_px,
        },
        "brightness_thresholds": {
            "dark_max": dark_thr,
            "bright_min": bright_thr,
            "q_low": brightness_q_low,
            "q_high": brightness_q_high,
        },
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
        },
        "splits": split_stats,
        "val_brightness_buckets": dict(val_bucket_counts),
        "dark_subset": {
            "images": dark_image_count,
            "avg_objects_per_image": dark_avg_obj,
            "size_distribution": dict(dark_size_counter),
        },
        "visualizations": visualizations,
    }

    write_json(report_root / "data_report.json", report)
    write_text(report_root / "val_image_metadata.csv", "\n".join(metadata_lines) + "\n")

    md = [
        "# DroneVehicle Data Report",
        "",
        "## Key Statistics",
        "",
        f"- total objects: {report['overall']['objects_total']}",
        f"- avg objects per image: {report['overall']['avg_objects_per_image']:.4f}",
        f"- brightness thresholds: dark <= {dark_thr:.3f}, bright >= {bright_thr:.3f}",
        f"- val brightness buckets: {dict(val_bucket_counts)}",
        f"- dark subset avg objects/image: {dark_avg_obj:.4f}",
        f"- dark subset size distribution: {dict(dark_size_counter)}",
        "",
        "## Required Metrics Protocol",
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


def read_val_metadata(metadata_csv: Path) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}
    with open(metadata_csv, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return rows
        for line in f:
            line = line.strip()
            if not line:
                continue
            stem, brightness, bucket, total, small, medium, large = line.split(",")
            rows[stem] = {
                "brightness": float(brightness),
                "bucket": bucket,
                "total": int(total),
                "small": int(small),
                "medium": int(medium),
                "large": int(large),
            }
    return rows


def write_subset_yaml(path: Path, dataset_root: Path) -> None:
    lines = [
        f"path: {dataset_root}",
        "val: images/val",
        "",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    write_text(path, "\n".join(lines) + "\n")


def run_subsets(
    prepared_root: Path,
    report_root: Path,
    config_root: Path,
    small_px: float,
    medium_px: float,
    keep_empty: bool,
    link_mode: str,
) -> Dict[str, object]:
    original_root = get_original_split_root(prepared_root)
    resplit_root = get_resplit_subset_root(prepared_root)
    metadata_csv = report_root / "val_image_metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Missing {metadata_csv}. Run 'report' before 'subsets' or use command 'all'."
        )

    val_meta = read_val_metadata(metadata_csv)
    if not val_meta:
        raise RuntimeError("val metadata is empty")

    small_thr = float(small_px * small_px)
    medium_thr = float(medium_px * medium_px)

    all_val_stems = sorted(val_meta.keys())
    bright_stems = sorted([k for k, v in val_meta.items() if v["bucket"] == "bright"])
    normal_stems = sorted([k for k, v in val_meta.items() if v["bucket"] == "normal"])
    dark_stems = sorted([k for k, v in val_meta.items() if v["bucket"] == "dark"])

    subset_defs = {
        "bright": {"stems": bright_stems, "size_filter": None},
        "normal": {"stems": normal_stems, "size_filter": None},
        "dark": {"stems": dark_stems, "size_filter": None},
        "small": {"stems": all_val_stems, "size_filter": "small"},
        "medium": {"stems": all_val_stems, "size_filter": "medium"},
        "large": {"stems": all_val_stems, "size_filter": "large"},
        "dark-small": {"stems": dark_stems, "size_filter": "small"},
    }

    subset_summary = {
        "prepared_root": str(prepared_root),
        "original_split_root": str(original_root),
        "resplit_subset_root": str(resplit_root),
        "keep_empty_images": keep_empty,
        "link_mode": link_mode,
        "subsets": {},
    }

    subset_config_dir = config_root / "subsets"
    ensure_dir(subset_config_dir)

    for modality in ("rgb", "ir"):
        src_img_dir = original_root / modality / "images" / "val"
        src_lab_dir = original_root / modality / "labels" / "val"
        src_imgs = collect_stem_to_file(src_img_dir, IMAGE_SUFFIXES)

        for subset_name, cfg in subset_defs.items():
            stems = cfg["stems"]
            size_filter = cfg["size_filter"]

            dst_root = resplit_root / modality / subset_name
            dst_img_dir = dst_root / "images" / "val"
            dst_lab_dir = dst_root / "labels" / "val"
            ensure_dir(dst_img_dir)
            ensure_dir(dst_lab_dir)

            images_written = 0
            labels_written = 0
            objects_written = 0

            for stem in stems:
                img_path = src_imgs.get(stem)
                if img_path is None:
                    continue

                with Image.open(img_path) as img:
                    w, h = img.size

                labels = load_yolo_labels(src_lab_dir / f"{stem}.txt")
                if size_filter is None:
                    out_lines = [
                        f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                        for cls_id, xc, yc, bw, bh in labels
                    ]
                else:
                    out_lines = filtered_lines_by_size(
                        labels,
                        w,
                        h,
                        size_filter,
                        small_thr,
                        medium_thr,
                    )

                if not keep_empty and len(out_lines) == 0:
                    continue

                img_dst = dst_img_dir / f"{stem}{img_path.suffix.lower()}"
                lab_dst = dst_lab_dir / f"{stem}.txt"

                safe_link_or_copy(img_path, img_dst, link_mode)
                write_text(lab_dst, "\n".join(out_lines) + ("\n" if out_lines else ""))

                images_written += 1
                labels_written += 1
                objects_written += len(out_lines)

            subset_summary["subsets"].setdefault(subset_name, {})[modality] = {
                "candidate_stems": len(stems),
                "images_written": images_written,
                "labels_written": labels_written,
                "objects_written": objects_written,
                "size_filter": size_filter,
            }

            subset_yaml = subset_config_dir / f"{modality}_{subset_name}.yaml"
            write_subset_yaml(subset_yaml, dst_root)

    write_json(report_root / "subset_summary.json", subset_summary)

    protocol_md = [
        "# Dataset Resplit Evaluation Protocol",
        "",
        "Run the same model checkpoint on these subsets:",
        f"- full val: {config_root / 'dronevehicle_resplit_rgb.yaml'}",
        f"- dark: {subset_config_dir / 'rgb_dark.yaml'}",
        f"- small: {subset_config_dir / 'rgb_small.yaml'}",
        f"- dark-small: {subset_config_dir / 'rgb_dark-small.yaml'}",
        "",
        "Always report:",
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
        default=Path("/mnt/disk2/lhr/VSD/experiments/dronevehicle_resplit"),
    )
    parser.add_argument("--iou-warn-threshold", type=float, default=0.90)
    parser.add_argument("--brightness-q-low", type=float, default=0.33)
    parser.add_argument("--brightness-q-high", type=float, default=0.67)
    parser.add_argument("--size-small", type=float, default=32.0)
    parser.add_argument("--size-medium", type=float, default=96.0)
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
        run_convert(args.raw_root, args.prepared_root, args.config_root, args.link_mode)

    if args.command in {"report", "all"}:
        print("[stage] report")
        run_report(
            args.prepared_root,
            args.report_root,
            args.brightness_q_low,
            args.brightness_q_high,
            args.size_small,
            args.size_medium,
        )

    if args.command in {"subsets", "all"}:
        print("[stage] subsets")
        run_subsets(
            args.prepared_root,
            args.report_root,
            args.config_root,
            args.size_small,
            args.size_medium,
            keep_empty=not args.drop_empty,
            link_mode=args.link_mode,
        )

    print("Done.")


if __name__ == "__main__":
    main()
