#!/usr/bin/env python3
"""E22 hard-negative mining utilities.

E22_0 classifies false positives from E20_0 into hard-negative groups.
E22_1 turns that taxonomy into per-category training lists; only
background_far is marked train-allowed for the first hard-negative curriculum.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml: {path}")
    return data


def _split_path(data_yaml: Path, split: str) -> Path:
    cfg = _load_yaml(data_yaml)
    root = Path(str(cfg.get("path", ".")))
    entry = Path(str(cfg[split]))
    return entry if entry.is_absolute() else root / entry


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _xywhn_to_xyxy(values: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = values
    return (
        (x - w / 2.0) * width,
        (y - h / 2.0) * height,
        (x + w / 2.0) * width,
        (y + h / 2.0) * height,
    )


def _read_gt(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        rows.append(
            {
                "index": idx,
                "class_id": int(float(parts[0])),
                "xyxy": _xywhn_to_xyxy([float(x) for x in parts[1:5]], width, height),
            }
        )
    return rows


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _taxonomy(fp_cls: int, pred_xyxy: tuple[float, float, float, float], gt: list[dict[str, Any]]) -> tuple[str, float, int | None]:
    best_any = 0.0
    best_any_cls: int | None = None
    best_same = 0.0
    for g in gt:
        iou = _iou(pred_xyxy, g["xyxy"])
        if iou > best_any:
            best_any = iou
            best_any_cls = int(g["class_id"])
        if int(g["class_id"]) == fp_cls and iou > best_same:
            best_same = iou

    if best_same >= 0.50:
        return "duplicate_or_conf_threshold", best_same, fp_cls
    if best_any >= 0.30 and best_any_cls is not None and best_any_cls != fp_cls:
        return "class_confusion", best_any, best_any_cls
    if best_same >= 0.10:
        return "localization_error", best_same, fp_cls
    if best_any >= 0.10:
        return "near_object_background", best_any, best_any_cls
    return "background_far", best_any, best_any_cls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", default="taxonomy", choices=["taxonomy", "lists"], help="Run E22_0 taxonomy or E22_1 list export.")
    parser.add_argument("--e20-dir", default="/mnt/disk2/lhr/VSD/results/S5_diagnostic_optimization/e20_0_error_delta_analysis")
    parser.add_argument("--taxonomy-csv", default="/mnt/disk2/lhr/VSD/results/S5_diagnostic_optimization/e22_0_hard_negative_taxonomy/hard_negative_list.csv")
    parser.add_argument("--data-rgb-ir", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--out-dir", default="/mnt/disk2/lhr/VSD/results/S5_diagnostic_optimization/e22_0_hard_negative_taxonomy")
    return parser.parse_args()


def _run_taxonomy(args: argparse.Namespace) -> Path:
    e20_dir = Path(args.e20_dir)
    fp_path = e20_dir / "fp_by_model.csv"
    if not fp_path.exists():
        raise SystemExit(f"Missing E20 FP list: {fp_path}")

    image_dir = _split_path(Path(args.data_rgb_ir), args.split)
    image_by_stem = {p.stem: p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}

    out_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    gt_cache: dict[str, list[dict[str, Any]]] = {}

    for row in _read_csv(fp_path):
        stem = row["stem"]
        image = image_by_stem.get(stem)
        if image is None:
            continue
        if stem not in gt_cache:
            with Image.open(image) as im:
                width, height = im.size
            gt_cache[stem] = _read_gt(_label_path_for_image(image), width, height)

        fp_cls = int(row["class_id"])
        xyxy = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
        label, nearest_iou, nearest_cls = _taxonomy(fp_cls, xyxy, gt_cache[stem])
        new_row = dict(row)
        new_row["split"] = str(args.split)
        new_row["taxonomy"] = label
        new_row["nearest_gt_iou"] = f"{nearest_iou:.6f}"
        new_row["nearest_gt_class_id"] = "" if nearest_cls is None else nearest_cls
        out_rows.append(new_row)
        counts[label] += 1
        by_model[f"{row['model']}/{label}"] += 1
        by_class[f"{row['class']}/{label}"] += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "hard_negative_list.csv", out_rows)
    summary_rows = [{"taxonomy": k, "count": v} for k, v in counts.most_common()]
    _write_csv(out_dir / "taxonomy_summary.csv", summary_rows)
    summary = {
        "experiment": "E22_0",
        "source": str(fp_path),
        "false_positive_count": len(out_rows),
        "taxonomy_counts": dict(counts),
        "taxonomy_by_model": dict(by_model),
        "taxonomy_by_class": dict(by_class),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# E22_0 Hard Negative Taxonomy",
        "",
        f"- Source FP rows: {len(out_rows)}",
        "",
        "| Taxonomy | Count |",
        "| --- | ---: |",
    ]
    for item in summary_rows:
        lines.append(f"| {item['taxonomy']} | {item['count']} |")
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "metrics_summary.md")
    return out_dir / "hard_negative_list.csv"


def _tag_flags(subset_tags: str) -> dict[str, str]:
    tags = {x.strip() for x in str(subset_tags or "").split(",") if x.strip()}
    return {
        "is_dark": "1" if "dark" in tags else "0",
        "is_small": "1" if "small" in tags or "dark-small" in tags else "0",
        "is_tiny": "1" if "tiny" in tags else "0",
        "is_low_contrast": "1" if "low-contrast" in tags else "0",
    }


def _modality_for(row: dict[str, str]) -> str:
    model = str(row.get("model", ""))
    image = str(row.get("image", ""))
    if model.startswith(("E5", "E6", "E10", "E11", "E12", "E13", "E18", "E22")):
        return "rgb_ir"
    if "/ir/" in image:
        return "ir"
    return "rgb"


def _list_row(row: dict[str, str]) -> dict[str, Any]:
    flags = _tag_flags(row.get("subset_tags", ""))
    return {
        "image_path": row.get("image", ""),
        "modality": _modality_for(row),
        "pred_class": row.get("class", row.get("class_id", "")),
        "confidence": row.get("conf", ""),
        "nearest_gt_class": row.get("nearest_gt_class_id", ""),
        "iou_to_nearest_gt": row.get("nearest_gt_iou", ""),
        "error_type": row.get("taxonomy", ""),
        "split": row.get("split", ""),
        **flags,
        "model": row.get("model", ""),
        "stem": row.get("stem", ""),
        "pred_class_id": row.get("class_id", ""),
        "x1": row.get("x1", ""),
        "y1": row.get("y1", ""),
        "x2": row.get("x2", ""),
        "y2": row.get("y2", ""),
    }


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("image_path"),
            row.get("modality"),
            row.get("pred_class_id"),
            row.get("error_type"),
            row.get("model"),
            row.get("x1"),
            row.get("y1"),
            row.get("x2"),
            row.get("y2"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _run_lists(args: argparse.Namespace) -> None:
    taxonomy_csv = Path(args.taxonomy_csv)
    if not taxonomy_csv.exists():
        raise SystemExit(f"Missing E22_0 taxonomy CSV: {taxonomy_csv}")

    rows_by_type: dict[str, list[dict[str, Any]]] = {
        "background_far": [],
        "class_confusion": [],
        "localization_error": [],
        "near_object_background": [],
        "duplicate_or_conf_threshold": [],
    }
    for row in _read_csv(taxonomy_csv):
        taxonomy = row.get("taxonomy", "")
        if taxonomy not in rows_by_type:
            rows_by_type[taxonomy] = []
        rows_by_type[taxonomy].append(_list_row(row))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_path",
        "modality",
        "pred_class",
        "confidence",
        "nearest_gt_class",
        "iou_to_nearest_gt",
        "error_type",
        "split",
        "is_dark",
        "is_small",
        "is_tiny",
        "is_low_contrast",
        "model",
        "stem",
        "pred_class_id",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    summary: dict[str, Any] = {
        "experiment": "E22_1",
        "source": str(taxonomy_csv),
        "source_split": str(args.split),
        "training_use_allowed": str(args.split) == "train",
        "train_allowed_taxonomies": ["background_far"],
        "blocked_for_first_training": ["class_confusion", "localization_error", "near_object_background", "duplicate_or_conf_threshold"],
        "files": {},
        "counts": {},
    }
    for taxonomy, rows in sorted(rows_by_type.items()):
        deduped = _dedupe_rows(rows)
        filename = f"hard_negative_{taxonomy}.txt"
        _write_tsv(out_dir / filename, deduped, fields)
        summary["files"][taxonomy] = str(out_dir / filename)
        summary["counts"][taxonomy] = {"raw": len(rows), "deduped": len(deduped)}

    allowed = _dedupe_rows(rows_by_type.get("background_far", []))
    _write_tsv(out_dir / "train_allowed_background_far.txt", allowed, fields)
    summary["files"]["train_allowed_background_far"] = str(out_dir / "train_allowed_background_far.txt")
    summary["counts"]["train_allowed_background_far"] = {"raw": len(rows_by_type.get("background_far", [])), "deduped": len(allowed)}

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E22_1 Hard Negative Lists",
        "",
        "- First curriculum train-allowed taxonomy: `background_far` only.",
        "- `class_confusion` is exported for diagnosis but blocked from the first hard-negative training pass.",
        f"- Source split: `{args.split}`.",
        f"- Training use allowed: `{'yes' if summary['training_use_allowed'] else 'no'}`.",
        "",
        "| Taxonomy | Raw | Deduped | Train allowed |",
        "| --- | ---: | ---: | --- |",
    ]
    for taxonomy in sorted(rows_by_type):
        counts = summary["counts"][taxonomy]
        allowed_flag = "yes" if taxonomy == "background_far" else "no"
        lines.append(f"| {taxonomy} | {counts['raw']} | {counts['deduped']} | {allowed_flag} |")
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "metrics_summary.md")


def main() -> None:
    args = parse_args()
    if args.mode == "lists":
        _run_lists(args)
    else:
        _run_taxonomy(args)


if __name__ == "__main__":
    main()
