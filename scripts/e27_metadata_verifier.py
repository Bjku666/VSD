#!/usr/bin/env python3
"""E27_1 metadata verifier for cached E6 proposals.

This experiment uses train/val only.  It trains a lightweight logistic verifier
from metadata and RGB/IR crop statistics on cached E6 predictions exported at
conf=0.25 and NMS IoU=0.70.  Training negatives are restricted to background_far
proposals; class_confusion and localization_error proposals are excluded from the
negative set so they are not treated as ordinary background.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from e25_e26_offline_calibration import (  # noqa: E402
    Box,
    ROOT,
    class_names,
    iou,
    iter_images,
    load_json,
    load_split,
    map50,
    match_predictions,
    read_object_subset,
    taxonomy,
    write_csv,
)


FEATURE_NAMES = [
    "bias",
    "bbox_area_norm",
    "bbox_width_norm",
    "bbox_height_norm",
    "aspect_ratio_log",
    "conf",
    "pred_cls_norm",
    "rgb_crop_mean",
    "rgb_crop_std",
    "ir_crop_mean",
    "ir_crop_std",
    "rgb_ir_mean_abs_diff",
    "border_distance_norm",
    "is_dark",
    "is_low_contrast",
    "is_tiny",
    "is_small",
]


@dataclass(frozen=True)
class ProposalRecord:
    stem: str
    box: Box
    tags: set[str]
    label: int | None
    taxonomy: str


@dataclass(frozen=True)
class LogisticModel:
    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def rgb_to_ir_path(rgb_path: Path) -> Path:
    text = str(rgb_path)
    if "/rgb/images/" in text:
        return Path(text.replace("/rgb/images/", "/ir/images/"))
    return rgb_path


def image_maps(images: list[Path]) -> tuple[dict[str, Path], dict[str, Path]]:
    rgb = {p.stem: p for p in images}
    ir = {stem: rgb_to_ir_path(path) for stem, path in rgb.items()}
    return rgb, ir


def label_for_training(pred: Box, gt: list[Box]) -> tuple[int | None, str]:
    best_same = 0.0
    for g in gt:
        if g.cls == pred.cls:
            best_same = max(best_same, iou(pred.xyxy, g.xyxy))
    if best_same >= 0.50:
        return 1, "tp"
    label, _, _ = taxonomy(pred, gt)
    if label == "background_far":
        return 0, label
    return None, label


def collect_records(data: Any, include_unlabeled: bool) -> list[ProposalRecord]:
    records: list[ProposalRecord] = []
    for stem in sorted(data.pred):
        gt = data.gt.get(stem, [])
        tags = data.tags.get(stem, set())
        for pred in data.pred.get(stem, []):
            label, tax = label_for_training(pred, gt)
            if include_unlabeled or label is not None:
                records.append(ProposalRecord(stem=stem, box=pred, tags=tags, label=label, taxonomy=tax))
    return records


def clamp_box(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.xyxy
    ix1 = max(0, min(width - 1, int(math.floor(x1))))
    iy1 = max(0, min(height - 1, int(math.floor(y1))))
    ix2 = max(ix1 + 1, min(width, int(math.ceil(x2))))
    iy2 = max(iy1 + 1, min(height, int(math.ceil(y2))))
    return ix1, iy1, ix2, iy2


def crop_stats(image: np.ndarray, box: Box) -> tuple[float, float]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, 0.0
    return float(crop.mean() / 255.0), float(crop.std() / 255.0)


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def feature_matrix(
    records: list[ProposalRecord],
    rgb_paths: dict[str, Path],
    ir_paths: dict[str, Path],
    tiny_area: float,
    small_area: float,
) -> np.ndarray:
    rows: list[list[float]] = []
    rgb_cache: dict[str, np.ndarray] = {}
    ir_cache: dict[str, np.ndarray] = {}
    for record in records:
        rgb_path = rgb_paths[record.stem]
        ir_path = ir_paths[record.stem]
        if record.stem not in rgb_cache:
            rgb_cache[record.stem] = load_gray(rgb_path)
        if record.stem not in ir_cache:
            ir_cache[record.stem] = load_gray(ir_path)
        rgb = rgb_cache[record.stem]
        ir = ir_cache[record.stem]
        height, width = rgb.shape[:2]
        x1, y1, x2, y2 = record.box.xyxy
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        area = box_w * box_h
        rgb_mean, rgb_std = crop_stats(rgb, record.box)
        ir_mean, ir_std = crop_stats(ir, record.box)
        border = min(max(0.0, x1), max(0.0, y1), max(0.0, width - x2), max(0.0, height - y2))
        rows.append(
            [
                1.0,
                area / max(1.0, float(width * height)),
                box_w / max(1.0, float(width)),
                box_h / max(1.0, float(height)),
                math.log((box_w + 1.0) / (box_h + 1.0)),
                record.box.conf,
                record.box.cls / 4.0,
                rgb_mean,
                rgb_std,
                ir_mean,
                ir_std,
                abs(rgb_mean - ir_mean),
                border / max(1.0, float(min(width, height))),
                1.0 if "dark" in record.tags else 0.0,
                1.0 if "low-contrast" in record.tags else 0.0,
                1.0 if area <= tiny_area else 0.0,
                1.0 if area <= small_area else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def balanced_sample(records: list[ProposalRecord], max_per_class: int, seed: int) -> list[ProposalRecord]:
    rng = np.random.default_rng(seed)
    positives = [r for r in records if r.label == 1]
    negatives = [r for r in records if r.label == 0]
    if len(positives) > max_per_class:
        positives = [positives[i] for i in rng.choice(len(positives), size=max_per_class, replace=False)]
    if len(negatives) > max_per_class:
        negatives = [negatives[i] for i in rng.choice(len(negatives), size=max_per_class, replace=False)]
    sampled = positives + negatives
    order = rng.permutation(len(sampled))
    return [sampled[int(i)] for i in order]


def train_logistic(x: np.ndarray, y: np.ndarray, steps: int, lr: float, l2: float) -> LogisticModel:
    mean = x[:, 1:].mean(axis=0)
    std = x[:, 1:].std(axis=0)
    std[std < 1e-6] = 1.0
    z = x.copy()
    z[:, 1:] = (z[:, 1:] - mean) / std
    weights = np.zeros(z.shape[1], dtype=np.float64)
    for _ in range(steps):
        logits = np.clip(z @ weights, -40.0, 40.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad = (z.T @ (pred - y)) / len(y)
        grad[1:] += l2 * weights[1:]
        weights -= lr * grad
    full_mean = np.concatenate([[0.0], mean])
    full_std = np.concatenate([[1.0], std])
    return LogisticModel(weights=weights, mean=full_mean, std=full_std)


def predict_proba(model: LogisticModel, x: np.ndarray) -> np.ndarray:
    z = (x - model.mean) / model.std
    z[:, 0] = 1.0
    logits = np.clip(z @ model.weights, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def auc_score(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def score_records(records: list[ProposalRecord], p_object: np.ndarray) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = {}
    per_stem_index: dict[str, int] = {}
    for record, prob in zip(records, p_object):
        idx = per_stem_index.get(record.stem, 0)
        per_stem_index[record.stem] = idx + 1
        scores[(record.stem, idx)] = float(record.box.conf * prob)
    return scores


def evaluate_with_scores(data: Any, final_scores: dict[tuple[str, int], float], threshold: float) -> dict[str, Any]:
    def keep_factory(stem: str) -> Callable[[Box, set[str]], bool]:
        counter = {"idx": 0}

        def keep(_box: Box, _tags: set[str]) -> bool:
            idx = counter["idx"]
            counter["idx"] += 1
            return final_scores.get((stem, idx), 0.0) >= threshold

        return keep

    total_gt = 0
    total_pred = 0
    tp = 0
    fp = 0
    fp_taxonomy: dict[str, int] = {
        "class_confusion": 0,
        "background_far": 0,
        "localization_error": 0,
        "duplicate_or_conf_threshold": 0,
        "near_object_background": 0,
    }
    dark_fp = 0
    low_contrast_fp = 0
    dark_images = 0
    low_contrast_images = 0
    for stem in sorted(data.gt):
        gt = data.gt.get(stem, [])
        tags = data.tags.get(stem, set())
        total_gt += len(gt)
        if "dark" in tags:
            dark_images += 1
        if "low-contrast" in tags:
            low_contrast_images += 1
        preds = []
        for idx, pred in enumerate(data.pred.get(stem, [])):
            score = final_scores.get((stem, idx), 0.0)
            if score >= threshold:
                preds.append((score, pred))
        preds.sort(key=lambda item: item[0], reverse=True)
        total_pred += len(preds)
        used: set[int] = set()
        for _, pred in preds:
            best_i = -1
            best_iou = 0.0
            for gt_idx, gt_box in enumerate(gt):
                if gt_idx in used or gt_box.cls != pred.cls:
                    continue
                val = iou(pred.xyxy, gt_box.xyxy)
                if val >= 0.50 and val > best_iou:
                    best_iou = val
                    best_i = gt_idx
            if best_i >= 0:
                used.add(best_i)
                tp += 1
            else:
                fp += 1
                label, _, _ = taxonomy(pred, gt)
                fp_taxonomy[label] = fp_taxonomy.get(label, 0) + 1
                if "dark" in tags:
                    dark_fp += 1
                if "low-contrast" in tags:
                    low_contrast_fp += 1
    precision = tp / total_pred if total_pred else 0.0
    recall = tp / total_gt if total_gt else 0.0
    return {
        "images": len(data.gt),
        "gt": total_gt,
        "pred": total_pred,
        "tp": tp,
        "fp": fp,
        "fn": total_gt - tp,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fp_per_image": fp / len(data.gt) if data.gt else 0.0,
        "fppi_dark": dark_fp / dark_images if dark_images else 0.0,
        "fppi_low_contrast": low_contrast_fp / low_contrast_images if low_contrast_images else 0.0,
        "class_confusion_fp": fp_taxonomy.get("class_confusion", 0),
        "background_far_fp": fp_taxonomy.get("background_far", 0),
        "localization_error_fp": fp_taxonomy.get("localization_error", 0),
    }


def map50_with_scores(data: Any, final_scores: dict[tuple[str, int], float]) -> float:
    names = class_names(ROOT / "configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml")
    ap_values: list[float] = []
    for cls_id in names:
        total_gt = sum(1 for boxes in data.gt.values() for g in boxes if g.cls == cls_id)
        if total_gt == 0:
            continue
        candidates: list[tuple[float, str, Box]] = []
        for stem in sorted(data.pred):
            for idx, pred in enumerate(data.pred.get(stem, [])):
                if pred.cls == cls_id:
                    candidates.append((final_scores.get((stem, idx), 0.0), stem, pred))
        candidates.sort(key=lambda item: item[0], reverse=True)
        used: dict[str, set[int]] = {}
        tp_flags: list[int] = []
        fp_flags: list[int] = []
        for score, stem, pred in candidates:
            if score <= 0:
                continue
            used.setdefault(stem, set())
            best_i = -1
            best_iou = 0.0
            for gt_idx, gt_box in enumerate(data.gt.get(stem, [])):
                if gt_idx in used[stem] or gt_box.cls != cls_id:
                    continue
                val = iou(pred.xyxy, gt_box.xyxy)
                if val >= 0.50 and val > best_iou:
                    best_iou = val
                    best_i = gt_idx
            if best_i >= 0:
                used[stem].add(best_i)
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)
        if not tp_flags:
            ap_values.append(0.0)
            continue
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
        for threshold in [x / 100 for x in range(101)]:
            prec = max((p for p, r in zip(precisions, recalls) if r >= threshold), default=0.0)
            ap += prec / 101.0
        ap_values.append(ap)
    return sum(ap_values) / len(ap_values) if ap_values else 0.0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_split("train", ROOT / "results/val/e20_train_for_e22_hn/predictions/E6/labels")
    val = load_split("val", ROOT / "results/val/e20_0_error_delta_analysis/predictions/E6/labels")
    thresholds = load_json(ROOT / "results/dataset_audit/train_thresholds.json")
    tiny_area = float(thresholds.get("object_area_tiny_threshold", 880.0))
    small_area = float(thresholds.get("object_area_small_threshold", 1288.0))
    train_rgb, train_ir = image_maps(train.images)
    val_rgb, val_ir = image_maps(val.images)

    train_records_all = collect_records(train, include_unlabeled=False)
    train_records = balanced_sample(train_records_all, max_per_class=args.max_per_class, seed=args.seed)
    y = np.asarray([float(r.label) for r in train_records], dtype=np.float64)
    x = feature_matrix(train_records, train_rgb, train_ir, tiny_area, small_area)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(y))
    split = int(len(order) * 0.8)
    fit_idx = order[:split]
    hold_idx = order[split:]
    model = train_logistic(x[fit_idx], y[fit_idx], steps=args.steps, lr=args.lr, l2=args.l2)
    hold_prob = predict_proba(model, x[hold_idx])
    hold_pred = hold_prob >= 0.5
    hold_y = y[hold_idx]
    hold_precision = float(((hold_pred == 1) & (hold_y == 1)).sum() / max(1, (hold_pred == 1).sum()))
    hold_recall = float(((hold_pred == 1) & (hold_y == 1)).sum() / max(1, (hold_y == 1).sum()))
    hold_auc = auc_score(hold_y, hold_prob)

    val_records = collect_records(val, include_unlabeled=True)
    val_x = feature_matrix(val_records, val_rgb, val_ir, tiny_area, small_area)
    val_prob = predict_proba(model, val_x)
    final_scores = score_records(val_records, val_prob)

    baseline = match_predictions(val, lambda p, _tags: p.conf >= 0.25)
    rows: list[dict[str, Any]] = []
    for threshold in [x / 100 for x in range(1, 81)]:
        metrics = evaluate_with_scores(val, final_scores, threshold)
        rows.append({"score_threshold": threshold, **metrics})
    rows.sort(key=lambda row: (-float(row["f1"]), float(row["fp_per_image"])))
    best = rows[0]
    low_fp = min((r for r in rows if float(r["recall"]) >= 0.90), key=lambda r: float(r["fp_per_image"]), default=min(rows, key=lambda r: float(r["fp_per_image"])))
    high_recall = max(rows, key=lambda r: (float(r["recall"]), -float(r["fp_per_image"])))
    write_csv(out_dir / "calibration_grid.csv", rows)

    val_map50_final = map50_with_scores(val, final_scores)
    dark_small_stems, dark_small_gt = read_object_subset("dark-small_object")
    object_proxy_data = type("SubsetData", (), {})()
    object_proxy_data.gt = dark_small_gt
    object_proxy_data.pred = {stem: val.pred.get(stem, []) for stem in dark_small_stems}
    object_proxy_data.tags = {stem: val.tags.get(stem, set()) for stem in dark_small_stems}
    object_map50_final = map50_with_scores(object_proxy_data, final_scores)

    train_counts = {
        "available_positive_tp": sum(1 for r in train_records_all if r.label == 1),
        "available_negative_background_far": sum(1 for r in train_records_all if r.label == 0),
        "sampled_positive_tp": int((y == 1).sum()),
        "sampled_negative_background_far": int((y == 0).sum()),
        "excluded_non_background_fp": sum(1 for stem in train.pred for pred in train.pred[stem] if label_for_training(pred, train.gt.get(stem, []))[0] is None),
    }
    payload = {
        "experiment": "E27_1",
        "source": {
            "train_predictions": str(ROOT / "results/val/e20_train_for_e22_hn/predictions/E6/labels"),
            "val_predictions": str(ROOT / "results/val/e20_0_error_delta_analysis/predictions/E6/labels"),
            "minimum_conf_in_cache": 0.25,
            "nms_iou_in_cache": 0.70,
            "negative_policy": "background_far_only; class_confusion/localization_error excluded from negative training",
        },
        "train_counts": train_counts,
        "holdout": {
            "precision_at_0.5": hold_precision,
            "recall_at_0.5": hold_recall,
            "auc": hold_auc,
        },
        "baseline_cached_conf025": {k: v for k, v in baseline.items() if k != "per_class"},
        "E27_1_balanced": best,
        "E27_1_low_fp": low_fp,
        "E27_1_high_recall": high_recall,
        "val_mAP50_cached_score_final": val_map50_final,
        "dark_small_object_mAP50_cached_score_final": object_map50_final,
    }
    write_json(out_dir / "best_operating_points.json", payload)
    write_json(
        out_dir / "feature_schema.json",
        {
            "features": FEATURE_NAMES,
            "model": "numpy logistic regression with L2 regularization",
            "score_final": "score_E6 * p_object",
        },
    )
    write_json(
        out_dir / "verifier_weights.json",
        {
            "features": FEATURE_NAMES,
            "weights": [float(x) for x in model.weights],
            "mean": [float(x) for x in model.mean],
            "std": [float(x) for x in model.std],
        },
    )
    required = {
        "Precision": best["precision"],
        "Recall": best["recall"],
        "False Positives/image": best["fp_per_image"],
        "FPPI_dark": best["fppi_dark"],
        "FPPI_low-contrast": best["fppi_low_contrast"],
        "class_confusion_fp": best["class_confusion_fp"],
        "background_far_fp": best["background_far_fp"],
        "localization_error_fp": best["localization_error_fp"],
        "mAP50_cached_score_final": val_map50_final,
        "AP50_dark-small_object_cached_score_final": object_map50_final,
        "holdout_auc": hold_auc,
    }
    write_json(out_dir / "required_metrics.json", required)
    lines = [
        "# E27_1 Metadata Verifier",
        "",
        "- Source: cached E6 train/val predictions, conf>=0.25, NMS IoU=0.70.",
        "- Training positives: train predictions matched to GT with same-class IoU>=0.50.",
        "- Training negatives: background_far only; class_confusion and localization_error are excluded from negative training.",
        f"- Train sample: TP={train_counts['sampled_positive_tp']}, background_far={train_counts['sampled_negative_background_far']}.",
        f"- Holdout AUC: {hold_auc:.6f}",
        f"- Baseline cached conf>=0.25: FP/image={baseline['fp_per_image']:.6f}, FPPI_dark={baseline['fppi_dark']:.6f}, recall={baseline['recall']:.6f}.",
        f"- Best score_final threshold={best['score_threshold']:.2f}: FP/image={best['fp_per_image']:.6f}, FPPI_dark={best['fppi_dark']:.6f}, recall={best['recall']:.6f}, class_confusion FP={best['class_confusion_fp']}.",
        f"- Cached mAP50 score_final: val={val_map50_final:.6f}, dark-small object={object_map50_final:.6f}.",
    ]
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "metrics_summary.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out-dir", default=str(ROOT / "results/val/e27_1_metadata_verifier"))
    parser.add_argument("--max-per-class", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
