#!/usr/bin/env python3
"""E20_0: compare E6, E12_1 and E13_2 val detections case-by-case.

The script exports validation predictions when needed, matches predictions to
ground truth at a fixed IoU threshold, and reports:

- GT boxes detected by E6 but missed by E12_1 or E13_2.
- E6 false positives removed by E12_1 or E13_2.
- FP/TP/FN counts by class and image-level subset tags.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "") or str(Path(__file__).resolve().parent)
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from e6_feature_fusion_multiscale_core import E6DetectionTrainer
from e12_gated_fusion_core import E12DetectionTrainer
from e13_tiny_aware_loss_core import E13DetectionTrainer


MODEL_SPECS = {
    "E6": {
        "trainer": E6DetectionTrainer,
        "weights": "/mnt/disk2/lhr/VSD/results/val/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt",
    },
    "E12_1": {
        "trainer": E12DetectionTrainer,
        "weights": "/mnt/disk2/lhr/VSD/results/val/e12_1_residual_gated_fusion/weights/best.pt",
    },
    "E13_2": {
        "trainer": E13DetectionTrainer,
        "weights": "/mnt/disk2/lhr/VSD/results/val/e13_2_e6_scale_aware_loss/weights/best.pt",
    },
}


@dataclass(frozen=True)
class Box:
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float = 1.0


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


def _iter_images(image_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in suffixes)


def _class_names(data_yaml: Path) -> dict[int, str]:
    names = _load_yaml(data_yaml).get("names", {})
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    return {}


def _xywhn_to_xyxy(values: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = values
    cx = x * width
    cy = y * height
    bw = w * width
    bh = h * height
    return (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)


def _read_label_boxes(path: Path, width: int, height: int, with_conf: bool = False) -> list[Box]:
    if not path.exists():
        return []
    boxes: list[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        vals = [float(x) for x in parts[1:5]]
        conf = float(parts[5]) if with_conf and len(parts) >= 6 else 1.0
        boxes.append(Box(cls=cls, xyxy=_xywhn_to_xyxy(vals, width, height), conf=conf))
    return boxes


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


def _match(gt: list[Box], pred: list[Box], iou_thr: float) -> tuple[set[int], set[int], dict[int, int]]:
    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            if p.cls != g.cls:
                continue
            iou = _iou(p.xyxy, g.xyxy)
            if iou >= iou_thr:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    pred_to_gt: dict[int, int] = {}
    for _, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pred_to_gt[pi] = gi
    return used_g, set(range(len(pred))) - used_p, pred_to_gt


def _stem_set_from_subset(yaml_path: Path, split: str) -> set[str]:
    if not yaml_path.exists():
        return set()
    image_dir = _split_path(yaml_path, split)
    return {p.stem for p in _iter_images(image_dir)}


def _subset_tags(stem: str, subsets: dict[str, set[str]]) -> list[str]:
    return [name for name, stems in subsets.items() if stem in stems]


def _export_predictions(args: argparse.Namespace, model_id: str, spec: dict[str, Any]) -> Path:
    labels_dir = Path(args.out_dir) / "predictions" / model_id / "labels"
    if labels_dir.exists() and any(labels_dir.glob("*.txt")) and not args.force_predict:
        return labels_dir

    run_project = Path(args.out_dir) / "predictions"
    run_name = model_id
    if args.force_predict and (run_project / run_name).exists():
        shutil.rmtree(run_project / run_name)

    overrides = {
        "task": "detect",
        "mode": "train",
        "model": str(Path(spec["weights"])),
        "data": str(Path(args.data_rgb_ir)),
        "epochs": 1,
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "device": args.device,
        "project": str(run_project),
        "name": run_name,
        "resume": False,
        "plots": False,
        "exist_ok": True,
        "save_txt": True,
        "save_conf": True,
        "conf": float(args.conf),
        "iou": float(args.nms_iou),
    }

    trainer = spec["trainer"](overrides=overrides)
    trainer.set_fusion_mode("rgb_ir")
    trainer.set_ir_data(str(Path(args.data_ir)))
    if model_id == "E13_2" and hasattr(trainer, "set_loss_config"):
        trainer.set_loss_config(loss_mode="scale-aware")
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device).float().eval()
    split_key = args.split if args.split in trainer.data else "val"
    trainer.test_loader = trainer.get_dataloader(trainer.data[split_key], batch_size=int(args.batch), rank=-1, mode="val")
    trainer.validator = trainer.get_validator()
    trainer.validator(model=trainer.model)

    if not labels_dir.exists():
        labels_dir.mkdir(parents=True, exist_ok=True)
    return labels_dir


def _box_row(
    *,
    image: Path,
    stem: str,
    box_index: int,
    box: Box,
    names: dict[int, str],
    tags: list[str],
    width: int,
    height: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x1, y1, x2, y2 = box.xyxy
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    row = {
        "image": str(image),
        "stem": stem,
        "box_index": box_index,
        "class_id": box.cls,
        "class": names.get(box.cls, str(box.cls)),
        "conf": f"{box.conf:.6f}",
        "x1": f"{x1:.3f}",
        "y1": f"{y1:.3f}",
        "x2": f"{x2:.3f}",
        "y2": f"{y2:.3f}",
        "area_px": f"{area:.3f}",
        "image_width": width,
        "image_height": height,
        "subset_tags": ",".join(tags),
    }
    if extra:
        row.update(extra)
    return row


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-rgb-ir", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml")
    parser.add_argument("--data-ir", default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml")
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--out-dir", default="/mnt/disk2/lhr/VSD/results/val/e20_0_error_delta_analysis")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--force-predict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data_rgb_ir)
    image_dir = _split_path(data_yaml, args.split)
    images = _iter_images(image_dir)
    if not images:
        raise SystemExit(f"No images found for {args.split}: {image_dir}")

    names = _class_names(data_yaml)
    subset_root = Path("/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets")
    subsets = {
        "dark": _stem_set_from_subset(subset_root / "rgb_ir_dark.yaml", args.split),
        "small": _stem_set_from_subset(subset_root / "rgb_ir_small.yaml", args.split),
        "dark-small": _stem_set_from_subset(subset_root / "rgb_ir_dark-small.yaml", args.split),
        "tiny": _stem_set_from_subset(subset_root / "rgb_ir_tiny.yaml", args.split),
        "low-contrast": _stem_set_from_subset(subset_root / "rgb_ir_low-contrast.yaml", args.split),
    }

    pred_dirs = {model_id: _export_predictions(args, model_id, spec) for model_id, spec in MODEL_SPECS.items()}

    model_counts: dict[str, Counter[str]] = {m: Counter() for m in MODEL_SPECS}
    class_counts: dict[str, Counter[str]] = {m: Counter() for m in MODEL_SPECS}
    tag_counts: dict[str, Counter[str]] = {m: Counter() for m in MODEL_SPECS}
    lost_rows: dict[str, list[dict[str, Any]]] = {"E12_1": [], "E13_2": []}
    eliminated_rows: dict[str, list[dict[str, Any]]] = {"E12_1": [], "E13_2": []}
    fp_rows: list[dict[str, Any]] = []

    for image in images:
        with Image.open(image) as im:
            width, height = im.size
        stem = image.stem
        tags = _subset_tags(stem, subsets)
        gt = _read_label_boxes(_label_path_for_image(image), width, height, with_conf=False)

        preds: dict[str, list[Box]] = {}
        matches: dict[str, tuple[set[int], set[int], dict[int, int]]] = {}
        for model_id, pred_dir in pred_dirs.items():
            pred = _read_label_boxes(pred_dir / f"{stem}.txt", width, height, with_conf=True)
            preds[model_id] = pred
            matched_gt, fp_idx, pred_to_gt = _match(gt, pred, float(args.match_iou))
            matches[model_id] = (matched_gt, fp_idx, pred_to_gt)
            model_counts[model_id]["gt"] += len(gt)
            model_counts[model_id]["pred"] += len(pred)
            model_counts[model_id]["tp"] += len(matched_gt)
            model_counts[model_id]["fp"] += len(fp_idx)
            model_counts[model_id]["fn"] += len(gt) - len(matched_gt)
            for gi in matched_gt:
                class_counts[model_id][f"tp/{names.get(gt[gi].cls, gt[gi].cls)}"] += 1
            for pi in fp_idx:
                p = pred[pi]
                class_counts[model_id][f"fp/{names.get(p.cls, p.cls)}"] += 1
                for tag in tags or ["untagged"]:
                    tag_counts[model_id][f"fp/{tag}"] += 1
                fp_rows.append(
                    _box_row(
                        image=image,
                        stem=stem,
                        box_index=pi,
                        box=p,
                        names=names,
                        tags=tags,
                        width=width,
                        height=height,
                        extra={"model": model_id},
                    )
                )

        e6_matched = matches["E6"][0]
        for target in ("E12_1", "E13_2"):
            target_matched = matches[target][0]
            for gi in sorted(e6_matched - target_matched):
                g = gt[gi]
                for tag in tags or ["untagged"]:
                    tag_counts[target][f"lost_e6_tp/{tag}"] += 1
                lost_rows[target].append(
                    _box_row(
                        image=image,
                        stem=stem,
                        box_index=gi,
                        box=g,
                        names=names,
                        tags=tags,
                        width=width,
                        height=height,
                        extra={"lost_by": target},
                    )
                )

            target_preds = preds[target]
            for pi in sorted(matches["E6"][1]):
                p = preds["E6"][pi]
                still_present = any(q.cls == p.cls and _iou(q.xyxy, p.xyxy) >= float(args.match_iou) for q in target_preds)
                if not still_present:
                    eliminated_rows[target].append(
                        _box_row(
                            image=image,
                            stem=stem,
                            box_index=pi,
                            box=p,
                            names=names,
                            tags=tags,
                            width=width,
                            height=height,
                            extra={"eliminated_by": target},
                        )
                    )

    summary = {
        "experiment": "E20_0",
        "split": args.split,
        "image_count": len(images),
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "match_iou": args.match_iou,
        "model_counts": {k: dict(v) for k, v in model_counts.items()},
        "class_counts": {k: dict(v) for k, v in class_counts.items()},
        "subset_tag_counts": {k: dict(v) for k, v in tag_counts.items()},
        "lost_e6_tp": {k: len(v) for k, v in lost_rows.items()},
        "eliminated_e6_fp": {k: len(v) for k, v in eliminated_rows.items()},
        "prediction_dirs": {k: str(v) for k, v in pred_dirs.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_dir / "fp_by_model.csv", fp_rows)
    for target, rows in lost_rows.items():
        _write_csv(out_dir / f"lost_tp_e6_to_{target.lower()}.csv", rows)
    for target, rows in eliminated_rows.items():
        _write_csv(out_dir / f"eliminated_fp_by_{target.lower()}.csv", rows)

    lines = [
        "# E20_0 Error Delta Analysis",
        "",
        f"- Split: {args.split}",
        f"- Images: {len(images)}",
        f"- Confidence threshold: {args.conf:.3f}",
        f"- Match IoU: {args.match_iou:.3f}",
        "",
        "## Counts",
        "",
        "| Model | GT | Pred | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_id, counts in summary["model_counts"].items():
        lines.append(
            f"| {model_id} | {counts.get('gt', 0)} | {counts.get('pred', 0)} | "
            f"{counts.get('tp', 0)} | {counts.get('fp', 0)} | {counts.get('fn', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Delta",
            "",
            f"- E6 TP missed by E12_1: {summary['lost_e6_tp']['E12_1']}",
            f"- E6 TP missed by E13_2: {summary['lost_e6_tp']['E13_2']}",
            f"- E6 FP eliminated by E12_1: {summary['eliminated_e6_fp']['E12_1']}",
            f"- E6 FP eliminated by E13_2: {summary['eliminated_e6_fp']['E13_2']}",
        ]
    )
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "metrics_summary.md")


if __name__ == "__main__":
    main()
