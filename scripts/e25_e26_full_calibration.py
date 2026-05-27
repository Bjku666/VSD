#!/usr/bin/env python3
"""Full re-inference calibration helpers for S6.5 E25/E26.

This script exports fresh train/val predictions instead of consuming the E20
cache. It is intentionally limited to train/val splits and does not touch test.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import yaml

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "") or str(Path(__file__).resolve().parent)
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import e25_e26_offline_calibration as cal
from e13_tiny_aware_loss_core import E13DetectionTrainer
from e6_feature_fusion_multiscale_core import E6DetectionTrainer


ROOT = Path("/mnt/disk2/lhr/VSD")
DATA_RGB_IR = ROOT / "configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml"
DATA_IR = ROOT / "configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml"
E6_WEIGHTS = ROOT / "results/val/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt"
E6_BASELINE = {
    "False Positives/image": 1.469027,
    "FPPI_dark": 2.536932,
    "FPPI_low-contrast": 1.612707,
    "AP_dark-small_object": 0.100028,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _eval_yaml_for_split(base_yaml: Path, split: str, out_dir: Path, name: str) -> Path:
    if split == "val":
        return base_yaml
    cfg = dict(_load_yaml(base_yaml))
    cfg["val"] = cfg[split]
    for key in ("rgb", "ir"):
        node = cfg.get(key)
        if isinstance(node, dict) and split in node:
            node = dict(node)
            node["val"] = node[split]
            cfg[key] = node
    out = out_dir / "_generated_eval_data" / f"{name}_{split}_as_val.yaml"
    _write_yaml(out, cfg)
    return out


def _trainer_class(validator: str):
    if validator == "e13":
        return E13DetectionTrainer
    if validator == "e6":
        return E6DetectionTrainer
    raise ValueError(f"Unsupported validator: {validator}")


def export_predictions(
    *,
    weights: Path,
    validator: str,
    split: str,
    out_dir: Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    conf: float,
    nms_iou: float,
    force: bool,
) -> Path:
    labels_dir = out_dir / "labels"
    if labels_dir.exists() and any(labels_dir.glob("*.txt")) and not force:
        return labels_dir

    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_data_rgb_ir = _eval_yaml_for_split(DATA_RGB_IR, split, out_dir, "rgb_ir")
    eval_data_ir = _eval_yaml_for_split(DATA_IR, split, out_dir, "ir")
    overrides = {
        "task": "detect",
        "mode": "train",
        "model": str(weights),
        "data": str(eval_data_rgb_ir),
        "epochs": 1,
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "device": str(device),
        "project": str(out_dir.parent),
        "name": out_dir.name,
        "resume": False,
        "plots": False,
        "exist_ok": True,
        "save_txt": True,
        "save_conf": True,
        "conf": float(conf),
        "iou": float(nms_iou),
        "max_det": 300,
    }
    trainer = _trainer_class(validator)(overrides=overrides)
    trainer.set_fusion_mode("rgb_ir")
    trainer.set_ir_data(str(eval_data_ir))
    if validator == "e13" and hasattr(trainer, "set_loss_config"):
        trainer.set_loss_config(loss_mode="center-aware", loss_scope="small", aux_weight=0.5)
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device).float().eval()
    trainer.test_loader = trainer.get_dataloader(trainer.data["val"], batch_size=int(batch), rank=-1, mode="val")
    trainer.validator = trainer.get_validator()
    trainer.validator(model=trainer.model)
    if not labels_dir.exists():
        raise RuntimeError(f"Prediction export did not create labels: {labels_dir}")
    return labels_dir


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cal.write_csv(path, rows)


def _ap_for_class(
    data: cal.SplitData,
    class_id: int,
    keep: Callable[[cal.Box, set[str]], bool],
    iou_thr: float,
    stems: set[str] | None = None,
    gt_override: dict[str, list[cal.Box]] | None = None,
) -> float | None:
    selected_stems = stems if stems is not None else set(data.gt)
    gt_map = gt_override if gt_override is not None else data.gt
    total_gt = sum(1 for stem in selected_stems for g in gt_map.get(stem, []) if g.cls == class_id)
    if total_gt <= 0:
        return None

    preds: list[tuple[float, str, cal.Box]] = []
    for stem in selected_stems:
        tags = data.tags.get(stem, set())
        for p in data.pred.get(stem, []):
            if p.cls == class_id and keep(p, tags):
                preds.append((p.conf, stem, p))
    preds.sort(key=lambda x: x[0], reverse=True)

    used: dict[str, set[int]] = defaultdict(set)
    tp: list[float] = []
    fp: list[float] = []
    for _score, stem, pred in preds:
        best_i = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_map.get(stem, [])):
            if idx in used[stem] or gt.cls != class_id:
                continue
            val = cal.iou(pred.xyxy, gt.xyxy)
            if val >= iou_thr and val > best_iou:
                best_iou = val
                best_i = idx
        if best_i >= 0:
            used[stem].add(best_i)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    if not preds:
        return 0.0
    cum_tp: list[float] = []
    cum_fp: list[float] = []
    run_tp = 0.0
    run_fp = 0.0
    for t, f in zip(tp, fp):
        run_tp += t
        run_fp += f
        cum_tp.append(run_tp)
        cum_fp.append(run_fp)
    recalls = [v / total_gt for v in cum_tp]
    precisions = [cum_tp[i] / max(cum_tp[i] + cum_fp[i], 1e-9) for i in range(len(cum_tp))]

    ap = 0.0
    for r in [x / 100.0 for x in range(101)]:
        p_at_r = max((p for p, rec in zip(precisions, recalls) if rec >= r), default=0.0)
        ap += p_at_r
    return ap / 101.0


def map50_95(
    data: cal.SplitData,
    keep: Callable[[cal.Box, set[str]], bool],
    stems: set[str] | None = None,
    gt_override: dict[str, list[cal.Box]] | None = None,
) -> float:
    vals: list[float] = []
    for i in range(10):
        thr = 0.50 + 0.05 * i
        class_vals = []
        for cls_id in data.names:
            ap = _ap_for_class(data, cls_id, keep, thr, stems=stems, gt_override=gt_override)
            if ap is not None:
                class_vals.append(ap)
        vals.append(sum(class_vals) / len(class_vals) if class_vals else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def _scope_map(data: cal.SplitData, keep: Callable[[cal.Box, set[str]], bool]) -> dict[str, float]:
    out = {
        "AP50-95_full": map50_95(data, keep),
    }
    for scope in ("dark-small", "tiny", "low-contrast"):
        stems, gt = cal.read_object_subset(f"{scope}_object")
        out[f"AP_{scope}_object"] = map50_95(data, keep, stems=stems, gt_override=gt)
    return out


def _rule_grid(data: cal.SplitData) -> list[tuple[str, str, Callable[[cal.Box, set[str]], bool], dict[str, Any]]]:
    thresholds = cal.load_json(ROOT / "results/dataset_audit/train_thresholds.json")
    tiny_area = float(thresholds.get("object_area_tiny_threshold", 880.0))
    small_area = float(thresholds.get("object_area_small_threshold", 1288.0))
    rules: list[tuple[str, str, Callable[[cal.Box, set[str]], bool], dict[str, Any]]] = []
    for t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        rules.append(("global", f"conf_{t:.2f}", cal.make_global_keep(t), {"global_conf": t}))
    for tiny_t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        for small_t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            other_t = 0.30

            def keep(p: cal.Box, tags: set[str], tiny_t: float = tiny_t, small_t: float = small_t, other_t: float = other_t) -> bool:
                bucket = cal.size_bucket(p, tiny_area, small_area)
                return p.conf >= {"tiny": tiny_t, "small": small_t, "other": other_t}[bucket]

            rules.append(("size_wise", f"tiny{tiny_t:.2f}_small{small_t:.2f}_other{other_t:.2f}", keep, {"tiny_conf": tiny_t, "small_conf": small_t, "other_conf": other_t}))
    for dark_t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        for other_t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):

            def keep(p: cal.Box, tags: set[str], dark_t: float = dark_t, other_t: float = other_t) -> bool:
                return p.conf >= (dark_t if "dark" in tags else other_t)

            rules.append(("illumination_wise", f"dark{dark_t:.2f}_other{other_t:.2f}", keep, {"dark_conf": dark_t, "other_conf": other_t}))
    for cls_id, cls_name in data.names.items():
        for cls_t in (0.30, 0.35, 0.40, 0.45, 0.50):

            def keep(p: cal.Box, tags: set[str], cls_id: int = cls_id, cls_t: float = cls_t) -> bool:
                return p.conf >= (cls_t if p.cls == cls_id else 0.25)

            rules.append(("class_wise_single", f"{cls_name}_{cls_t:.2f}", keep, {"raised_class": cls_name, "raised_conf": cls_t}))
    return rules


def run_e25_1_full(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_root = out_dir / "predictions"
    ious = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    rows: list[dict[str, Any]] = []

    for nms_iou in ious:
        train_labels = export_predictions(
            weights=Path(args.weights), validator="e6", split="train", out_dir=pred_root / f"train_iou{nms_iou:.2f}",
            imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device, conf=args.export_conf, nms_iou=nms_iou, force=args.force_predict,
        )
        val_labels = export_predictions(
            weights=Path(args.weights), validator="e6", split="val", out_dir=pred_root / f"val_iou{nms_iou:.2f}",
            imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device, conf=args.export_conf, nms_iou=nms_iou, force=args.force_predict,
        )
        train = cal.load_split("train", train_labels)
        val = cal.load_split("val", val_labels)
        for strategy, rule_id, keep, params in _rule_grid(train):
            train_metrics = cal.match_predictions(train, keep)
            val_metrics = cal.match_predictions(val, keep)
            row = {"strategy": strategy, "rule_id": rule_id, "nms_iou": nms_iou, **params}
            row.update(cal.flatten_metric("train", train_metrics))
            row.update(cal.flatten_metric("val", val_metrics))
            row.update(_scope_map(val, keep))
            rows.append(row)

    rows.sort(key=lambda r: (float(r["val_fp_per_image"]), -float(r["val_recall"])))
    _write_csv(out_dir / "calibration_grid.csv", rows)
    pareto = cal.pareto_rows(rows)
    _write_csv(out_dir / "pareto_curve.csv", pareto)
    cal.render_pareto_png(out_dir / "pareto_ap_obj_vs_fppi_dark.png", rows, "val_fppi_dark", "AP_dark-small_object", "FPPI_dark", "dark-small object AP50-95")

    def score(row: dict[str, Any]) -> tuple[float, float, float]:
        return (float(row["val_f1"]), -float(row["val_fp_per_image"]), float(row["AP_dark-small_object"]))

    best_f1 = max(rows, key=score)
    best_low_fp = min((r for r in rows if float(r["val_recall"]) >= 0.90), key=lambda r: float(r["val_fp_per_image"]), default=min(rows, key=lambda r: float(r["val_fp_per_image"])))
    accepted = [
        r for r in rows
        if float(r["AP_dark-small_object"]) >= 0.098
        and float(r["val_fp_per_image"]) < E6_BASELINE["False Positives/image"]
        and float(r["val_fppi_dark"]) < E6_BASELINE["FPPI_dark"]
        and float(r["val_fppi_low_contrast"]) < E6_BASELINE["FPPI_low-contrast"]
    ]
    best_accept = max(accepted, key=score) if accepted else None
    best = {
        "experiment": "E25_1_full",
        "source": {
            "weights": str(Path(args.weights)),
            "export_conf": args.export_conf,
            "nms_iou_grid": ious,
            "train_val_only": True,
        },
        "best_f1": best_f1,
        "best_low_fp_recall90": best_low_fp,
        "best_accepting_gate": best_accept,
        "accepted_count": len(accepted),
        "gate": E6_BASELINE,
    }
    (out_dir / "best_operating_points.json").write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    classwise = {
        "thresholds": {
            "car": 0.40,
            "truck": 0.50,
            "bus": 0.50,
            "van": 0.45,
            "freight_car": 0.45,
        },
        "source": "E26_1 cached thresholds carried forward for E26_1_full validation",
    }
    (out_dir / "classwise_thresholds.json").write_text(json.dumps(classwise, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    target = best_accept or best_f1
    required = {
        "selected_rule_id": target["rule_id"],
        "selected_strategy": target["strategy"],
        "selected_nms_iou": target["nms_iou"],
        "Precision": target["val_precision"],
        "Recall": target["val_recall"],
        "False Positives/image": target["val_fp_per_image"],
        "FPPI_dark": target["val_fppi_dark"],
        "FPPI_low-contrast": target["val_fppi_low_contrast"],
        "AP_dark-small_object": target["AP_dark-small_object"],
        "AP_tiny_object": target["AP_tiny_object"],
        "AP_low-contrast_object": target["AP_low-contrast_object"],
        "accepted_gate": bool(best_accept),
    }
    (out_dir / "required_metrics.json").write_text(json.dumps(required, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E25_1_full E6 Full Re-Inference Calibration",
        "",
        f"- Grid rows: {len(rows)}",
        f"- Accepted gate rows: {len(accepted)}",
        f"- Selected: {target['strategy']} / {target['rule_id']} / NMS {float(target['nms_iou']):.2f}",
        f"- FP/image: {float(target['val_fp_per_image']):.6f}",
        f"- FPPI_dark: {float(target['val_fppi_dark']):.6f}",
        f"- FPPI_low-contrast: {float(target['val_fppi_low_contrast']):.6f}",
        f"- AP_dark-small_object: {float(target['AP_dark-small_object']):.6f}",
        f"- AP_tiny_object: {float(target['AP_tiny_object']):.6f}",
        f"- AP_low-contrast_object: {float(target['AP_low-contrast_object']):.6f}",
    ]
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "metrics_summary.md")


def run_e26_1_full(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_root = Path(args.pred_root)
    train_labels = pred_root / "train_iou0.70" / "labels"
    val_labels = pred_root / "val_iou0.70" / "labels"
    if not train_labels.exists() or not val_labels.exists():
        raise SystemExit(f"Missing E25_1_full predictions at {pred_root}; run e25_1_full first.")
    train = cal.load_split("train", train_labels)
    val = cal.load_split("val", val_labels)
    thresholds_by_name = {
        "car": 0.40,
        "truck": 0.50,
        "bus": 0.50,
        "van": 0.45,
        "freight_car": 0.45,
    }
    thresholds = {cls_id: thresholds_by_name.get(name, 0.25) for cls_id, name in train.names.items()}

    def keep(p: cal.Box, tags: set[str]) -> bool:
        return p.conf >= thresholds.get(p.cls, 0.25)

    train_metrics = cal.match_predictions(train, keep)
    val_metrics = cal.match_predictions(val, keep)
    object_metrics = _scope_map(val, keep)
    per_class_rows = []
    for cls_id, cls_name in val.names.items():
        cls_metrics = val_metrics["per_class"].get(str(cls_id), {})
        per_class_rows.append({
            "class_id": cls_id,
            "class": cls_name,
            "threshold": thresholds.get(cls_id, 0.25),
            "precision": cls_metrics.get("precision", 0.0),
            "recall": cls_metrics.get("recall", 0.0),
            "gt": cls_metrics.get("gt", 0),
            "pred": cls_metrics.get("pred", 0),
            "fp": cls_metrics.get("fp", 0),
        })
    _write_csv(out_dir / "per_class_metrics.csv", per_class_rows)
    best = {
        "experiment": "E26_1_full",
        "class_thresholds": thresholds_by_name,
        "source_predictions": str(pred_root),
        "train": {k: v for k, v in train_metrics.items() if k != "per_class"},
        "val": {k: v for k, v in val_metrics.items() if k != "per_class"},
        "object": object_metrics,
    }
    (out_dir / "best_operating_points.json").write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    passed = (
        float(val_metrics["fp_per_image"]) <= 1.20
        and float(val_metrics["fppi_dark"]) <= 1.90
        and float(object_metrics["AP_dark-small_object"]) >= 0.098
    )
    required = {
        "Precision": val_metrics["precision"],
        "Recall": val_metrics["recall"],
        "False Positives/image": val_metrics["fp_per_image"],
        "FPPI_dark": val_metrics["fppi_dark"],
        "FPPI_low-contrast": val_metrics["fppi_low_contrast"],
        "class_confusion_fp": val_metrics["class_confusion_fp"],
        "AP_dark-small_object": object_metrics["AP_dark-small_object"],
        "AP_tiny_object": object_metrics["AP_tiny_object"],
        "AP_low-contrast_object": object_metrics["AP_low-contrast_object"],
        "accepted_gate": passed,
    }
    (out_dir / "required_metrics.json").write_text(json.dumps(required, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# E26_1_full Class-Wise Threshold Full Re-Inference",
        "",
        f"- Class thresholds: {json.dumps(thresholds_by_name, ensure_ascii=False)}",
        f"- FP/image: {float(val_metrics['fp_per_image']):.6f}",
        f"- FPPI_dark: {float(val_metrics['fppi_dark']):.6f}",
        f"- FPPI_low-contrast: {float(val_metrics['fppi_low_contrast']):.6f}",
        f"- class_confusion FP: {val_metrics['class_confusion_fp']}",
        f"- AP_dark-small_object: {float(object_metrics['AP_dark-small_object']):.6f}",
        f"- AP_tiny_object: {float(object_metrics['AP_tiny_object']):.6f}",
        f"- AP_low-contrast_object: {float(object_metrics['AP_low-contrast_object']):.6f}",
        f"- Accepted gate: {passed}",
    ]
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "metrics_summary.md")


def run_export(args: argparse.Namespace) -> None:
    labels = export_predictions(
        weights=Path(args.weights),
        validator=args.validator,
        split=args.split,
        out_dir=Path(args.out_dir),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        conf=args.conf,
        nms_iou=args.nms_iou,
        force=args.force,
    )
    print(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    p1 = sub.add_parser("e25_1_full")
    p1.add_argument("--out-dir", default=str(ROOT / "results/val/e25_1_full_e6_calibration_sweep"))
    p1.add_argument("--weights", default=str(E6_WEIGHTS))
    p1.add_argument("--imgsz", type=int, default=640)
    p1.add_argument("--batch", type=int, default=32)
    p1.add_argument("--workers", type=int, default=8)
    p1.add_argument("--device", default="0")
    p1.add_argument("--export-conf", type=float, default=0.01)
    p1.add_argument("--force-predict", action="store_true")
    p1.set_defaults(func=run_e25_1_full)
    p2 = sub.add_parser("e26_1_full")
    p2.add_argument("--out-dir", default=str(ROOT / "results/val/e26_1_full_classwise_threshold_calibration"))
    p2.add_argument("--pred-root", default=str(ROOT / "results/val/e25_1_full_e6_calibration_sweep/predictions"))
    p2.set_defaults(func=run_e26_1_full)
    p3 = sub.add_parser("export")
    p3.add_argument("--weights", required=True)
    p3.add_argument("--validator", default="e6", choices=["e6", "e13"])
    p3.add_argument("--split", default="val", choices=["train", "val"])
    p3.add_argument("--out-dir", required=True)
    p3.add_argument("--imgsz", type=int, default=640)
    p3.add_argument("--batch", type=int, default=16)
    p3.add_argument("--workers", type=int, default=8)
    p3.add_argument("--device", default="0")
    p3.add_argument("--conf", type=float, default=0.01)
    p3.add_argument("--nms-iou", type=float, default=0.70)
    p3.add_argument("--force", action="store_true")
    p3.set_defaults(func=run_export)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
