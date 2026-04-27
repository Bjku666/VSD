#!/usr/bin/env python3
"""评估 YOLO11n 的 IR-only baseline，并导出完整指标与可视化结果。

本脚本会输出：
- 标准指标：mAP50、mAP50-95、Precision、Recall、各类别 AP。
- 自定义指标：AP_small、Recall_small、AP_dark、Recall_dark、AP_dark-small。
- 错误分析样本：检对、漏检、误检三类可视化图片。
- 结果文件：metrics_summary、required_metrics（json/csv）和指标图表。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import torch
import yaml
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_metrics(metrics: Any) -> Dict[str, Any]:
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

    if maps is not None:
        maps_arr = np.array(maps, dtype=float).reshape(-1)
        ap50_arr = None
        if all_ap is not None:
            all_ap_arr = np.array(all_ap, dtype=float)
            if all_ap_arr.ndim == 2 and all_ap_arr.shape[1] > 0:
                ap50_arr = all_ap_arr[:, 0]

        for idx, ap in enumerate(maps_arr):
            cls_name = str(names.get(idx, idx))
            cls_info = {
                "AP50-95": _to_float(ap),
            }
            if ap50_arr is not None and idx < len(ap50_arr):
                cls_info["AP50"] = _to_float(ap50_arr[idx])
            result["per_class_AP"][cls_name] = cls_info

    return result


def _run_val(
    model: YOLO,
    data_yaml: str,
    imgsz: int,
    split: str,
    device: str,
    ultra_project: Path,
    ultra_name: str,
) -> Dict[str, Any]:
    # Ultralytics 校验要求 data yaml 同时包含 train/val，子集 yaml 常只有 val，这里临时补齐。
    tmp_yaml: Optional[str] = None
    eval_yaml = data_yaml
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and "train" not in cfg and "val" in cfg:
        cfg = dict(cfg)
        cfg["train"] = cfg["val"]
        fd, tmp_yaml = tempfile.mkstemp(prefix="eval_subset_", suffix=".yaml")
        os.close(fd)
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        eval_yaml = tmp_yaml

    try:
        metrics = model.val(
            data=eval_yaml,
            imgsz=imgsz,
            split=split,
            device=device,
            verbose=False,
            project=str(ultra_project),
            name=ultra_name,
            exist_ok=True,
        )
        parsed = _extract_metrics(metrics)
    finally:
        if tmp_yaml is not None and Path(tmp_yaml).exists():
            Path(tmp_yaml).unlink()
        _release_cuda_memory()

    return parsed


def _iter_image_paths(path_or_txt: Path) -> List[Path]:
    if path_or_txt.is_dir():
        files = [
            p
            for p in sorted(path_or_txt.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        return files

    if path_or_txt.is_file() and path_or_txt.suffix.lower() == ".txt":
        items: List[Path] = []
        with open(path_or_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(Path(line))
        return items

    if path_or_txt.is_file() and path_or_txt.suffix.lower() in IMAGE_SUFFIXES:
        return [path_or_txt]

    return []


def _resolve_data_yaml(data_yaml: str, split: str) -> Tuple[List[Path], Dict[int, str]]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_root = Path(cfg.get("path", "."))
    split_entry = cfg.get(split)
    if split_entry is None:
        raise ValueError(f"Split '{split}' not found in {data_yaml}")

    entries: Sequence[str]
    if isinstance(split_entry, (list, tuple)):
        entries = [str(x) for x in split_entry]
    else:
        entries = [str(split_entry)]

    image_paths: List[Path] = []
    for item in entries:
        p = Path(item)
        if not p.is_absolute():
            p = base_root / p
        for img_p in _iter_image_paths(p):
            if not img_p.is_absolute():
                image_paths.append((base_root / img_p).resolve())
            else:
                image_paths.append(img_p.resolve())

    dedup = sorted(set(image_paths))

    names_cfg = cfg.get("names", {})
    names: Dict[int, str] = {}
    if isinstance(names_cfg, dict):
        for k, v in names_cfg.items():
            try:
                idx = int(k)
            except Exception:
                continue
            names[idx] = str(v)
    elif isinstance(names_cfg, list):
        for idx, v in enumerate(names_cfg):
            names[idx] = str(v)

    return dedup, names


def _img_to_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def _load_gt_boxes(label_path: Path, width: int, height: int) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    boxes: List[Tuple[int, Tuple[float, float, float, float]]] = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
            except Exception:
                continue
            x1 = (xc - bw / 2.0) * width
            y1 = (yc - bh / 2.0) * height
            x2 = (xc + bw / 2.0) * width
            y2 = (yc + bh / 2.0) * height
            boxes.append((cls_id, (x1, y1, x2, y2)))
    return boxes


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _match_tp_fp_fn(
    gt_boxes: List[Tuple[int, Tuple[float, float, float, float]]],
    pred_boxes: List[Tuple[int, float, Tuple[float, float, float, float]]],
    iou_thr: float,
) -> Tuple[int, int, int, set, set]:
    matched_gt = set()
    matched_pred = set()

    pred_order = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i][1], reverse=True)
    for pred_idx in pred_order:
        pred_cls, _pred_conf, pred_xyxy = pred_boxes[pred_idx]
        best_gt_idx = -1
        best_iou = iou_thr
        for gt_idx, (gt_cls, gt_xyxy) in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            if gt_cls != pred_cls:
                continue
            iou = _iou(pred_xyxy, gt_xyxy)
            if iou >= best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, matched_gt, matched_pred


def _draw_case(
    case: Dict[str, Any],
    names: Dict[int, str],
    save_path: Path,
) -> None:
    img = cv2.imread(str(case["image_path"]))
    if img is None:
        return

    matched_gt = set(case["matched_gt_indices"])
    matched_pred = set(case["matched_pred_indices"])

    for idx, (cls_id, xyxy) in enumerate(case["gt_boxes"]):
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        is_fn = idx not in matched_gt
        color = (0, 215, 255) if is_fn else (0, 255, 0)
        tag = "FN" if is_fn else "GT"
        cls_name = names.get(int(cls_id), str(cls_id))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{tag}:{cls_name}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    for idx, (cls_id, conf, xyxy) in enumerate(case["pred_boxes"]):
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        is_fp = idx not in matched_pred
        color = (0, 0, 255) if is_fp else (255, 0, 0)
        tag = "FP" if is_fp else "TP"
        cls_name = names.get(int(cls_id), str(cls_id))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{tag}:{cls_name} {conf:.2f}",
            (x1, min(img.shape[0] - 5, y2 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)


def _mine_visual_cases(
    model: YOLO,
    data_yaml: str,
    split: str,
    imgsz: int,
    device: str,
    conf_thr: float,
    iou_thr: float,
    topk: int,
    max_images: int,
    batch: int,
    out_dir: Path,
) -> Dict[str, Any]:
    image_paths, names = _resolve_data_yaml(data_yaml, split)
    if max_images > 0:
        image_paths = image_paths[:max_images]

    cases: List[Dict[str, Any]] = []
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=conf_thr,
        iou=0.7,
        device=device,
        verbose=False,
        stream=True,
        batch=max(1, int(batch)),
        half=(str(device).lower() != "cpu"),
        save=False,
    )

    for idx, res in enumerate(results):
        if idx >= len(image_paths):
            break
        img_path = image_paths[idx]
        h, w = res.orig_shape
        gt_boxes = _load_gt_boxes(_img_to_label_path(img_path), w, h)

        pred_boxes: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                pred_boxes.append(
                    (
                        int(cls[i]),
                        float(conf[i]),
                        (
                            float(xyxy[i][0]),
                            float(xyxy[i][1]),
                            float(xyxy[i][2]),
                            float(xyxy[i][3]),
                        ),
                    )
                )

        tp, fp, fn, matched_gt, matched_pred = _match_tp_fp_fn(gt_boxes, pred_boxes, iou_thr)
        cases.append(
            {
                "image_path": str(img_path),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "gt_count": int(len(gt_boxes)),
                "pred_count": int(len(pred_boxes)),
                "gt_boxes": gt_boxes,
                "pred_boxes": pred_boxes,
                "matched_gt_indices": sorted(list(matched_gt)),
                "matched_pred_indices": sorted(list(matched_pred)),
            }
        )

    strict_correct = [c for c in cases if c["tp"] > 0 and c["fp"] == 0 and c["fn"] == 0]
    if strict_correct:
        correct = strict_correct
    else:
        # 兜底：如果没有严格“全对”样本，则选 TP 高且 (FP+FN) 低的样本。
        correct = [c for c in cases if c["tp"] > 0]
    missed = [c for c in cases if c["fn"] > 0]
    false_positive = [c for c in cases if c["fp"] > 0]

    correct.sort(key=lambda c: (-(c["fp"] + c["fn"]), c["tp"], c["gt_count"]), reverse=True)
    missed.sort(key=lambda c: (c["fn"], c["fp"], c["gt_count"]), reverse=True)
    false_positive.sort(key=lambda c: (c["fp"], c["fn"], c["pred_count"]), reverse=True)

    selected = {
        "correct": correct[:topk],
        "missed": missed[:topk],
        "false_positive": false_positive[:topk],
    }

    case_root = out_dir / "visual_cases"
    (case_root / "correct").mkdir(parents=True, exist_ok=True)
    (case_root / "missed").mkdir(parents=True, exist_ok=True)
    (case_root / "false_positive").mkdir(parents=True, exist_ok=True)
    for bucket, items in selected.items():
        for rank, case in enumerate(items, start=1):
            stem = Path(case["image_path"]).stem
            filename = (
                f"{rank:02d}_{stem}_tp{case['tp']}_fp{case['fp']}_fn{case['fn']}.jpg"
            )
            _draw_case(case, names, case_root / bucket / filename)

    summary = {
        "data_yaml": data_yaml,
        "split": split,
        "device": device,
        "conf_thr": conf_thr,
        "iou_thr": iou_thr,
        "topk": topk,
        "max_images": max_images,
        "total_images_scanned": len(cases),
        "bucket_counts": {
            "correct": len(correct),
            "missed": len(missed),
            "false_positive": len(false_positive),
        },
        "selected_samples": {
            key: [
                {
                    "image_path": item["image_path"],
                    "tp": item["tp"],
                    "fp": item["fp"],
                    "fn": item["fn"],
                }
                for item in val
            ]
            for key, val in selected.items()
        },
        "output_dir": str(case_root),
    }

    with open(out_dir / "visual_cases_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def _build_required_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    std = summary["standard_metrics"]
    custom = summary["custom_metrics"]

    required = {
        "mAP50": std["mAP50"],
        "mAP50-95": std["mAP50-95"],
        "Precision": std["Precision"],
        "Recall": std["Recall"],
        "per_class_AP": std["per_class_AP"],
        "AP_small": custom["AP_small"],
        "Recall_small": custom["Recall_small"],
        "AP_dark": custom["AP_dark"],
        "Recall_dark": custom["Recall_dark"],
        "AP_dark-small": custom["AP_dark-small"],
    }
    return required


def _write_required_metrics_files(out_dir: Path, required: Dict[str, Any]) -> Tuple[Path, Path]:
    json_path = out_dir / "required_metrics.json"
    csv_path = out_dir / "required_metrics.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(required, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
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
        per_class = required.get("per_class_AP", {})
        for cls_name, vals in per_class.items():
            writer.writerow(
                [
                    cls_name,
                    f"{_to_float(vals.get('AP50', 0.0)):.6f}",
                    f"{_to_float(vals.get('AP50-95', 0.0)):.6f}",
                ]
            )

    return json_path, csv_path


def _render_required_metric_figures(out_dir: Path, required: Dict[str, Any]) -> List[Path]:
    fig_dir = out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    core_names = [
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall",
        "AP_small",
        "Recall_small",
        "AP_dark",
        "Recall_dark",
        "AP_dark-small",
    ]
    core_values = [float(required[k]) for k in core_names]

    plt.figure(figsize=(11, 5))
    plt.bar(core_names, core_values)
    plt.ylim(0.0, 1.0)
    plt.title("IR-only Baseline Metric Overview")
    plt.ylabel("Score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig1 = fig_dir / "required_metrics_overview.png"
    plt.savefig(fig1, dpi=180)
    plt.close()

    per_class = required.get("per_class_AP", {})
    cls_names = list(per_class.keys())
    cls_ap = [
        _to_float(per_class[c].get("AP50-95", 0.0))
        for c in cls_names
    ]
    plt.figure(figsize=(8, 4.5))
    plt.bar(cls_names, cls_ap)
    plt.ylim(0.0, 1.0)
    plt.title("Per-class AP50-95")
    plt.ylabel("AP50-95")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig2 = fig_dir / "per_class_ap50_95.png"
    plt.savefig(fig2, dpi=180)
    plt.close()

    return [fig1, fig2]


def _format_md(summary: Dict[str, Any]) -> str:
    std = summary["standard_metrics"]
    custom = summary["custom_metrics"]
    visual = summary.get("visual_cases")
    required_files = summary.get("required_metric_files", {})
    required_figures = summary.get("required_metric_figures", [])

    lines = [
        "# YOLO11n IR-only Baseline 指标报告",
        "",
        "## 标准指标",
        "",
        f"- mAP50: {std['mAP50']:.6f}",
        f"- mAP50-95: {std['mAP50-95']:.6f}",
        f"- Precision: {std['Precision']:.6f}",
        f"- Recall: {std['Recall']:.6f}",
        "",
        "## 自定义指标",
        "",
        f"- AP_small: {custom['AP_small']:.6f}",
        f"- Recall_small: {custom['Recall_small']:.6f}",
        f"- AP_dark: {custom['AP_dark']:.6f}",
        f"- Recall_dark: {custom['Recall_dark']:.6f}",
        f"- AP_dark-small: {custom['AP_dark-small']:.6f}",
        "",
        "## 各类别 AP（完整验证集）",
        "",
        "| 类别 | AP50 | AP50-95 |",
        "| --- | ---: | ---: |",
    ]

    for cls_name, cls_metrics in summary["standard_metrics"]["per_class_AP"].items():
        ap50 = cls_metrics.get("AP50", 0.0)
        ap = cls_metrics.get("AP50-95", 0.0)
        lines.append(f"| {cls_name} | {ap50:.6f} | {ap:.6f} |")

    if visual is not None:
        lines.extend(
            [
                "",
                "## 可视化误差样本",
                "",
                f"- 扫描图像数: {visual['total_images_scanned']}",
                f"- 检对样本数: {visual['bucket_counts']['correct']}",
                f"- 漏检样本数: {visual['bucket_counts']['missed']}",
                f"- 误检样本数: {visual['bucket_counts']['false_positive']}",
                f"- 图片输出目录: {visual['output_dir']}",
            ]
        )

    if required_files:
        lines.extend(
            [
                "",
                "## 要求指标文件",
                "",
                f"- JSON: {required_files.get('json', '')}",
                f"- CSV: {required_files.get('csv', '')}",
            ]
        )

    if required_figures:
        lines.extend(
            [
                "",
                "## 指标图表",
                "",
            ]
        )
        for fig in required_figures:
            lines.append(f"- {fig}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 YOLO11n IR-only baseline 并导出完整指标。")
    parser.add_argument("--model", type=str, required=True, help="训练权重路径，例如 best.pt")
    parser.add_argument(
        "--data-full",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml",
    )
    parser.add_argument(
        "--data-small",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/ir_small.yaml",
    )
    parser.add_argument(
        "--data-dark",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/ir_dark.yaml",
    )
    parser.add_argument(
        "--data-dark-small",
        type=str,
        default="/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/ir_dark-small.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--device", type=str, default="0", help="CUDA 设备号，例如 0 或 0,1")
    parser.add_argument("--case-conf", type=float, default=0.25)
    parser.add_argument("--case-iou", type=float, default=0.50)
    parser.add_argument("--case-topk", type=int, default=20)
    parser.add_argument("--case-max-images", type=int, default=0, help="0 表示使用该 split 的全部图像")
    parser.add_argument("--case-batch", type=int, default=1, help="可视化样本挖掘的推理 batch 大小")
    parser.add_argument(
        "--case-device",
        type=str,
        default="",
        help="可视化样本挖掘设备，默认与 --device 一致；可设为 cpu 避免显存不足",
    )
    parser.add_argument("--skip-case-viz", action="store_true", help="关闭检对/漏检/误检可视化")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="评测输出目录；默认自动按实验名写入 experiments/e2_ir_only/<run_name>/<split>",
    )
    parser.add_argument(
        "--ultra-project",
        type=str,
        default="",
        help="Ultralytics 原生 val 产物目录；默认写到 out-dir/ultralytics_val",
    )
    args = parser.parse_args()

    if args.out_dir.strip():
        out_dir = Path(args.out_dir)
    else:
        model_path = Path(args.model)
        run_name = model_path.parent.parent.name if len(model_path.parents) >= 2 else model_path.stem
        out_dir = Path("/mnt/disk2/lhr/VSD/experiments/e2_ir_only") / run_name / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ultra_project.strip():
        ultra_project = Path(args.ultra_project)
    else:
        ultra_project = out_dir / "ultralytics_val"
    ultra_project.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    _release_cuda_memory()

    standard = _run_val(
        model,
        args.data_full,
        args.imgsz,
        args.split,
        args.device,
        ultra_project,
        f"{args.split}_full",
    )
    small = _run_val(
        model,
        args.data_small,
        args.imgsz,
        args.split,
        args.device,
        ultra_project,
        f"{args.split}_small",
    )
    dark = _run_val(
        model,
        args.data_dark,
        args.imgsz,
        args.split,
        args.device,
        ultra_project,
        f"{args.split}_dark",
    )
    dark_small = _run_val(
        model,
        args.data_dark_small,
        args.imgsz,
        args.split,
        args.device,
        ultra_project,
        f"{args.split}_dark-small",
    )

    # 验证结束后不再复用该模型做可视化挖掘，避免推理态冲突和显存碎片累积。
    del model
    _release_cuda_memory()

    visual_cases: Optional[Dict[str, Any]] = None
    if not args.skip_case_viz:
        case_device = args.case_device.strip() or args.device
        case_model = YOLO(args.model)
        try:
            visual_cases = _mine_visual_cases(
                model=case_model,
                data_yaml=args.data_full,
                split=args.split,
                imgsz=args.imgsz,
                device=case_device,
                conf_thr=args.case_conf,
                iou_thr=args.case_iou,
                topk=args.case_topk,
                max_images=args.case_max_images,
                batch=args.case_batch,
                out_dir=out_dir,
            )
        except Exception as ex:
            err_text = str(ex).lower()
            fallback = (
                isinstance(ex, torch.OutOfMemoryError)
                or "out of memory" in err_text
                or "inference tensors do not track version counter" in err_text
            )
            if not fallback:
                raise
            print("[warn] 可视化样本挖掘在当前设备失败，自动切换到 CPU 重试。")
            print(f"[warn] 原始错误：{ex}")
            del case_model
            _release_cuda_memory()
            cpu_model = YOLO(args.model)
            visual_cases = _mine_visual_cases(
                model=cpu_model,
                data_yaml=args.data_full,
                split=args.split,
                imgsz=args.imgsz,
                device="cpu",
                conf_thr=args.case_conf,
                iou_thr=args.case_iou,
                topk=args.case_topk,
                max_images=args.case_max_images,
                batch=1,
                out_dir=out_dir,
            )
            del cpu_model
            _release_cuda_memory()
        else:
            del case_model
            _release_cuda_memory()

    summary: Dict[str, Any] = {
        "model": args.model,
        "imgsz": args.imgsz,
        "split": args.split,
        "device": args.device,
        "data": {
            "full": args.data_full,
            "small": args.data_small,
            "dark": args.data_dark,
            "dark_small": args.data_dark_small,
        },
        "ultralytics_project_dir": str(ultra_project),
        "standard_metrics": standard,
        "custom_metrics": {
            "AP_small": small["mAP50-95"],
            "Recall_small": small["Recall"],
            "AP_dark": dark["mAP50-95"],
            "Recall_dark": dark["Recall"],
            "AP_dark-small": dark_small["mAP50-95"],
        },
        "visual_cases": visual_cases,
    }

    required = _build_required_metrics(summary)
    req_json_path, req_csv_path = _write_required_metrics_files(out_dir, required)
    fig_paths = _render_required_metric_figures(out_dir, required)
    summary["required_metrics"] = required
    summary["required_metric_files"] = {
        "json": str(req_json_path),
        "csv": str(req_csv_path),
    }
    summary["required_metric_figures"] = [str(p) for p in fig_paths]

    json_path = out_dir / "metrics_summary.json"
    md_path = out_dir / "metrics_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_format_md(summary) + "\n")

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {req_json_path}")
    print(f"Saved: {req_csv_path}")
    for p in fig_paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
