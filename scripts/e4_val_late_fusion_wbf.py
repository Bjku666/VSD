#!/usr/bin/env python3
"""Late Fusion evaluation for paired RGB/IR DroneVehicle models.

Outputs:
- Standard metrics: mAP50, mAP50-95, Precision, Recall.
- Task metrics: AP_dark, Recall_dark, AP_small, Recall_small, AP_dark-small.
- Error metrics: FP per image, background confusion stats, confusion matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import yaml
from ensemble_boxes import weighted_boxes_fusion
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _iter_image_paths(path_or_txt: Path) -> List[Path]:
    if path_or_txt.is_dir():
        return [
            p
            for p in sorted(path_or_txt.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]

    if path_or_txt.is_file() and path_or_txt.suffix.lower() == ".txt":
        items: List[Path] = []
        with open(path_or_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(Path(line))
        return items

    if path_or_txt.is_file() and path_or_txt.suffix.lower() in IMAGE_SUFFIXES:
        return [path_or_txt]

    return []


def _resolve_data_yaml_images(data_yaml: Path, split: str) -> Tuple[List[Path], Dict[int, str]]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", "."))
    split_entry = cfg.get(split)
    if split_entry is None:
        raise ValueError(f"Split '{split}' not found in {data_yaml}")

    entries: Sequence[str]
    if isinstance(split_entry, (list, tuple)):
        entries = [str(x) for x in split_entry]
    else:
        entries = [str(split_entry)]

    images: List[Path] = []
    for item in entries:
        p = Path(item)
        if not p.is_absolute():
            p = root / p
        for ip in _iter_image_paths(p):
            images.append(ip if ip.is_absolute() else (root / ip))

    names_cfg = cfg.get("names", {})
    names: Dict[int, str] = {}
    if isinstance(names_cfg, dict):
        for k, v in names_cfg.items():
            try:
                names[int(k)] = str(v)
            except Exception:
                continue
    elif isinstance(names_cfg, list):
        for i, v in enumerate(names_cfg):
            names[i] = str(v)

    return sorted(set(images)), names


def _pair_rgb_ir(rgb_images: List[Path], ir_images: List[Path]) -> List[Tuple[Path, Path, str]]:
    rgb_map = {p.stem: p for p in rgb_images}
    ir_map = {p.stem: p for p in ir_images}
    common = sorted(set(rgb_map.keys()) & set(ir_map.keys()))
    pairs = [(rgb_map[s], ir_map[s], s) for s in common]
    return pairs


def _label_path_from_image(img_path: Path) -> Path:
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
            ps = line.strip().split()
            if len(ps) != 5:
                continue
            try:
                cls_id = int(ps[0])
                xc, yc, bw, bh = map(float, ps[1:])
            except Exception:
                continue

            x1 = (xc - bw / 2.0) * width
            y1 = (yc - bh / 2.0) * height
            x2 = (xc + bw / 2.0) * width
            y2 = (yc + bh / 2.0) * height
            boxes.append((cls_id, (x1, y1, x2, y2)))
    return boxes


def _clip_norm_box(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = box
    x1 = min(1.0, max(0.0, float(x1)))
    y1 = min(1.0, max(0.0, float(y1)))
    x2 = min(1.0, max(0.0, float(x2)))
    y2 = min(1.0, max(0.0, float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _predict_by_path(
    model: YOLO,
    image_paths: List[Path],
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    batch: int,
) -> Dict[str, Dict[str, Any]]:
    pred_map: Dict[str, Dict[str, Any]] = {}

    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        stream=True,
        verbose=False,
        batch=max(1, int(batch)),
        save=False,
        half=(str(device).lower() != "cpu"),
    )

    for img_path, res in zip(image_paths, results):
        stem = img_path.stem
        h, w = res.orig_shape

        boxes_norm: List[List[float]] = []
        boxes_abs: List[Tuple[float, float, float, float]] = []
        scores: List[float] = []
        labels: List[int] = []

        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(float, xyxy[i])
                boxes_abs.append((x1, y1, x2, y2))
                boxes_norm.append(_clip_norm_box([x1 / w, y1 / h, x2 / w, y2 / h]))
                scores.append(float(confs[i]))
                labels.append(int(clss[i]))

        pred_map[stem] = {
            "width": int(w),
            "height": int(h),
            "boxes_norm": boxes_norm,
            "boxes_abs": boxes_abs,
            "scores": scores,
            "labels": labels,
        }

    if len(pred_map) != len(image_paths):
        raise RuntimeError(
            f"Prediction count mismatch: got {len(pred_map)}, expected {len(image_paths)}"
        )

    return pred_map


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
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _match_pairs(
    gt_boxes: List[Tuple[int, Tuple[float, float, float, float]]],
    pred_boxes: List[Tuple[int, float, Tuple[float, float, float, float]]],
    iou_thr: float,
) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[int, int, float]] = []
    for gi, (_gcls, gxyxy) in enumerate(gt_boxes):
        for pi, (_pcls, _pscore, pxyxy) in enumerate(pred_boxes):
            iou = _iou(gxyxy, pxyxy)
            if iou >= iou_thr:
                pairs.append((gi, pi, iou))

    pairs.sort(key=lambda x: x[2], reverse=True)
    used_g = set()
    used_p = set()
    matches: List[Tuple[int, int, float]] = []
    for gi, pi, iou in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, iou))
    return matches


def _nms_indices(
    boxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    iou_thr: float,
) -> List[int]:
    if not boxes:
        return []

    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    order = np.argsort(s)[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(b[i, 0], b[rest, 0])
        yy1 = np.maximum(b[i, 1], b[rest, 1])
        xx2 = np.minimum(b[i, 2], b[rest, 2])
        yy2 = np.minimum(b[i, 3], b[rest, 3])
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih

        ai = np.maximum(0.0, b[i, 2] - b[i, 0]) * np.maximum(0.0, b[i, 3] - b[i, 1])
        ar = np.maximum(0.0, b[rest, 2] - b[rest, 0]) * np.maximum(0.0, b[rest, 3] - b[rest, 1])
        union = ai + ar - inter
        ious = np.where(union > 0.0, inter / union, 0.0)

        order = rest[ious <= iou_thr]

    return keep


def _evaluate_coco(
    coco_gt: COCO,
    detections: List[Dict[str, Any]],
    class_names: Dict[int, str],
    img_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if not detections:
        return {
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "Recall": 0.0,
            "per_class_AP": {v: {"AP50": 0.0, "AP50-95": 0.0} for _, v in sorted(class_names.items())},
        }

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    if img_ids is not None:
        coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    m_ap = _to_float(coco_eval.stats[0])
    m_ap50 = _to_float(coco_eval.stats[1])
    recall = _to_float(coco_eval.stats[8])

    per_class: Dict[str, Dict[str, float]] = {}
    precision = coco_eval.eval.get("precision", None)
    if precision is not None and precision.ndim == 5:
        for cls_idx, cls_name in sorted(class_names.items()):
            cls_pos = cls_idx
            p_all = precision[:, :, cls_pos, 0, -1]
            p_all = p_all[p_all > -1]
            ap = float(np.mean(p_all)) if p_all.size > 0 else 0.0

            p_50 = precision[0, :, cls_pos, 0, -1]
            p_50 = p_50[p_50 > -1]
            ap50 = float(np.mean(p_50)) if p_50.size > 0 else 0.0
            per_class[cls_name] = {"AP50": ap50, "AP50-95": ap}

    return {
        "mAP50": m_ap50,
        "mAP50-95": m_ap,
        "Recall": recall,
        "per_class_AP": per_class,
    }


def _save_confusion_matrix(
    matrix: np.ndarray,
    class_names: Dict[int, str],
    out_csv: Path,
    out_png: Path,
) -> None:
    labels = [class_names[i] for i in sorted(class_names.keys())] + ["background"]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred"] + labels)
        for i, row_name in enumerate(labels):
            writer.writerow([row_name] + [int(x) for x in matrix[i].tolist()])

    disp = matrix.astype(float)
    row_sums = disp.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm = disp / row_sums

    plt.figure(figsize=(8, 7))
    plt.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.title("Confusion Matrix (row-normalized)")
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=30, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _build_coco_eval_from_yaml(
    data_yaml: Path,
    split: str,
    class_names: Dict[int, str],
    pred_by_stem: Dict[str, Dict[str, Any]],
) -> Tuple[COCO, List[Dict[str, Any]], int]:
    images, _ = _resolve_data_yaml_images(data_yaml, split)

    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    coco_dets: List[Dict[str, Any]] = []

    ann_id = 1
    img_id = 1
    for img_path in images:
        stem = img_path.stem
        if stem not in pred_by_stem:
            continue

        pred = pred_by_stem[stem]
        w, h = int(pred["width"]), int(pred["height"])
        coco_images.append({"id": img_id, "file_name": stem + ".jpg", "width": w, "height": h})

        gt_boxes = _load_gt_boxes(_label_path_from_image(img_path), w, h)
        for cls_id, (x1, y1, x2, y2) in gt_boxes:
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls_id) + 1,
                    "bbox": [float(x1), float(y1), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        for cls_id, score, (x1, y1, x2, y2) in pred.get("fused_pred", []):
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            coco_dets.append(
                {
                    "image_id": img_id,
                    "category_id": int(cls_id) + 1,
                    "bbox": [float(x1), float(y1), float(bw), float(bh)],
                    "score": float(score),
                }
            )

        img_id += 1

    categories = [{"id": i + 1, "name": class_names[i]} for i in sorted(class_names.keys())]
    coco_gt_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()
    return coco_gt, coco_dets, len(coco_images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RGB+IR late fusion with NMS or WBF.")
    parser.add_argument("--method", type=str, default="late_wbf", choices=["late_nms", "late_wbf"])
    parser.add_argument(
        "--rgb-model",
        type=str,
        default="/mnt/disk2/lhr/VSD/experiments/e1_rgb_only/e1_yolo11n_rgb_only_640_ddp_restart/weights/best.pt",
    )
    parser.add_argument(
        "--ir-model",
        type=str,
        default="/mnt/disk2/lhr/VSD/experiments/e2_ir_only/e2_yolo11n_ir_only_640_ddp_restart/weights/best.pt",
    )
    parser.add_argument(
        "--rgb-data",
        type=Path,
        default=Path("/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb.yaml"),
    )
    parser.add_argument(
        "--ir-data",
        type=Path,
        default=Path("/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml"),
    )
    parser.add_argument(
        "--data-small",
        type=Path,
        default=Path("/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/ir_small.yaml"),
    )
    parser.add_argument(
        "--data-dark",
        type=Path,
        default=Path("/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/ir_dark.yaml"),
    )
    parser.add_argument(
        "--data-dark-small",
        type=Path,
        default=Path("/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/subsets/ir_dark-small.yaml"),
    )
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--pred-conf", type=float, default=0.001)
    parser.add_argument("--pred-iou", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--nms-iou", type=float, default=0.55)
    parser.add_argument("--wbf-iou", type=float, default=0.55)
    parser.add_argument("--wbf-skip-box-thr", type=float, default=0.001)
    parser.add_argument("--wbf-weight-rgb", type=float, default=1.0)
    parser.add_argument("--wbf-weight-ir", type=float, default=1.0)
    parser.add_argument("--analysis-conf", type=float, default=0.25)
    parser.add_argument("--analysis-iou", type=float, default=0.5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for fusion experiment artifacts.",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        if args.method == "late_nms":
            out_dir = Path(f"/mnt/disk2/lhr/VSD/results/{args.split}/e3_late_fusion_nms_{args.split}")
        else:
            out_dir = Path(f"/mnt/disk2/lhr/VSD/results/{args.split}/e4_late_fusion_wbf_{args.split}")
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_images, class_names = _resolve_data_yaml_images(args.rgb_data, args.split)
    ir_images, class_names_ir = _resolve_data_yaml_images(args.ir_data, args.split)
    if class_names != class_names_ir:
        raise ValueError("RGB/IR class names mismatch.")

    pairs = _pair_rgb_ir(rgb_images, ir_images)
    if not pairs:
        raise RuntimeError("No paired RGB/IR images found.")

    rgb_paths = [x[0] for x in pairs]
    ir_paths = [x[1] for x in pairs]
    stems = [x[2] for x in pairs]

    print(f"Paired images: {len(pairs)}")

    rgb_model = YOLO(args.rgb_model)
    ir_model = YOLO(args.ir_model)

    rgb_pred = _predict_by_path(
        model=rgb_model,
        image_paths=rgb_paths,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.pred_conf,
        iou=args.pred_iou,
        batch=args.batch,
    )
    ir_pred = _predict_by_path(
        model=ir_model,
        image_paths=ir_paths,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.pred_conf,
        iou=args.pred_iou,
        batch=args.batch,
    )

    nc = len(class_names)
    bg_idx = nc
    conf_mat = np.zeros((nc + 1, nc + 1), dtype=np.int64)

    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    coco_dets: List[Dict[str, Any]] = []
    ann_id = 1
    pred_by_stem: Dict[str, Dict[str, Any]] = {}

    for i, (rgb_p, ir_p, stem) in enumerate(pairs, start=1):
        if stem not in rgb_pred or stem not in ir_pred:
            raise KeyError(f"Missing prediction for image stem: {stem}")
        rp = rgb_pred[stem]
        ip = ir_pred[stem]
        w, h = int(rp["width"]), int(rp["height"])

        coco_images.append({"id": i, "file_name": stem + ".jpg", "width": w, "height": h})

        gt_boxes = _load_gt_boxes(_label_path_from_image(rgb_p), w, h)
        for cls_id, (x1, y1, x2, y2) in gt_boxes:
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": int(cls_id) + 1,
                    "bbox": [float(x1), float(y1), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        fused_pred: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
        if args.method == "late_wbf":
            boxes_list = [rp["boxes_norm"], ip["boxes_norm"]]
            scores_list = [rp["scores"], ip["scores"]]
            labels_list = [rp["labels"], ip["labels"]]
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=[args.wbf_weight_rgb, args.wbf_weight_ir],
                iou_thr=args.wbf_iou,
                skip_box_thr=args.wbf_skip_box_thr,
            )

            for b, s, l in zip(fused_boxes, fused_scores, fused_labels):
                x1 = float(b[0]) * w
                y1 = float(b[1]) * h
                x2 = float(b[2]) * w
                y2 = float(b[3]) * h
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                cls_id = int(l)
                score = float(s)
                coco_dets.append(
                    {
                        "image_id": i,
                        "category_id": cls_id + 1,
                        "bbox": [x1, y1, bw, bh],
                        "score": score,
                    }
                )
                fused_pred.append((cls_id, score, (x1, y1, x2, y2)))
        else:
            cls_to_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}
            cls_to_scores: Dict[int, List[float]] = {}
            for pred in (rp, ip):
                for c, s, b in zip(pred["labels"], pred["scores"], pred["boxes_abs"]):
                    c_int = int(c)
                    cls_to_boxes.setdefault(c_int, []).append(tuple(map(float, b)))
                    cls_to_scores.setdefault(c_int, []).append(float(s))

            for cls_id in sorted(cls_to_boxes.keys()):
                boxes_cls = cls_to_boxes[cls_id]
                scores_cls = cls_to_scores[cls_id]
                keep = _nms_indices(boxes_cls, scores_cls, args.nms_iou)
                for k in keep:
                    x1, y1, x2, y2 = boxes_cls[k]
                    score = scores_cls[k]
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                    coco_dets.append(
                        {
                            "image_id": i,
                            "category_id": cls_id + 1,
                            "bbox": [x1, y1, bw, bh],
                            "score": score,
                        }
                    )
                    fused_pred.append((cls_id, float(score), (x1, y1, x2, y2)))

        analysis_pred = [p for p in fused_pred if p[1] >= args.analysis_conf]
        matches = _match_pairs(gt_boxes, analysis_pred, args.analysis_iou)
        match_g = {gi: pi for gi, pi, _ in matches}
        matched_p = {pi for _gi, pi, _ in matches}

        pred_by_stem[stem] = {
            "width": w,
            "height": h,
            "fused_pred": fused_pred,
        }

        for gi, (gcls, _gxy) in enumerate(gt_boxes):
            if gi in match_g:
                pred_cls = analysis_pred[match_g[gi]][0]
                conf_mat[gcls, pred_cls] += 1
            else:
                conf_mat[gcls, bg_idx] += 1

        for pi, (pcls, _ps, _pxy) in enumerate(analysis_pred):
            if pi not in matched_p:
                conf_mat[bg_idx, pcls] += 1

    categories = [{"id": i + 1, "name": class_names[i]} for i in sorted(class_names.keys())]
    coco_gt_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    full_metrics = _evaluate_coco(coco_gt, coco_dets, class_names)

    small_gt, small_dets, n_small_imgs = _build_coco_eval_from_yaml(
        data_yaml=args.data_small,
        split=args.split,
        class_names=class_names,
        pred_by_stem=pred_by_stem,
    )
    dark_gt, dark_dets, n_dark_imgs = _build_coco_eval_from_yaml(
        data_yaml=args.data_dark,
        split=args.split,
        class_names=class_names,
        pred_by_stem=pred_by_stem,
    )
    dark_small_gt, dark_small_dets, n_dark_small_imgs = _build_coco_eval_from_yaml(
        data_yaml=args.data_dark_small,
        split=args.split,
        class_names=class_names,
        pred_by_stem=pred_by_stem,
    )

    small_metrics = _evaluate_coco(small_gt, small_dets, class_names)
    dark_metrics = _evaluate_coco(dark_gt, dark_dets, class_names)
    dark_small_metrics = _evaluate_coco(dark_small_gt, dark_small_dets, class_names)

    tp = int(np.trace(conf_mat[:nc, :nc]))
    offdiag = int(np.sum(conf_mat[:nc, :nc]) - tp)
    bg_fp = int(np.sum(conf_mat[bg_idx, :nc]))
    bg_fn = int(np.sum(conf_mat[:nc, bg_idx]))
    fp_total = offdiag + bg_fp
    fn_total = offdiag + bg_fn

    precision = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
    recall = tp / (tp + fn_total) if (tp + fn_total) > 0 else 0.0

    n_images = len(coco_images)
    fp_per_image = bg_fp / n_images if n_images > 0 else 0.0
    total_gt = int(np.sum(conf_mat[:nc, :]))
    background_confusion_rate = (bg_fp + bg_fn) / total_gt if total_gt > 0 else 0.0

    summary: Dict[str, Any] = {
        "experiment": {
            "type": args.method,
            "rgb_model": args.rgb_model,
            "ir_model": args.ir_model,
            "split": args.split,
            "imgsz": args.imgsz,
            "device": args.device,
            "pred_conf": args.pred_conf,
            "pred_iou": args.pred_iou,
            "nms_iou": args.nms_iou,
            "wbf_iou": args.wbf_iou,
            "wbf_skip_box_thr": args.wbf_skip_box_thr,
            "wbf_weights": {
                "rgb": args.wbf_weight_rgb,
                "ir": args.wbf_weight_ir,
            },
            "analysis_conf": args.analysis_conf,
            "analysis_iou": args.analysis_iou,
            "paired_images": n_images,
        },
        "standard_metrics": {
            "mAP50": full_metrics["mAP50"],
            "mAP50-95": full_metrics["mAP50-95"],
            "Precision": precision,
            "Recall": recall,
            "Recall_coco_AR100": full_metrics["Recall"],
            "per_class_AP": full_metrics.get("per_class_AP", {}),
        },
        "task_metrics": {
            "AP_dark": dark_metrics["mAP50-95"],
            "Recall_dark": dark_metrics["Recall"],
            "AP_small": small_metrics["mAP50-95"],
            "Recall_small": small_metrics["Recall"],
            "AP_dark-small": dark_small_metrics["mAP50-95"],
        },
        "error_metrics": {
            "false_positives": bg_fp,
            "false_positives_per_image": fp_per_image,
            "background_confusion_rate": background_confusion_rate,
            "background_false_positive": bg_fp,
            "background_false_negative": bg_fn,
            "confusion_offdiag": offdiag,
        },
        "subset_image_count": {
            "small": n_small_imgs,
            "dark": n_dark_imgs,
            "dark_small": n_dark_small_imgs,
        },
    }

    required = {
        "mAP50": summary["standard_metrics"]["mAP50"],
        "mAP50-95": summary["standard_metrics"]["mAP50-95"],
        "Precision": summary["standard_metrics"]["Precision"],
        "Recall": summary["standard_metrics"]["Recall"],
        "per_class_AP": summary["standard_metrics"].get("per_class_AP", {}),
        "AP_dark": summary["task_metrics"]["AP_dark"],
        "Recall_dark": summary["task_metrics"]["Recall_dark"],
        "AP_small": summary["task_metrics"]["AP_small"],
        "Recall_small": summary["task_metrics"]["Recall_small"],
        "AP_dark-small": summary["task_metrics"]["AP_dark-small"],
    }

    summary["required_metrics"] = required

    metrics_json = out_dir / "metrics_summary.json"
    metrics_md = out_dir / "metrics_summary.md"
    req_json = out_dir / "required_metrics.json"
    req_csv = out_dir / "required_metrics.csv"
    conf_csv = out_dir / "confusion_matrix.csv"
    conf_png = out_dir / "confusion_matrix.png"
    conf_norm_png = out_dir / "confusion_matrix_normalized.png"

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(req_json, "w", encoding="utf-8") as f:
        json.dump(required, f, ensure_ascii=False, indent=2)

    with open(req_csv, "w", encoding="utf-8", newline="") as f:
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
        for cls_name, vals in required["per_class_AP"].items():
            writer.writerow([
                cls_name,
                f"{_to_float(vals.get('AP50', 0.0)):.6f}",
                f"{_to_float(vals.get('AP50-95', 0.0)):.6f}",
            ])

    _save_confusion_matrix(conf_mat, class_names, conf_csv, conf_png)
    if conf_png.exists():
        shutil.copy2(conf_png, conf_norm_png)

    summary["required_metric_files"] = {
        "json": str(req_json),
        "csv": str(req_csv),
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_lines = [
        f"# Late Fusion ({args.method}) 指标报告",
        "",
        "## 核心指标",
        f"- mAP50: {required['mAP50']:.6f}",
        f"- mAP50-95: {required['mAP50-95']:.6f}",
        f"- Precision: {required['Precision']:.6f}",
        f"- Recall: {required['Recall']:.6f}",
        "",
        "## 重点任务指标",
        f"- AP_dark: {required['AP_dark']:.6f}",
        f"- Recall_dark: {required['Recall_dark']:.6f}",
        f"- AP_small: {required['AP_small']:.6f}",
        f"- Recall_small: {required['Recall_small']:.6f}",
        f"- AP_dark-small: {required['AP_dark-small']:.6f}",
        "",
        "## 误报与背景混淆",
        f"- False Positives / image: {summary['error_metrics']['false_positives_per_image']:.6f}",
        f"- 背景混淆率: {summary['error_metrics']['background_confusion_rate']:.6f}",
        f"- Confusion Matrix CSV: {conf_csv}",
        f"- Confusion Matrix PNG: {conf_png}",
        f"- Confusion Matrix Normalized PNG: {conf_norm_png}",
        "",
    ]
    with open(metrics_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Saved: {metrics_json}")
    print(f"Saved: {metrics_md}")
    print(f"Saved: {req_json}")
    print(f"Saved: {req_csv}")
    print(f"Saved: {conf_csv}")
    print(f"Saved: {conf_png}")
    print(f"Saved: {conf_norm_png}")


if __name__ == "__main__":
    main()
