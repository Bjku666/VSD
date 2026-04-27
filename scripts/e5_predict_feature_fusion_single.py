#!/usr/bin/env python3
"""Inference script for YOLO11n E5 fusion model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

from e5_feature_fusion_single_core import E5SingleLayerFusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with E5 fusion model")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--mode", type=str, default="rgb_ir", choices=["rgb", "ir", "rgb_ir"])
    parser.add_argument("--rgb", type=str, required=True, help="Path to RGB image")
    parser.add_argument("--ir", type=str, default="", help="Path to IR image (required for rgb_ir mode)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--save-vis", action="store_true")
    return parser.parse_args()


def _load_checkpoint_model(weights: Path, device: torch.device, mode: str) -> tuple[torch.nn.Module, dict[int, str]]:
    ckpt = torch.load(str(weights), map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        model = ckpt.get("ema") or ckpt.get("model")
    else:
        model = ckpt
    if model is None:
        raise RuntimeError("Failed to find model object in checkpoint")

    if not isinstance(model, torch.nn.Module):
        raise TypeError("Checkpoint model object is not a torch.nn.Module")

    if isinstance(model, E5SingleLayerFusionModel):
        model.set_fusion_mode(mode)

    model = model.to(device).float().eval()
    names_obj = getattr(model, "names", {})
    names = {int(k): str(v) for k, v in dict(names_obj).items()} if isinstance(names_obj, dict) else {}
    return model, names


def _read_image(path: Path) -> Any:
    if not str(path):
        raise ValueError("Image path is empty")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _prepare_tensor(rgb_bgr, ir_bgr, mode: str, imgsz: int, device: torch.device):
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, scale_fill=False, scaleup=False, stride=32)

    rgb_lb = letterbox(image=rgb_bgr)
    rgb_rgb = cv2.cvtColor(rgb_lb, cv2.COLOR_BGR2RGB)
    rgb_tensor = torch.from_numpy(rgb_rgb).permute(2, 0, 1).contiguous().float() / 255.0

    if mode == "rgb_ir":
        if ir_bgr is None:
            raise ValueError("--ir is required in rgb_ir mode")
        ir_lb = letterbox(image=ir_bgr)
        ir_rgb = cv2.cvtColor(ir_lb, cv2.COLOR_BGR2RGB)
        ir_tensor = torch.from_numpy(ir_rgb).permute(2, 0, 1).contiguous().float() / 255.0
        inp = torch.cat((rgb_tensor, ir_tensor), dim=0)
    elif mode == "ir":
        if ir_bgr is None:
            raise ValueError("--ir is required in ir mode")
        ir_lb = letterbox(image=ir_bgr)
        ir_rgb = cv2.cvtColor(ir_lb, cv2.COLOR_BGR2RGB)
        inp = torch.from_numpy(ir_rgb).permute(2, 0, 1).contiguous().float() / 255.0
    else:
        inp = rgb_tensor

    return inp.unsqueeze(0).to(device), rgb_bgr.shape[:2]


def _postprocess(pred, inp_shape, orig_shape, conf: float, iou: float, max_det: int):
    pred = pred.clone()
    dets = non_max_suppression(
        pred,
        conf_thres=conf,
        iou_thres=iou,
        max_det=max_det,
        nc=0,
    )

    out = []
    for det in dets:
        if det is None or len(det) == 0:
            continue
        det = det.clone()
        det[:, :4] = ops.scale_boxes(inp_shape, det[:, :4], orig_shape)
        for row in det:
            x1, y1, x2, y2, score, cls_id = row.tolist()
            out.append(
                {
                    "xyxy": [x1, y1, x2, y2],
                    "score": score,
                    "class_id": int(cls_id),
                }
            )
    return out


def _draw_boxes(img, detections, names: dict[int, str]):
    vis = img.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        cid = int(d["class_id"])
        score = float(d["score"])
        label = f"{names.get(cid, str(cid))}:{score:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    weights = Path(args.weights)
    rgb_path = Path(args.rgb)
    ir_path = Path(args.ir) if args.ir else None

    model, names = _load_checkpoint_model(weights, device, args.mode)

    rgb_img = _read_image(rgb_path)
    ir_img = _read_image(ir_path) if ir_path is not None else None

    inp, orig_shape = _prepare_tensor(rgb_img, ir_img, args.mode, args.imgsz, device)
    with torch.inference_mode():
        raw = model(inp)

    pred = raw[0] if isinstance(raw, (tuple, list)) else raw
    detections = _postprocess(pred, inp.shape[2:], orig_shape, args.conf, args.iou, args.max_det)

    result = {
        "mode": args.mode,
        "weights": str(weights),
        "rgb": str(rgb_path),
        "ir": str(ir_path) if ir_path else None,
        "num_detections": len(detections),
        "detections": detections,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if args.save_vis:
        vis = _draw_boxes(rgb_img, detections, names)
        vis_path = Path(args.out).with_suffix(".jpg") if args.out else rgb_path.with_name(f"{rgb_path.stem}_e5_vis.jpg")
        cv2.imwrite(str(vis_path), vis)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
