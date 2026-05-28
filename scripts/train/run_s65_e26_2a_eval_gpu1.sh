#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/lhr/VSD"
PY="/mnt/disk2/lhr/conda_envs/vsd/bin/python"
export VSD_E6_SCRIPTS_DIR="$ROOT/scripts"
export LD_LIBRARY_PATH="/mnt/disk2/lhr/conda_envs/vsd/lib:${LD_LIBRARY_PATH:-}"

GPU_DEVICE="${GPU_DEVICE:-1}"
VAL_BATCH="${VAL_BATCH:-64}"
VAL_WORKERS="${VAL_WORKERS:-8}"
OBJECT_BATCH="${OBJECT_BATCH:-64}"
OBJECT_WORKERS="${OBJECT_WORKERS:-8}"
EXPORT_BATCH="${EXPORT_BATCH:-64}"
EXPORT_WORKERS="${EXPORT_WORKERS:-8}"

cd "$ROOT"

NAME="e26_2a_class_confusion_cls125"
WEIGHTS="$ROOT/results/S6_5_reliability_calibration/${NAME}/weights/best.pt"
VAL_DIR="$ROOT/results/S6_5_reliability_calibration/${NAME}_val"
OBJ_DIR="$ROOT/results/S6_5_reliability_calibration/${NAME}_object_level"
PRED_DIR="$ROOT/results/S6_5_reliability_calibration/${NAME}_predictions"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "[S6.5-B] missing E26_2a weights: $WEIGHTS" >&2
  exit 1
fi

echo "[S6.5-B] E26_2a validation start"
"$PY" scripts/e13_val_tiny_aware_loss.py \
  --weights "$WEIGHTS" \
  --mode rgb_ir \
  --split val \
  --imgsz 640 \
  --batch "$VAL_BATCH" \
  --workers "$VAL_WORKERS" \
  --device "$GPU_DEVICE" \
  --out-dir "$VAL_DIR" \
  --exist-ok
echo "[S6.5-B] E26_2a validation done"

echo "[S6.5-B] E26_2a object-level start"
"$PY" scripts/e23_object_level_subset_eval.py \
  --weights "$WEIGHTS" \
  --validator e13 \
  --mode rgb_ir \
  --split val \
  --imgsz 640 \
  --batch "$OBJECT_BATCH" \
  --workers "$OBJECT_WORKERS" \
  --device "$GPU_DEVICE" \
  --image-metrics "$VAL_DIR/required_metrics.json" \
  --out-dir "$OBJ_DIR"
echo "[S6.5-B] E26_2a object-level done"

echo "[S6.5-B] E26_2a prediction export start"
"$PY" scripts/e25_e26_full_calibration.py export \
  --weights "$WEIGHTS" \
  --validator e13 \
  --split val \
  --out-dir "$PRED_DIR" \
  --imgsz 640 \
  --batch "$EXPORT_BATCH" \
  --workers "$EXPORT_WORKERS" \
  --device "$GPU_DEVICE" \
  --conf 0.01 \
  --nms-iou 0.70
echo "[S6.5-B] E26_2a prediction export done"
