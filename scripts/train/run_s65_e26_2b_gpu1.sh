#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/lhr/VSD"
PY="/mnt/disk2/lhr/conda_envs/vsd/bin/python"
export VSD_E6_SCRIPTS_DIR="$ROOT/scripts"
export LD_LIBRARY_PATH="/mnt/disk2/lhr/conda_envs/vsd/lib:${LD_LIBRARY_PATH:-}"

GPU_DEVICE="${GPU_DEVICE:-1}"
TRAIN_BATCH="${TRAIN_BATCH:-48}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
VAL_BATCH="${VAL_BATCH:-64}"
VAL_WORKERS="${VAL_WORKERS:-8}"
OBJECT_BATCH="${OBJECT_BATCH:-64}"
OBJECT_WORKERS="${OBJECT_WORKERS:-8}"
EXPORT_BATCH="${EXPORT_BATCH:-64}"
EXPORT_WORKERS="${EXPORT_WORKERS:-8}"

cd "$ROOT"

NAME="e26_2b_class_confusion_cls150"
TRAIN_DIR="$ROOT/results/S6_5_reliability_calibration/${NAME}"
WEIGHTS="$TRAIN_DIR/weights/best.pt"
VAL_DIR="$ROOT/results/S6_5_reliability_calibration/${NAME}_val"
OBJ_DIR="$ROOT/results/S6_5_reliability_calibration/${NAME}_object_level"
PRED_DIR="$ROOT/results/S6_5_reliability_calibration/${NAME}_predictions"
CLASS_CONFUSION_MAP="$ROOT/results/S5_diagnostic_optimization/e22_0_train_hard_negative_taxonomy/hard_negative_list.csv"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "[S6.5-B] E26_2b train start"
  "$PY" scripts/e13_train_tiny_aware_loss.py \
    --mode rgb_ir \
    --loss class-confusion-cls \
    --class-confusion-map "$CLASS_CONFUSION_MAP" \
    --class-confusion-cls-gain 1.50 \
    --model "$ROOT/results/S2_fusion_mainline/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt" \
    --epochs 100 \
    --imgsz 640 \
    --batch "$TRAIN_BATCH" \
    --workers "$TRAIN_WORKERS" \
    --device "$GPU_DEVICE" \
    --project "$ROOT/results/S6_5_reliability_calibration" \
    --name "$NAME" \
    --seed 0 \
    --close-mosaic 10 \
    --patience 100 \
    --exist-ok
  echo "[S6.5-B] E26_2b train done"
else
  echo "[S6.5-B] E26_2b weights exist; skip train"
fi

if [[ ! -f "$WEIGHTS" ]]; then
  echo "[S6.5-B] missing E26_2b weights: $WEIGHTS" >&2
  exit 1
fi

echo "[S6.5-B] E26_2b validation start"
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
echo "[S6.5-B] E26_2b validation done"

echo "[S6.5-B] E26_2b object-level start"
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
echo "[S6.5-B] E26_2b object-level done"

echo "[S6.5-B] E26_2b prediction export start"
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
echo "[S6.5-B] E26_2b prediction export done"
