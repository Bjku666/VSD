#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/lhr/VSD"
PY="/mnt/disk2/lhr/conda_envs/vsd/bin/python"
export VSD_E6_SCRIPTS_DIR="$ROOT/scripts"
export LD_LIBRARY_PATH="/mnt/disk2/lhr/conda_envs/vsd/lib:${LD_LIBRARY_PATH:-}"
GPU_DEVICE="${GPU_DEVICE:-1}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
TRAIN_RESUME="${TRAIN_RESUME:-1}"
VAL_BATCH="${VAL_BATCH:-64}"
VAL_WORKERS="${VAL_WORKERS:-8}"
OBJECT_BATCH="${OBJECT_BATCH:-64}"
OBJECT_WORKERS="${OBJECT_WORKERS:-8}"
EXPORT_BATCH="${EXPORT_BATCH:-64}"
EXPORT_WORKERS="${EXPORT_WORKERS:-8}"

cd "$ROOT"
echo "[S6.5-A] config GPU=${GPU_DEVICE} train_batch=${TRAIN_BATCH} train_workers=${TRAIN_WORKERS} train_resume=${TRAIN_RESUME} val_batch=${VAL_BATCH} object_batch=${OBJECT_BATCH} export_batch=${EXPORT_BATCH}"

train_seed() {
  local seed="$1"
  local train_name="e25_0_e13_3b_seed${seed}"
  local train_dir="$ROOT/results/S6_5_reliability_calibration/${train_name}"
  local weights="$train_dir/weights/best.pt"

  if [[ -f "$ROOT/results/S6_5_reliability_calibration/${train_name}_val/required_metrics.json" \
     && -f "$ROOT/results/S6_5_reliability_calibration/${train_name}_object_level/required_metrics.json" \
     && -d "$ROOT/results/S6_5_reliability_calibration/${train_name}_predictions/labels" ]]; then
    echo "[S6.5-A] E25_0 seed=${seed} complete outputs exist; skip train/val/object/export"
    return
  fi

  if [[ "$TRAIN_RESUME" == "1" && -f "$train_dir/weights/last.pt" && ! -f "$ROOT/results/S6_5_reliability_calibration/${train_name}_val/required_metrics.json" ]]; then
    echo "[S6.5-A] E25_0 seed=${seed} resume train from last.pt on GPU${GPU_DEVICE}"
    "$PY" scripts/e13_train_tiny_aware_loss.py \
      --mode rgb_ir \
      --loss center-aware \
      --loss-scope small \
      --aux-weight 0.5 \
      --small-px 32.0 \
      --center-alpha 0.25 \
      --center-max 4.0 \
      --model "$train_dir/weights/last.pt" \
      --epochs 100 \
      --imgsz 640 \
      --batch "$TRAIN_BATCH" \
      --workers "$TRAIN_WORKERS" \
      --device "$GPU_DEVICE" \
      --project "$ROOT/results/S6_5_reliability_calibration" \
      --name "$train_name" \
      --seed "$seed" \
      --close-mosaic 10 \
      --patience 100 \
      --resume \
      --exist-ok
  elif [[ ! -f "$weights" || ! -f "$ROOT/results/S6_5_reliability_calibration/${train_name}_val/required_metrics.json" ]]; then
    echo "[S6.5-A] E25_0 seed=${seed} fresh train on GPU${GPU_DEVICE}"
    "$PY" scripts/e13_train_tiny_aware_loss.py \
      --mode rgb_ir \
      --loss center-aware \
      --loss-scope small \
      --aux-weight 0.5 \
      --small-px 32.0 \
      --center-alpha 0.25 \
      --center-max 4.0 \
      --model "$ROOT/results/S2_fusion_mainline/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt" \
      --epochs 100 \
      --imgsz 640 \
      --batch "$TRAIN_BATCH" \
      --workers "$TRAIN_WORKERS" \
      --device "$GPU_DEVICE" \
      --project "$ROOT/results/S6_5_reliability_calibration" \
      --name "$train_name" \
      --seed "$seed" \
      --close-mosaic 10 \
      --patience 100 \
      --exist-ok
  fi

  weights="$train_dir/weights/best.pt"
  if [[ ! -f "$weights" ]]; then
    echo "[S6.5-A] missing best.pt for seed=${seed}: $weights" >&2
    exit 1
  fi

  local val_dir="$ROOT/results/S6_5_reliability_calibration/${train_name}_val"
  local obj_dir="$ROOT/results/S6_5_reliability_calibration/${train_name}_object_level"
  local pred_dir="$ROOT/results/S6_5_reliability_calibration/${train_name}_predictions"

  if [[ ! -f "$val_dir/required_metrics.json" ]]; then
    echo "[S6.5-A] E25_0 seed=${seed} validation start"
    "$PY" scripts/e13_val_tiny_aware_loss.py \
      --weights "$weights" \
      --mode rgb_ir \
      --split val \
      --imgsz 640 \
      --batch "$VAL_BATCH" \
      --workers "$VAL_WORKERS" \
      --device "$GPU_DEVICE" \
      --out-dir "$val_dir" \
      --exist-ok
    echo "[S6.5-A] E25_0 seed=${seed} validation done"
  fi

  if [[ ! -f "$obj_dir/required_metrics.json" ]]; then
    echo "[S6.5-A] E25_0 seed=${seed} object-level start"
    "$PY" scripts/e23_object_level_subset_eval.py \
      --weights "$weights" \
      --validator e13 \
      --mode rgb_ir \
      --split val \
      --imgsz 640 \
      --batch "$OBJECT_BATCH" \
      --workers "$OBJECT_WORKERS" \
      --device "$GPU_DEVICE" \
      --image-metrics "$val_dir/required_metrics.json" \
      --out-dir "$obj_dir"
    echo "[S6.5-A] E25_0 seed=${seed} object-level done"
  fi

  if [[ ! -d "$pred_dir/labels" ]]; then
    echo "[S6.5-A] E25_0 seed=${seed} prediction export start"
    "$PY" scripts/e25_e26_full_calibration.py export \
      --weights "$weights" \
      --validator e13 \
      --split val \
      --out-dir "$pred_dir" \
      --imgsz 640 \
      --batch "$EXPORT_BATCH" \
      --workers "$EXPORT_WORKERS" \
      --device "$GPU_DEVICE" \
      --conf 0.01 \
      --nms-iou 0.70
    echo "[S6.5-A] E25_0 seed=${seed} prediction export done"
  fi
}

for seed in ${SEEDS:-42 43 44}; do
  train_seed "$seed"
done

echo "[S6.5-A] E25_0 audit start"
"$PY" scripts/e25_e26_offline_calibration.py e25_0
echo "[S6.5-A] E25_0 audit done"
