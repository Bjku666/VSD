#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/lhr/VSD"
PY="/mnt/disk2/lhr/conda_envs/vsd/bin/python"
export VSD_E6_SCRIPTS_DIR="$ROOT/scripts"
export LD_LIBRARY_PATH="/mnt/disk2/lhr/conda_envs/vsd/lib:${LD_LIBRARY_PATH:-}"

DEVICE="${DEVICE:-0}"
TRAIN_BATCH="${TRAIN_BATCH:-48}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
EVAL_BATCH="${EVAL_BATCH:-48}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
EPOCHS="${EPOCHS:-100}"
TAG="${TAG:-seedfix_b48_20260527}"
BASE_E6_WEIGHTS="${BASE_E6_WEIGHTS:-$ROOT/results/val/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt}"

cd "$ROOT"

echo "[seedfix] device=${DEVICE} train_batch=${TRAIN_BATCH} train_workers=${TRAIN_WORKERS} eval_batch=${EVAL_BATCH} tag=${TAG}"
echo "[seedfix] base_e6_weights=${BASE_E6_WEIGHTS}"

export_eval() {
  local weights="$1"
  local validator="$2"
  local pred_dir="$3"
  "$PY" scripts/e25_e26_full_calibration.py export \
    --weights "$weights" \
    --validator "$validator" \
    --split val \
    --out-dir "$pred_dir" \
    --imgsz 640 \
    --batch "$EVAL_BATCH" \
    --workers "$EVAL_WORKERS" \
    --device "$DEVICE" \
    --conf 0.01 \
    --nms-iou 0.70 \
    --force
}

seed_outputs_complete() {
  local name="$1"
  [[ -f "$ROOT/results/val/${name}/weights/best.pt" ]] &&
    [[ -f "$ROOT/results/val/${name}_val/required_metrics.json" ]] &&
    [[ -f "$ROOT/results/val/${name}_object_level/required_metrics.json" ]] &&
    [[ -d "$ROOT/results/val/${name}_predictions/labels" ]]
}

eval_model() {
  local weights="$1"
  local validator="$2"
  local val_dir="$3"
  local obj_dir="$4"
  local pred_dir="$5"

  if [[ "$validator" == "e6" ]]; then
    "$PY" scripts/e6_val_feature_fusion_multiscale.py \
      --weights "$weights" \
      --mode rgb_ir \
      --split val \
      --imgsz 640 \
      --batch "$EVAL_BATCH" \
      --workers "$EVAL_WORKERS" \
      --device "$DEVICE" \
      --out-dir "$val_dir" \
      --exist-ok
  else
    "$PY" scripts/e13_val_tiny_aware_loss.py \
      --weights "$weights" \
      --mode rgb_ir \
      --split val \
      --imgsz 640 \
      --batch "$EVAL_BATCH" \
      --workers "$EVAL_WORKERS" \
      --device "$DEVICE" \
      --out-dir "$val_dir" \
      --exist-ok
  fi

  "$PY" scripts/e23_object_level_subset_eval.py \
    --weights "$weights" \
    --validator "$validator" \
    --mode rgb_ir \
    --split val \
    --imgsz 640 \
    --batch "$EVAL_BATCH" \
    --workers "$EVAL_WORKERS" \
    --device "$DEVICE" \
    --image-metrics "$val_dir/required_metrics.json" \
    --out-dir "$obj_dir"

  export_eval "$weights" "$validator" "$pred_dir"
}

train_e6_seed() {
  local seed="$1"
  local name="seedfix_e6_seed${seed}_${TAG}"
  local train_dir="$ROOT/results/val/${name}"
  local weights="$train_dir/weights/best.pt"

  if seed_outputs_complete "$name"; then
    echo "[seedfix] E6 seed=${seed} skip complete name=${name}"
    return 0
  fi

  echo "[seedfix] E6 seed=${seed} train start name=${name}"
  "$PY" scripts/e6_train_feature_fusion_multiscale.py \
    --mode rgb_ir \
    --model "$ROOT/weights/pretrained/yolo11n.pt" \
    --epochs "$EPOCHS" \
    --imgsz 640 \
    --batch "$TRAIN_BATCH" \
    --workers "$TRAIN_WORKERS" \
    --device "$DEVICE" \
    --project "$ROOT/results/val" \
    --name "$name" \
    --seed "$seed" \
    --close-mosaic 10 \
    --patience 100 \
    --exist-ok
  echo "[seedfix] E6 seed=${seed} train done"

  eval_model "$weights" e6 \
    "$ROOT/results/val/${name}_val" \
    "$ROOT/results/val/${name}_object_level" \
    "$ROOT/results/val/${name}_predictions"
  echo "[seedfix] E6 seed=${seed} eval/export done"
}

train_e13_3b_seed() {
  local seed="$1"
  local name="$2"
  local train_dir="$ROOT/results/val/${name}"
  local weights="$train_dir/weights/best.pt"

  if seed_outputs_complete "$name"; then
    echo "[seedfix] E13_3b seed=${seed} skip complete name=${name}"
    return 0
  fi

  echo "[seedfix] E13_3b seed=${seed} train start name=${name}"
  "$PY" scripts/e13_train_tiny_aware_loss.py \
    --mode rgb_ir \
    --loss center-aware \
    --loss-scope small \
    --aux-weight 0.5 \
    --small-px 32.0 \
    --center-alpha 0.25 \
    --center-max 4.0 \
    --model "$BASE_E6_WEIGHTS" \
    --epochs "$EPOCHS" \
    --imgsz 640 \
    --batch "$TRAIN_BATCH" \
    --workers "$TRAIN_WORKERS" \
    --device "$DEVICE" \
    --project "$ROOT/results/val" \
    --name "$name" \
    --seed "$seed" \
    --close-mosaic 10 \
    --patience 100 \
    --exist-ok
  echo "[seedfix] E13_3b seed=${seed} train done"

  eval_model "$weights" e13 \
    "$ROOT/results/val/${name}_val" \
    "$ROOT/results/val/${name}_object_level" \
    "$ROOT/results/val/${name}_predictions"
  echo "[seedfix] E13_3b seed=${seed} eval/export done"
}

for seed in 0 1 2; do
  train_e6_seed "$seed"
done

for seed in 0 1 2; do
  train_e13_3b_seed "$seed" "seedfix_e13_3b_seed${seed}_${TAG}"
done

echo "[seedfix] E25_0 seed42/43/44 are owned by GPU1 split queue; GPU0 queue stops after E13_3b seeds"
echo "[seedfix] corrected E6/E13_3b reruns complete"
