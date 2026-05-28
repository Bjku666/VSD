#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/lhr/VSD"
PY="/mnt/disk2/lhr/conda_envs/vsd/bin/python"
export VSD_E6_SCRIPTS_DIR="$ROOT/scripts"
export LD_LIBRARY_PATH="/mnt/disk2/lhr/conda_envs/vsd/lib:${LD_LIBRARY_PATH:-}"

DEVICE="${DEVICE:-1}"
TRAIN_BATCH="${TRAIN_BATCH:-48}"
TRAIN_WORKERS="${TRAIN_WORKERS:-12}"
EVAL_BATCH="${EVAL_BATCH:-64}"
EVAL_WORKERS="${EVAL_WORKERS:-12}"
EPOCHS="${EPOCHS:-100}"
TAG="${TAG:-seedfix_b48_20260527}"
BASE_E6_WEIGHTS="${BASE_E6_WEIGHTS:-$ROOT/results/S2_fusion_mainline/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt}"

cd "$ROOT"

echo "[seedfix-gpu1] device=${DEVICE} train_batch=${TRAIN_BATCH} train_workers=${TRAIN_WORKERS} eval_batch=${EVAL_BATCH} tag=${TAG}"
echo "[seedfix-gpu1] base_e6_weights=${BASE_E6_WEIGHTS}"

seed_outputs_complete() {
  local name="$1"
  [[ -f "$ROOT/results/S6_5_reliability_calibration/${name}/weights/best.pt" ]] &&
    [[ -f "$ROOT/results/S6_5_reliability_calibration/${name}_val/required_metrics.json" ]] &&
    [[ -f "$ROOT/results/S6_5_reliability_calibration/${name}_object_level/required_metrics.json" ]] &&
    [[ -d "$ROOT/results/S6_5_reliability_calibration/${name}_predictions/labels" ]]
}

export_eval() {
  local weights="$1"
  local pred_dir="$2"
  "$PY" scripts/e25_e26_full_calibration.py export \
    --weights "$weights" \
    --validator e13 \
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

eval_model() {
  local weights="$1"
  local val_dir="$2"
  local obj_dir="$3"
  local pred_dir="$4"

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

  "$PY" scripts/e23_object_level_subset_eval.py \
    --weights "$weights" \
    --validator e13 \
    --mode rgb_ir \
    --split val \
    --imgsz 640 \
    --batch "$EVAL_BATCH" \
    --workers "$EVAL_WORKERS" \
    --device "$DEVICE" \
    --image-metrics "$val_dir/required_metrics.json" \
    --out-dir "$obj_dir"

  export_eval "$weights" "$pred_dir"
}

train_e25_seed() {
  local seed="$1"
  local name="seedfix_e25_0_e13_3b_seed${seed}_${TAG}"
  local train_dir="$ROOT/results/S6_5_reliability_calibration/${name}"
  local weights="$train_dir/weights/best.pt"
  local last_weights="$train_dir/weights/last.pt"
  local resume_args=()

  if seed_outputs_complete "$name"; then
    echo "[seedfix-gpu1] E25_0 seed=${seed} skip complete name=${name}"
    return 0
  fi

  if [[ -f "$last_weights" ]]; then
    echo "[seedfix-gpu1] E25_0 seed=${seed} resume from ${last_weights}"
    resume_args=(--resume-path "$last_weights")
  fi

  echo "[seedfix-gpu1] E25_0 seed=${seed} train start name=${name}"
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
    --project "$ROOT/results/S6_5_reliability_calibration" \
    --name "$name" \
    --seed "$seed" \
    --close-mosaic 10 \
    --patience 100 \
    --exist-ok \
    "${resume_args[@]}"
  echo "[seedfix-gpu1] E25_0 seed=${seed} train done"

  eval_model "$weights" \
    "$ROOT/results/S6_5_reliability_calibration/${name}_val" \
    "$ROOT/results/S6_5_reliability_calibration/${name}_object_level" \
    "$ROOT/results/S6_5_reliability_calibration/${name}_predictions"
  echo "[seedfix-gpu1] E25_0 seed=${seed} eval/export done"
}

for seed in 42 43 44; do
  train_e25_seed "$seed"
done

echo "[seedfix-gpu1] all E25_0 corrected seed reruns complete"
