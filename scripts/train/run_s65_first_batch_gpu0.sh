#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/lhr/VSD"
PY="/mnt/disk2/lhr/conda_envs/vsd/bin/python"
export VSD_E6_SCRIPTS_DIR="$ROOT/scripts"
export LD_LIBRARY_PATH="/mnt/disk2/lhr/conda_envs/vsd/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT"

echo "[S6.5-A] E25_1_full start"
"$PY" scripts/e25_e26_full_calibration.py e25_1_full \
  --device 0 \
  --batch 32 \
  --workers 8
echo "[S6.5-A] E25_1_full done"

echo "[S6.5-A] E26_1_full start"
"$PY" scripts/e25_e26_full_calibration.py e26_1_full
echo "[S6.5-A] E26_1_full done"

for seed in 42 43 44; do
  train_name="e25_0_e13_3b_seed${seed}"
  val_dir="$ROOT/results/val/${train_name}_val"
  obj_dir="$ROOT/results/val/${train_name}_object_level"
  pred_dir="$ROOT/results/val/${train_name}_predictions"
  weights="$ROOT/results/val/${train_name}/weights/best.pt"

  echo "[S6.5-A] E25_0 seed=${seed} train start"
  "$PY" scripts/e13_train_tiny_aware_loss.py \
    --mode rgb_ir \
    --loss center-aware \
    --loss-scope small \
    --aux-weight 0.5 \
    --small-px 32.0 \
    --center-alpha 0.25 \
    --center-max 4.0 \
    --model "$ROOT/results/val/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt" \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --workers 8 \
    --device 0 \
    --project "$ROOT/results/val" \
    --name "$train_name" \
    --seed "$seed" \
    --close-mosaic 10 \
    --patience 100 \
    --exist-ok
  echo "[S6.5-A] E25_0 seed=${seed} train done"

  echo "[S6.5-A] E25_0 seed=${seed} validation start"
  "$PY" scripts/e13_val_tiny_aware_loss.py \
    --weights "$weights" \
    --mode rgb_ir \
    --split val \
    --imgsz 640 \
    --batch 16 \
    --workers 8 \
    --device 0 \
    --out-dir "$val_dir" \
    --exist-ok
  echo "[S6.5-A] E25_0 seed=${seed} validation done"

  echo "[S6.5-A] E25_0 seed=${seed} object-level start"
  "$PY" scripts/e23_object_level_subset_eval.py \
    --weights "$weights" \
    --validator e13 \
    --mode rgb_ir \
    --split val \
    --imgsz 640 \
    --batch 16 \
    --workers 8 \
    --device 0 \
    --image-metrics "$val_dir/required_metrics.json" \
    --out-dir "$obj_dir"
  echo "[S6.5-A] E25_0 seed=${seed} object-level done"

  echo "[S6.5-A] E25_0 seed=${seed} prediction export start"
  "$PY" scripts/e25_e26_full_calibration.py export \
    --weights "$weights" \
    --validator e13 \
    --split val \
    --out-dir "$pred_dir" \
    --imgsz 640 \
    --batch 16 \
    --workers 8 \
    --device 0 \
    --conf 0.01 \
    --nms-iou 0.70
  echo "[S6.5-A] E25_0 seed=${seed} prediction export done"
done

echo "[S6.5-A] E25_0 audit start"
"$PY" scripts/e25_e26_offline_calibration.py e25_0
echo "[S6.5-A] E25_0 audit done"

echo "[S6.5-A] first batch complete"
