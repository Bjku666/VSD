#!/bin/bash
# 评估 YOLO11n IR-only baseline，并导出 val/test 的标准与自定义指标。
# 用法：
#   ./scripts/e2_val_ir_only.sh [split] [model] [out_dir] [device] [case_topk] [case_max_images] [ultra_project] [skip_case_viz] [case_device] [case_batch] [imgsz]
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/disk2/lhr/conda_envs/vsd

SPLIT="${1:-val}"
MODEL_PATH="${2:-/mnt/disk2/lhr/VSD/experiments/e2_ir_only/e2_yolo11n_ir_only_640_ddp/weights/best.pt}"
RUN_NAME="$(basename "$(dirname "$(dirname "$MODEL_PATH")")")"
OUT_DIR="${3:-/mnt/disk2/lhr/VSD/experiments/e2_ir_only/${RUN_NAME}/${SPLIT}}"
DEVICE="${4:-0}"
CASE_TOPK="${5:-20}"
CASE_MAX_IMAGES="${6:-0}"
ULTRA_PROJECT="${7:-${OUT_DIR}/ultralytics_val}"
SKIP_CASE_VIZ="${8:-0}"
CASE_DEVICE="${9:-}"
CASE_BATCH="${10:-1}"
IMGSZ="${11:-640}"

# 降低 CUDA 内存碎片风险。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(
    /mnt/disk2/lhr/conda_envs/vsd/bin/python /mnt/disk2/lhr/VSD/scripts/e2_val_ir_only.py
    --model "$MODEL_PATH"
    --split "$SPLIT"
    --imgsz "$IMGSZ"
    --device "$DEVICE"
    --case-topk "$CASE_TOPK"
    --case-max-images "$CASE_MAX_IMAGES"
    --case-batch "$CASE_BATCH"
    --out-dir "$OUT_DIR"
)

if [[ -n "$ULTRA_PROJECT" ]]; then
    CMD+=(--ultra-project "$ULTRA_PROJECT")
fi

if [[ "$SKIP_CASE_VIZ" == "1" ]]; then
    CMD+=(--skip-case-viz)
fi

if [[ -n "$CASE_DEVICE" ]]; then
    CMD+=(--case-device "$CASE_DEVICE")
fi

"${CMD[@]}"
