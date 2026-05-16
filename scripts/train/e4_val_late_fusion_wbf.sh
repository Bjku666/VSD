#!/bin/bash
# 评估 E4：RGB+IR Late Fusion (WBF)
# 用法:
#   ./scripts/e4_val_late_fusion_wbf.sh [split] [rgb_model] [ir_model] [out_dir] [device] [batch]
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/disk2/lhr/conda_envs/vsd

SPLIT="${1:-val}"
RGB_MODEL="${2:-/mnt/disk2/lhr/VSD/results/val/e1_yolo11n_rgb_only_640_ddp/weights/best.pt}"
IR_MODEL="${3:-/mnt/disk2/lhr/VSD/results/val/e2_yolo11n_ir_only_640_ddp/weights/best.pt}"
OUT_DIR="${4:-/mnt/disk2/lhr/VSD/results/${SPLIT}/e4_late_fusion_wbf_${SPLIT}}"
DEVICE="${5:-0}"
BATCH="${6:-16}"

mkdir -p "${OUT_DIR}"

/mnt/disk2/lhr/conda_envs/vsd/bin/python /mnt/disk2/lhr/VSD/scripts/e4_val_late_fusion_wbf.py \
  --method late_wbf \
  --split "${SPLIT}" \
  --rgb-model "${RGB_MODEL}" \
  --ir-model "${IR_MODEL}" \
  --device "${DEVICE}" \
  --batch "${BATCH}" \
  --out-dir "${OUT_DIR}"
