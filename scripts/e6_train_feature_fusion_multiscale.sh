#!/bin/bash
# 训练 YOLO11n E6（双分支多尺度融合）
# 用法:
#   ./scripts/e6_train_feature_fusion_multiscale.sh [mode] [device] [batch] [run_name] [workers] [mode_fg_bg]
# 示例:
#   ./scripts/e6_train_feature_fusion_multiscale.sh rgb_ir 0,1 64 yolo11n_e6_rgb_ir_640_ddp 16 bg
#   ./scripts/e6_train_feature_fusion_multiscale.sh rgb 0 48 yolo11n_e6_rgb_640_gpu0 12 fg
set -euo pipefail

FUSION_MODE="${1:-rgb_ir}"
DEVICE="${2:-0,1}"
BATCH="${3:-64}"
RUN_NAME="${4:-yolo11n_e6_rgb_ir_640_ddp}"
WORKERS="${5:-16}"
MODE="${6:-bg}"

PROJECT_DIR="/mnt/disk2/lhr/VSD/experiments/e6_feature_fusion_multiscale"
LOG_DIR="/mnt/disk2/lhr/VSD/logs/e6_feature_fusion_multiscale"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${FUSION_MODE}_gpu${DEVICE//,/+}_${TS}.log"

mkdir -p "${LOG_DIR}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/latest_train.log"

if [[ "${MODE}" == "bg" ]]; then
    nohup "$0" "${FUSION_MODE}" "${DEVICE}" "${BATCH}" "${RUN_NAME}" "${WORKERS}" "fg" >/dev/null 2>&1 &
    PID=$!
    echo "已后台启动训练，PID=${PID}"
    echo "日志：${LOG_DIR}/latest_train.log"
    exit 0
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/disk2/lhr/conda_envs/vsd
export VSD_E6_SCRIPTS_DIR="/mnt/disk2/lhr/VSD/scripts"
export PYTHONPATH="${VSD_E6_SCRIPTS_DIR}:${PYTHONPATH:-}"

echo "[train-e6] fusion_mode=${FUSION_MODE} device=${DEVICE} batch=${BATCH} workers=${WORKERS} run_name=${RUN_NAME}" | tee -a "${LOG_FILE}"
echo "[train-e6] log_file=${LOG_FILE}" | tee -a "${LOG_FILE}"

python /mnt/disk2/lhr/VSD/scripts/e6_train_feature_fusion_multiscale.py \
    --mode "${FUSION_MODE}" \
    --device "${DEVICE}" \
    --batch "${BATCH}" \
    --workers "${WORKERS}" \
    --name "${RUN_NAME}" \
    --project "${PROJECT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
