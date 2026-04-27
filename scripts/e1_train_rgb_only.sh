#!/bin/bash
# 在 DroneVehicle resplit 数据集上训练 YOLO11n RGB-only baseline。
# 用法：
#   ./scripts/e1_train_yolo11n_rgb_only.sh [device] [batch] [run_name] [workers] [mode]
# 示例：
#   ./scripts/e1_train_yolo11n_rgb_only.sh 0,1 96 e1_yolo11n_rgb_only_640_ddp 16 bg
#   ./scripts/e1_train_yolo11n_rgb_only.sh 0 48 e1_yolo11n_rgb_only_640_gpu0 12 fg
set -euo pipefail

DEVICE="${1:-0,1}"
BATCH="${2:-96}"
RUN_NAME="${3:-e1_yolo11n_rgb_only_640_ddp}"
WORKERS="${4:-16}"
MODE="${5:-bg}"

PROJECT_DIR="/mnt/disk2/lhr/VSD/experiments/e1_rgb_only"
LOG_DIR="/mnt/disk2/lhr/VSD/logs/e1_rgb_only"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_gpu${DEVICE//,/+}_${TS}.log"

mkdir -p "${LOG_DIR}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/latest_train.log"

if [[ "${MODE}" == "bg" ]]; then
    nohup "$0" "${DEVICE}" "${BATCH}" "${RUN_NAME}" "${WORKERS}" "fg" >/dev/null 2>&1 &
    PID=$!
    echo "已后台启动训练，PID=${PID}"
    echo "日志：${LOG_DIR}/latest_train.log"
    exit 0
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/disk2/lhr/conda_envs/vsd

echo "[train] device=${DEVICE} batch=${BATCH} workers=${WORKERS} run_name=${RUN_NAME}" | tee -a "${LOG_FILE}"
echo "[train] log_file=${LOG_FILE}" | tee -a "${LOG_FILE}"

yolo detect train \
    model=/mnt/disk2/lhr/VSD/weights/pretrained/yolo11n.pt \
    data=/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb.yaml \
    imgsz=640 \
    batch="${BATCH}" \
    workers="${WORKERS}" \
    device="${DEVICE}" \
    project="${PROJECT_DIR}" \
    name="${RUN_NAME}" 2>&1 | tee -a "${LOG_FILE}"
