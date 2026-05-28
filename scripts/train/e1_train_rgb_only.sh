#!/bin/bash
# 在 DroneVehicle resplit 数据集上训练 YOLO11n RGB-only baseline。
# 用法：
#   ./scripts/train/e1_train_rgb_only.sh [device] [batch] [run_name] [workers] [mode] [imgsz] [close_mosaic]
# 示例：
#   ./scripts/train/e1_train_rgb_only.sh 0,1 96 e1_yolo11n_rgb_only_640_ddp 16 bg
#   ./scripts/train/e1_train_rgb_only.sh 0 48 e1_yolo11n_rgb_only_640_gpu0 12 fg
set -euo pipefail

DEVICE="${1:-0,1}"
BATCH="${2:-96}"
RUN_NAME="${3:-e1_yolo11n_rgb_only_640_ddp}"
WORKERS="${4:-16}"
MODE="${5:-bg}"
IMGSZ="${6:-640}"
CLOSE_MOSAIC="${7:-10}"

PROJECT_DIR="/mnt/disk2/lhr/VSD/results/S1_baselines"
LOG_DIR="/mnt/disk2/lhr/VSD/results/S1_baselines/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_gpu${DEVICE//,/+}_${TS}.log"

mkdir -p "${LOG_DIR}"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/latest_train.log"

if [[ "${MODE}" == "bg" ]]; then
    nohup "$0" "${DEVICE}" "${BATCH}" "${RUN_NAME}" "${WORKERS}" "fg" "${IMGSZ}" "${CLOSE_MOSAIC}" >/dev/null 2>&1 &
    PID=$!
    echo "已后台启动训练，PID=${PID}"
    echo "日志：${LOG_DIR}/latest_train.log"
    exit 0
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/disk2/lhr/conda_envs/vsd

echo "[train] device=${DEVICE} batch=${BATCH} workers=${WORKERS} imgsz=${IMGSZ} close_mosaic=${CLOSE_MOSAIC} run_name=${RUN_NAME}" | tee -a "${LOG_FILE}"
echo "[train] log_file=${LOG_FILE}" | tee -a "${LOG_FILE}"

yolo detect train \
    model=/mnt/disk2/lhr/VSD/weights/pretrained/yolo11n.pt \
    data=/mnt/disk2/lhr/VSD/configs/dronevehicle_resplit/dronevehicle_resplit_rgb.yaml \
    imgsz="${IMGSZ}" \
    batch="${BATCH}" \
    workers="${WORKERS}" \
    device="${DEVICE}" \
    project="${PROJECT_DIR}" \
    name="${RUN_NAME}" \
    close_mosaic="${CLOSE_MOSAIC}" \
    exist_ok=True 2>&1 | tee -a "${LOG_FILE}"
