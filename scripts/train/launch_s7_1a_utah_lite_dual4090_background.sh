#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/disk2/lhr/VSD}"
PY="${PY:-/mnt/disk2/lhr/conda_envs/vsd/bin/python}"
DEVICE="${DEVICE:-0,1}"
BATCH="${BATCH:-96}"
WORKERS="${WORKERS:-16}"
EPOCHS="${EPOCHS:-100}"
NAME="${NAME:-s7_1a_utah_lite_a06_b04}"
PROJECT="${PROJECT:-$ROOT/results/S7_architecture_incubation}"
LOG_DIR="${LOG_DIR:-$PROJECT/logs}"
PID_FILE="${PID_FILE:-$LOG_DIR/${NAME}.pid}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

export VSD_E6_SCRIPTS_DIR="$ROOT/scripts"
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/mnt/disk2/lhr/conda_envs/vsd/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-vsd}"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "S7_1a already appears to be running: PID=$old_pid"
    echo "PID file: $PID_FILE"
    exit 0
  fi
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${NAME}_${timestamp}.log"

{
  echo "===== $(date '+%F %T') start S7_1a UTAH-lite train ====="
  echo "cwd=$ROOT"
  echo "device=$DEVICE batch=$BATCH workers=$WORKERS epochs=$EPOCHS"
  echo "project=$PROJECT name=$NAME"
  echo "pid_file=$PID_FILE"
  printf 'command='
  printf '%q ' "$PY" scripts/s7_1_train_utah_lite.py \
    --mode rgb_ir \
    --model "$ROOT/results/S2_fusion_mainline/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt" \
    --epochs "$EPOCHS" \
    --imgsz 640 \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --project "$PROJECT" \
    --name "$NAME" \
    --seed 0 \
    --close-mosaic 10 \
    --patience 100 \
    --quality-alpha 0.6 \
    --quality-beta 0.4 \
    --quality-loss-weight 0.2 \
    --exist-ok
  printf '\n\n'
} > "$LOG_FILE"

(
  trap '' HUP
  env PYTHONUNBUFFERED=1 "$PY" scripts/s7_1_train_utah_lite.py \
    --mode rgb_ir \
    --model "$ROOT/results/S2_fusion_mainline/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt" \
    --epochs "$EPOCHS" \
    --imgsz 640 \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --project "$PROJECT" \
    --name "$NAME" \
    --seed 0 \
    --close-mosaic 10 \
    --patience 100 \
    --quality-alpha 0.6 \
    --quality-beta 0.4 \
    --quality-loss-weight 0.2 \
    --exist-ok >> "$LOG_FILE" 2>&1 < /dev/null
  status="$?"
  echo "===== $(date '+%F %T') S7_1a train exit status ${status} =====" >> "$LOG_FILE"
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$BASHPID" ]]; then
    rm -f "$PID_FILE"
  fi
  exit "$status"
) &

pid="$!"
echo "$pid" > "$PID_FILE"
disown "$pid" 2>/dev/null || true

echo "S7_1a PID=$pid"
echo "S7_1a log=$LOG_FILE"
echo "S7_1a pid_file=$PID_FILE"
echo "Monitor: tail -f '$LOG_FILE'"
