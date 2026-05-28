#!/usr/bin/env bash
set -uo pipefail

cd /mnt/disk2/lhr/VSD || exit 1

LOG="/mnt/disk2/lhr/VSD/results/S6_object_background_suppression/logs/e14_1_e6_cebs_a005_manual_val_gpu1_20260525_1231.log"
PYTHON="/mnt/disk2/lhr/conda_envs/vsd/bin/python"
export LD_LIBRARY_PATH="/mnt/disk2/lhr/conda_envs/vsd/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "START_E14_1_VALIDATE_WAIT $(date -Is)"
while true; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 | tr -d ' ')"
  echo "WAIT_GPU1 $(date -Is) used_mb=${used} util_pct=${util}"
  if [ "${used}" -lt 2000 ]; then
    break
  fi
  sleep 60
done

"$PYTHON" scripts/e14_val_cebs.py \
  --weights /mnt/disk2/lhr/VSD/results/S6_object_background_suppression/e14_1_e6_cebs_a005/weights/best.pt \
  --mode rgb_ir \
  --split val \
  --imgsz 640 \
  --batch 16 \
  --workers 4 \
  --device 1 \
  --out-dir /mnt/disk2/lhr/VSD/results/S6_object_background_suppression/e14_1_e6_cebs_a005_val \
  --cebs-alpha 0.05 \
  --dark-threshold 33.50320816040039 \
  --low-contrast-threshold 0.08425217866897583 \
  --contrast-kernel 7 \
  --suppression-temperature 0.08 \
  --exist-ok
code=$?
echo "VALIDATE_EXIT ${code} $(date -Is)"
exit "$code"
