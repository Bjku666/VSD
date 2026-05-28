#!/usr/bin/env bash
set -uo pipefail

cd /mnt/disk2/lhr/VSD || exit 1

LOG="/mnt/disk2/lhr/VSD/results/S6_object_background_suppression/logs/e14_2_e6_cebs_a010_object_eval_gpu0_20260525_1231.log"
PYTHON="/mnt/disk2/lhr/conda_envs/vsd/bin/python"
export LD_LIBRARY_PATH="/mnt/disk2/lhr/conda_envs/vsd/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "START_E14_2_OBJECT_WAIT $(date -Is)"
while true; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')"
  echo "WAIT_GPU0 $(date -Is) used_mb=${used} util_pct=${util}"
  if [ "${used}" -lt 2000 ]; then
    break
  fi
  sleep 60
done

"$PYTHON" scripts/e23_object_level_subset_eval.py \
  --weights /mnt/disk2/lhr/VSD/results/S6_object_background_suppression/e14_2_e6_cebs_a010/weights/best.pt \
  --validator e14 \
  --mode rgb_ir \
  --split val \
  --image-metrics /mnt/disk2/lhr/VSD/results/S6_object_background_suppression/e14_2_e6_cebs_a010_val/required_metrics.json \
  --imgsz 640 \
  --batch 16 \
  --workers 4 \
  --device 0 \
  --out-dir /mnt/disk2/lhr/VSD/results/S6_object_background_suppression/e23_e14_2_cebs_a010_object_level \
  --cebs-alpha 0.10 \
  --dark-threshold 33.50320816040039 \
  --low-contrast-threshold 0.08425217866897583 \
  --contrast-kernel 7 \
  --suppression-temperature 0.08
code=$?
echo "OBJECT_EVAL_EXIT ${code} $(date -Is)"
exit "$code"
