#!/usr/bin/env bash
set -euo pipefail

PARENT_PID="${1:-498029}"
LOG="${2:-/mnt/disk2/lhr/VSD/results/val/logs/s65_stop_duplicate_gpu0_seed43.log}"

echo "[watch] parent=${PARENT_PID} start $(date --iso-8601=seconds)" >> "$LOG"

while kill -0 "$PARENT_PID" 2>/dev/null; do
  mapfile -t pids < <(pgrep -f 'e25_0_e13_3b_seed43 .*--device 0' || true)
  if ((${#pids[@]} > 0)); then
    echo "[watch] duplicate GPU0 seed43 detected: ${pids[*]} $(date --iso-8601=seconds)" >> "$LOG"
    kill "${pids[@]}" 2>/dev/null || true
    sleep 2
    mapfile -t remaining < <(pgrep -f 'e25_0_e13_3b_seed43 .*--device 0' || true)
    if ((${#remaining[@]} > 0)); then
      echo "[watch] forcing duplicate GPU0 seed43 stop: ${remaining[*]} $(date --iso-8601=seconds)" >> "$LOG"
      kill -9 "${remaining[@]}" 2>/dev/null || true
    fi
    echo "[watch] done $(date --iso-8601=seconds)" >> "$LOG"
    exit 0
  fi
  sleep 10
done

echo "[watch] parent exited before duplicate seed43 appeared $(date --iso-8601=seconds)" >> "$LOG"
