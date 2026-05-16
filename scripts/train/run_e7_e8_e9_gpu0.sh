#!/usr/bin/env bash
# 按 shiyan.md 的 E7/E8/E9 顺序执行批量实验。
# 注意：重新开始训练后，E7 的 WBF 实验依赖 E1/E2 新训练出的 best.pt。
# GPU1 当前已被占用，本脚本只绑定物理 GPU0。

set -uo pipefail

cd /mnt/disk2/lhr/VSD || exit 1

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=/mnt/disk2/lhr/conda_envs/vsd/bin/python
RUNNER=scripts/dark_small_experiment_runner.py
LOG_DIR=/mnt/disk2/lhr/VSD/results/val/logs
STATUS_FILE="${LOG_DIR}/e7_e8_e9_status.tsv"

mkdir -p "${LOG_DIR}"
echo -e "time\texperiment\tstatus" > "${STATUS_FILE}"

record_status() {
  local exp_id="$1"
  local status="$2"
  echo -e "$(date '+%F %T')\t${exp_id}\t${status}" >> "${STATUS_FILE}"
  echo "$(date '+%F %T') ${exp_id} ${status}"
}

run_exp() {
  local exp_id="$1"

  echo
  echo "===== $(date '+%F %T') 开始 ${exp_id} ====="
  record_status "${exp_id}" "running"

  if "${PY}" "${RUNNER}" run "${exp_id}"; then
    record_status "${exp_id}" "done"
    "${PY}" "${RUNNER}" aggregate || true
  else
    record_status "${exp_id}" "failed"
    echo "===== ${exp_id} 失败，继续执行后续可独立实验 ====="
  fi
}

echo "GPU 绑定：CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "状态文件：${STATUS_FILE}"

# E7：强基线扩展。
run_exp E7_1
run_exp E7_2
run_exp E7_3

# E8：Mosaic 与 dark-small 难样本采样。
run_exp E8_1
run_exp E8_2
run_exp E8_3

if [[ -f /mnt/disk2/lhr/VSD/results/val/e8_2_ir_only_768_darksmall_x3/weights/best.pt ]]; then
  run_exp E8_4
else
  record_status E8_4 "skipped_missing_E8_2_best"
fi

# E9：特征级融合修正版重跑。
run_exp E9_1
run_exp E9_2
run_exp E9_3
run_exp E9_4

"${PY}" "${RUNNER}" aggregate || true
"${PY}" scripts/render_key_result_figures.py --results-val /mnt/disk2/lhr/VSD/results/val || true

echo
echo "===== $(date '+%F %T') E7/E8/E9 队列结束 ====="
