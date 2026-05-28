#!/usr/bin/env bash
# 历史 E7/E8/E10 队列脚本。
# 当前 E6 multi-scale fusion 已成为主线；本脚本保留已完成实验记录，
# 但不再自动扩展 WBF 权重搜索、IR-only 960/resampling 或 E5 768。
# GPU1 当前已被占用，本脚本只绑定物理 GPU0。

set -uo pipefail

cd /mnt/disk2/lhr/VSD || exit 1

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

PY=/mnt/disk2/lhr/conda_envs/vsd/bin/python
RUNNER=scripts/dark_small_experiment_runner.py
LOG_DIR=/mnt/disk2/lhr/VSD/results/S3_resolution_sampling/logs
STATUS_FILE="${LOG_DIR}/e7_e8_e10_status.tsv"

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

  if "${PY}" -u "${RUNNER}" run "${exp_id}" --work-dir /mnt/disk2/lhr/VSD; then
    record_status "${exp_id}" "done"
    "${PY}" -u "${RUNNER}" aggregate || true
  else
    record_status "${exp_id}" "failed"
    echo "===== ${exp_id} 失败，继续执行后续可独立实验 ====="
  fi
}

echo "GPU 绑定：CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "状态文件：${STATUS_FILE}"

# 当前只保留 E10_2：E6 768。
# E10_1 已停止并降级；E10_3/E10_4 等 E10_2 明确优于 E6 640 后再考虑。
record_status E10_1 "stopped_deprioritized"
run_exp E10_2
record_status E10_3 "paused_until_E10_2_beats_E6_640"
record_status E10_4 "paused_until_E10_2_beats_E6_640"

"${PY}" -u "${RUNNER}" aggregate || true
"${PY}" -u scripts/render_key_result_figures.py --results-val /mnt/disk2/lhr/VSD/results/S3_resolution_sampling || true

echo
echo "===== $(date '+%F %T') E7/E8/E10 队列结束 ====="
