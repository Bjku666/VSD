#!/usr/bin/env bash
# E6 mainline experiment demos for the next phase.
# Default behavior is dry-run: it prints the command that would be executed.
# Use RUN_MODE=run for implemented experiments only.
#
# Examples:
#   scripts/train/run_e6_mainline_demos.sh E10_2
#   RUN_MODE=run scripts/train/run_e6_mainline_demos.sh E10_2
#   scripts/train/run_e6_mainline_demos.sh E11_1
#   scripts/train/run_e6_mainline_demos.sh all

set -euo pipefail

ROOT="${ROOT:-/mnt/disk2/lhr/VSD}"
PY="${PY:-/mnt/disk2/lhr/conda_envs/vsd/bin/python}"
RUNNER="${ROOT}/scripts/dark_small_experiment_runner.py"
LOG_DIR="${ROOT}/results/val/logs"
RUN_MODE="${RUN_MODE:-dry-run}"
DEVICE="${DEVICE:-0,1}"
BATCH_768="${BATCH_768:-32}"
WORKERS="${WORKERS:-32}"
E6_640_WEIGHTS="${E6_640_WEIGHTS:-${ROOT}/results/val/e6_feature_fusion_multiscale/weights/best.pt}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

export VSD_E5_SCRIPTS_DIR="${ROOT}/scripts"
export VSD_E6_SCRIPTS_DIR="${ROOT}/scripts"
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_or_print() {
  if [[ "${RUN_MODE}" == "run" ]]; then
    "$@"
  else
    printf '%q ' "$@"
    printf '\n'
  fi
}

format_cmd() {
  printf '%q ' "$@"
}

run_background() {
  local exp_id="$1"
  shift
  local log_file="${LOG_DIR}/${exp_id}_$(date +%Y%m%d_%H%M%S).log"
  if [[ "${RUN_MODE}" == "run" ]]; then
    {
      echo "===== $(date '+%F %T') start ${exp_id} ====="
      echo "cwd=${ROOT}"
      echo "log_file=${log_file}"
      printf 'command='
      format_cmd "$@"
      printf '\n\n'
    } > "${log_file}"
    setsid env PYTHONUNBUFFERED=1 "$@" >> "${log_file}" 2>&1 < /dev/null &
    echo "${exp_id} PID=$!"
    echo "${exp_id} log=${log_file}"
  else
    echo "# log would be: ${log_file}"
    printf '%q ' "$@"
    printf '\n'
  fi
}

e10_2() {
  echo "# E10_2: E6 768, two-GPU train + validation through manifest"
  echo "# Implemented. This uses the manifest values: device=${DEVICE}, batch=${BATCH_768}, workers=${WORKERS}."
  run_background "e10_2_e6_768" \
    "${PY}" "${RUNNER}" run E10_2 --work-dir "${ROOT}"
}

e11_1() {
  echo "# E11_1: E6 + P2 detection head"
  echo "# Implemented. P2 is detection head only; no P2 cross-modal attention."
  run_or_print \
    "${PY}" "${ROOT}/scripts/e11_train_p2_head.py" \
      --mode rgb_ir \
      --model "${E6_640_WEIGHTS}" \
      --data-rgb-ir "${ROOT}/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml" \
      --data-ir "${ROOT}/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml" \
      --imgsz 640 \
      --batch 32 \
      --workers "${WORKERS}" \
      --device "${DEVICE}" \
      --project "${ROOT}/results/val" \
      --name e11_1_e6_p2_head \
      --validate-out "${ROOT}/results/val/e11_1_e6_p2_head_val" \
      --dry-run
}

e12_1() {
  echo "# E12_1: E6 + residual gated fusion"
  echo "# Implemented. Gate is lightweight residual fusion, not Transformer."
  run_or_print \
    "${PY}" "${ROOT}/scripts/e12_train_gated_fusion.py" \
      --mode rgb_ir \
      --model "${E6_640_WEIGHTS}" \
      --data-rgb-ir "${ROOT}/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml" \
      --data-ir "${ROOT}/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml" \
      --imgsz 640 \
      --batch 32 \
      --workers "${WORKERS}" \
      --device "${DEVICE}" \
      --project "${ROOT}/results/val" \
      --name e12_1_e6_residual_gated_fusion \
      --validate-out "${ROOT}/results/val/e12_1_e6_residual_gated_fusion_val" \
      --dry-run
}

e12_1b() {
  echo "# E12_1b: E6 + weak residual gated fusion"
  echo "# Implemented through the manifest with gate_lambda=0.1."
  run_background "e12_1b_weak_residual_gated_fusion" \
    "${PY}" "${RUNNER}" run E12_1b --work-dir "${ROOT}"
}

e13_2() {
  echo "# E13_2: E6 + scale-aware loss"
  echo "# Implemented. This uses the manifest values: device=${DEVICE}, batch=32, workers=${WORKERS}."
  run_background "e13_2_e6_scale_aware_loss" \
    "${PY}" "${RUNNER}" run E13_2 --work-dir "${ROOT}"
}

e22_3() {
  echo "# E22_3: hard negative mining 2x"
  echo "# Demo only. Use when E6/E11/E12 FP/image or FPPI_dark is clearly worse than E4 WBF."
  run_or_print \
    "${PY}" "${ROOT}/scripts/e22_hard_negative_mining.py" \
      --source-model "${E6_640_WEIGHTS}" \
      --subsets dark low-contrast \
      --multiplier 2 \
      --out-yaml "${ROOT}/generated_data/e22_3_hard_negative_2x.yaml" \
      --out-dir "${ROOT}/results/val/e22_3_hard_negative_2x" \
      --dry-run
}

usage() {
  cat <<'EOF'
Usage: scripts/train/run_e6_mainline_demos.sh {E10_2|E11_1|E12_1|E12_1b|E13_2|E22_3|all}

Environment:
  RUN_MODE=dry-run|run   default: dry-run
  DEVICE=0,1             default: 0,1
  BATCH_768=32           default: 32
  WORKERS=32             default: 32

E10_2, E11_1, E12_1, E12_1b, and E13_2 are implemented end to end. E22_3 remains a dry-run demo until hard-negative mining is implemented.
EOF
}

case "${1:-}" in
  E10_2|e10_2) e10_2 ;;
  E11_1|e11_1) e11_1 ;;
  E12_1|e12_1) e12_1 ;;
  E12_1b|e12_1b) e12_1b ;;
  E13_2|e13_2) e13_2 ;;
  E22_3|e22_3) e22_3 ;;
  all) e10_2; echo; e11_1; echo; e12_1; echo; e12_1b; echo; e13_2; echo; e22_3 ;;
  *) usage; exit 2 ;;
esac
