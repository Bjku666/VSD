#!/usr/bin/env bash
# Number-aligned demos for the DroneVehicle dark/small RGB-IR experiment plan.
#
# Default mode is dry-run. It prints commands without starting jobs.
# Set RUN_MODE=run to execute implemented experiments.
#
# Examples:
#   scripts/train/run_dark_small_experiment_demos.sh list
#   scripts/train/run_dark_small_experiment_demos.sh E10_2
#   RUN_MODE=run scripts/train/run_dark_small_experiment_demos.sh E10_2
#   scripts/train/run_dark_small_experiment_demos.sh all

set -euo pipefail

ROOT="${ROOT:-/mnt/disk2/lhr/VSD}"
PY="${PY:-/mnt/disk2/lhr/conda_envs/vsd/bin/python}"
YOLO="${YOLO:-/mnt/disk2/lhr/conda_envs/vsd/bin/yolo}"
RUNNER="${ROOT}/scripts/dark_small_experiment_runner.py"
RUN_MODE="${RUN_MODE:-dry-run}"
DEVICE="${DEVICE:-0,1}"
DEVICE_SINGLE="${DEVICE_SINGLE:-0}"
BATCH_SINGLE="${BATCH_SINGLE:-32}"
BATCH_FUSION="${BATCH_FUSION:-32}"
BATCH_LATE="${BATCH_LATE:-16}"
WORKERS="${WORKERS:-32}"
LOG_DIR="${ROOT}/results/val/logs"
MODEL_N="${MODEL_N:-${ROOT}/weights/pretrained/yolo11n.pt}"
MODEL_S="${MODEL_S:-${ROOT}/weights/pretrained/yolo11s.pt}"
RGB_DATA="${ROOT}/configs/dronevehicle_resplit/dronevehicle_resplit_rgb.yaml"
IR_DATA="${ROOT}/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml"
RGB_IR_DATA="${ROOT}/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml"
E1_W="${ROOT}/results/val/e1_yolo11n_rgb_only_640_ddp/weights/best.pt"
E2_W="${ROOT}/results/val/e2_yolo11n_ir_only_640_ddp/weights/best.pt"
E5_W="${ROOT}/results/val/yolo11n_e5_rgb_ir_640_ddp/weights/best.pt"
E6_W="${ROOT}/results/val/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt"
E10_W="${ROOT}/results/val/e10_2_e6_768/weights/best.pt"
FINAL_W="${FINAL_W:-${E10_W}}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

export VSD_E5_SCRIPTS_DIR="${ROOT}/scripts"
export VSD_E6_SCRIPTS_DIR="${ROOT}/scripts"
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

format_cmd() {
  printf '%q ' "$@"
}

run_or_print() {
  if [[ "${RUN_MODE}" == "run" ]]; then
    "$@"
  else
    print_cmd "$@"
  fi
}

run_logged() {
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
    print_cmd "$@"
  fi
}

planned_only() {
  local exp_id="$1"
  local reason="$2"
  shift 2
  echo "# ${exp_id}: demo only - ${reason}"
  print_cmd "$@"
}

runner_exp() {
  local exp_id="$1"
  echo "# ${exp_id}: manifest-backed experiment"
  run_logged "${exp_id}" "${PY}" "${RUNNER}" run "${exp_id}" --work-dir "${ROOT}"
}

# E0 protocol and dataset audit ------------------------------------------------
e0_1() { echo "# E0_1: RGB/IR pair audit"; run_or_print "${PY}" "${ROOT}/scripts/dataset_resplit_dronevehicle.py" --root "${ROOT}" --audit-pairs; }
e0_2() { echo "# E0_2: official split leakage check"; run_or_print "${PY}" "${ROOT}/scripts/dataset_resplit_dronevehicle.py" --root "${ROOT}" --check-leakage; }
e0_3() { echo "# E0_3: confirm dark/small/dark-small subsets"; run_or_print "${PY}" "${ROOT}/scripts/dataset_resplit_dronevehicle.py" --root "${ROOT}" --build-subsets dark small dark-small; }
e0_4() { echo "# E0_4: build low-contrast subset from train thresholds"; run_or_print "${PY}" "${ROOT}/scripts/dataset_resplit_dronevehicle.py" --root "${ROOT}" --build-subsets low-contrast; }
e0_5() { echo "# E0_5: build tiny/small/medium/large size buckets"; run_or_print "${PY}" "${ROOT}/scripts/dataset_resplit_dronevehicle.py" --root "${ROOT}" --build-size-buckets tiny small medium large; }
e0_6() { echo "# E0_6: aggregate unified metric protocol artifacts"; run_or_print "${PY}" "${ROOT}/scripts/dark_small_experiment_runner.py" aggregate --out-dir "${ROOT}/results/val"; }
e0_7() { echo "# E0_7: object-level evaluator smoke demo"; planned_only E0_7 "object-level evaluator is covered by E5/E6 val scripts; standalone CLI can be added later" "${PY}" "${ROOT}/scripts/e23_object_level_subset_eval.py" --model "${E6_W}" --subsets dark-small tiny low-contrast --dry-run; }

# E1-E6 baselines --------------------------------------------------------------
e1() { echo "# E1: RGB-only 640 baseline"; run_or_print "${ROOT}/scripts/train/e1_train_rgb_only.sh"; }
e2() { echo "# E2: IR-only 640 baseline"; run_or_print "${ROOT}/scripts/train/e2_train_ir_only.sh"; }
e3() { echo "# E3: late fusion NMS"; run_or_print "${ROOT}/scripts/train/e3_val_late_fusion_nms.sh"; }
e4() { echo "# E4: late fusion WBF equal weights"; run_or_print "${ROOT}/scripts/train/e4_val_late_fusion_wbf.sh"; }
e5() { echo "# E5: single-layer feature fusion 640"; run_or_print "${ROOT}/scripts/train/e5_train_feature_fusion_single.sh" rgb_ir "${DEVICE}" "${BATCH_FUSION}" yolo11n_e5_rgb_ir_640_ddp "${WORKERS}" bg; }
e6() { echo "# E6: multi-scale feature fusion 640"; run_or_print "${ROOT}/scripts/train/e6_train_feature_fusion_multiscale.sh" rgb_ir "${DEVICE}" "${BATCH_FUSION}" yolo11n_e6_rgb_ir_640_ddp "${WORKERS}" bg; }

# E7-E10 manifest-backed expansion -------------------------------------------
e7_1() { runner_exp E7_1; }
e7_2() { runner_exp E7_2; }
e7_3() { runner_exp E7_3; }
e7_4() { runner_exp E7_4; }
e7_5() { runner_exp E7_5; }
e7_6() { echo "# E7_6: skipped duplicate of E4 equal-weight WBF"; run_or_print "${PY}" "${RUNNER}" aggregate --out-dir "${ROOT}/results/val"; }
e8_1() { runner_exp E8_1; }
e8_2() { runner_exp E8_2; }
e8_3() { echo "# E8_3: paused by current route; demo retained"; runner_exp E8_3; }
e8_4() { echo "# E8_4: paused by current route; demo retained"; runner_exp E8_4; }
e9_1() { planned_only E9_1 "6-channel model support required" "${PY}" "${ROOT}/scripts/e9_train_early_fusion_6ch.py" --imgsz 640 --batch "${BATCH_SINGLE}" --workers "${WORKERS}" --device "${DEVICE}" --name e9_1_6ch_640 --dry-run; }
e9_2() { planned_only E9_2 "6-channel model support required" "${PY}" "${ROOT}/scripts/e9_train_early_fusion_6ch.py" --imgsz 768 --batch "${BATCH_SINGLE}" --workers "${WORKERS}" --device "${DEVICE}" --name e9_2_6ch_768 --dry-run; }
e10_1() { echo "# E10_1: stopped/deprioritized by current E6-first route"; runner_exp E10_1; }
e10_2() { runner_exp E10_2; }
e10_3() { echo "# E10_3: paused unless E10_2 beats E6 640"; runner_exp E10_3; }
e10_4() { echo "# E10_4: paused unless E10_2 beats E6 640"; runner_exp E10_4; }

# E11-E24 planned demos --------------------------------------------------------
e11_1() { planned_only E11_1 "E6 + P2 detection head; P2 is detection head only" "${PY}" "${ROOT}/scripts/e11_train_p2_head.py" --base fusion --model "${E6_W}" --imgsz 640 --batch "${BATCH_FUSION}" --workers "${WORKERS}" --device "${DEVICE}" --name e11_1_e6_p2_head --dry-run; }
e11_2() { planned_only E11_2 "E6 + P2 head + original P3/P4/P5 fusion" "${PY}" "${ROOT}/scripts/e11_train_p2_head.py" --base fusion --model "${E6_W}" --keep-p345-fusion --imgsz 640 --batch "${BATCH_FUSION}" --workers "${WORKERS}" --device "${DEVICE}" --name e11_2_e6_p2_p345_fusion --dry-run; }
e11_3() { planned_only E11_3 "E6 + P2 head + 768" "${PY}" "${ROOT}/scripts/e11_train_p2_head.py" --base fusion --model "${E10_W}" --imgsz 768 --batch "${BATCH_FUSION}" --workers "${WORKERS}" --device "${DEVICE}" --name e11_3_e6_p2_head_768 --dry-run; }
e12_1() { planned_only E12_1 "E6 + residual gated fusion" "${PY}" "${ROOT}/scripts/e12_train_gated_fusion.py" --gate residual --model "${E6_W}" --imgsz 640 --batch "${BATCH_FUSION}" --workers "${WORKERS}" --device "${DEVICE}" --name e12_1_residual_gated_fusion --dry-run; }
e12_1b() { echo "# E12_1b: E6 + weak residual gated fusion"; runner_exp E12_1b; }
e12_2() { planned_only E12_2 "E6 + spatial gate" "${PY}" "${ROOT}/scripts/e12_train_gated_fusion.py" --gate spatial --model "${E6_W}" --imgsz 640 --batch "${BATCH_FUSION}" --workers "${WORKERS}" --device "${DEVICE}" --name e12_2_spatial_gate --dry-run; }
e12_3() { planned_only E12_3 "E6 + dark-aware gate" "${PY}" "${ROOT}/scripts/e12_train_gated_fusion.py" --gate dark-aware --model "${E6_W}" --imgsz 640 --batch "${BATCH_FUSION}" --workers "${WORKERS}" --device "${DEVICE}" --name e12_3_dark_aware_gate --dry-run; }
e12_4() { planned_only E12_4 "E6 + dark-small resampling + dark-aware gate" "${PY}" "${ROOT}/scripts/e12_train_gated_fusion.py" --gate dark-aware --reweight-subset "${ROOT}/configs/dronevehicle_resplit/subsets/rgb_ir_dark-small.yaml" --reweight-multiplier 3 --model "${E6_W}" --imgsz 640 --batch "${BATCH_FUSION}" --workers "${WORKERS}" --device "${DEVICE}" --name e12_4_dark_aware_gate_darksmall_x3 --dry-run; }
e13_1() { planned_only E13_1 "loss baseline rerun for controlled ablation" "${PY}" "${ROOT}/scripts/e13_train_tiny_aware_loss.py" --loss baseline --model "${E6_W}" --name e13_1_e6_loss_baseline --dry-run; }
e13_2() { planned_only E13_2 "small/tiny scale-aware weight" "${PY}" "${ROOT}/scripts/e13_train_tiny_aware_loss.py" --loss scale-aware --model "${E6_W}" --name e13_2_scale_aware --dry-run; }
e13_3() { planned_only E13_3 "dark sample weighting" "${PY}" "${ROOT}/scripts/e13_train_tiny_aware_loss.py" --loss dark-weight --model "${E6_W}" --name e13_3_dark_weight --dry-run; }
e13_4() { planned_only E13_4 "dark-small sample weighting" "${PY}" "${ROOT}/scripts/e13_train_tiny_aware_loss.py" --loss dark-small-weight --model "${E6_W}" --name e13_4_darksmall_weight --dry-run; }
e13_5() { planned_only E13_5 "center/location-sensitive bbox loss" "${PY}" "${ROOT}/scripts/e13_train_tiny_aware_loss.py" --loss center-aware --model "${E6_W}" --name e13_5_center_aware --dry-run; }
e13_6() { planned_only E13_6 "WIoU/CIoU comparison" "${PY}" "${ROOT}/scripts/e13_train_tiny_aware_loss.py" --loss wiou-ciou --model "${E6_W}" --name e13_6_wiou_ciou --dry-run; }
e14_1() { planned_only E14_1 "RGB gamma/CLAHE enhancement" "${PY}" "${ROOT}/scripts/e14_train_low_contrast_enhance.py" --enhance rgb-gamma-clahe --name e14_1_rgb_gamma_clahe --dry-run; }
e14_2() { planned_only E14_2 "IR normalization and weak noise" "${PY}" "${ROOT}/scripts/e14_train_low_contrast_enhance.py" --enhance ir-normalize-weak-noise --name e14_2_ir_norm_noise --dry-run; }
e14_3() { planned_only E14_3 "IR background suppression attention" "${PY}" "${ROOT}/scripts/e14_train_low_contrast_enhance.py" --enhance ir-background-suppression --name e14_3_ir_bgs_attention --dry-run; }
e14_4() { planned_only E14_4 "low-contrast subset evaluation" "${PY}" "${ROOT}/scripts/e6_val_feature_fusion_multiscale.py" --weights "${FINAL_W}" --mode rgb_ir --split val --imgsz 640 --batch "${BATCH_FUSION}" --device "${DEVICE_SINGLE}" --out-dir "${ROOT}/results/val/e14_4_low_contrast_eval"; }
e15_1() { planned_only E15_1 "YOLOv10n strong model comparison" "${YOLO}" detect train model=yolov10n.pt data="${RGB_IR_DATA}" imgsz=640 epochs=100 batch="${BATCH_SINGLE}" workers="${WORKERS}" device="${DEVICE}" project="${ROOT}/results/val" name=e15_1_yolov10n; }
e15_2() { planned_only E15_2 "RT-DETR-R18 comparison" "${YOLO}" detect train model=rtdetr-r18.pt data="${RGB_IR_DATA}" imgsz=640 epochs=100 batch=16 workers="${WORKERS}" device="${DEVICE}" project="${ROOT}/results/val" name=e15_2_rtdetr_r18; }
e15_3() { planned_only E15_3 "YOLO11s modality/model-capacity comparison" "${PY}" "${ROOT}/scripts/e15_train_yolo11s_comparison.py" --modes rgb ir fusion --model "${MODEL_S}" --device "${DEVICE}" --dry-run; }
e15_4() { planned_only E15_4 "final method vs strong models" "${PY}" "${ROOT}/scripts/e15_compare_final_models.py" --final "${FINAL_W}" --baselines e15_1 e15_2 e15_3 --dry-run; }
e16_1() { planned_only E16_1 "normal registration robustness baseline" "${PY}" "${ROOT}/scripts/e16_eval_registration_shift.py" --shift 0 --model "${FINAL_W}" --dry-run; }
e16_2() { planned_only E16_2 "IR random shift +/-2 px" "${PY}" "${ROOT}/scripts/e16_eval_registration_shift.py" --shift 2 --model "${FINAL_W}" --dry-run; }
e16_3() { planned_only E16_3 "IR random shift +/-4 px" "${PY}" "${ROOT}/scripts/e16_eval_registration_shift.py" --shift 4 --model "${FINAL_W}" --dry-run; }
e16_4() { planned_only E16_4 "IR random shift +/-8 px" "${PY}" "${ROOT}/scripts/e16_eval_registration_shift.py" --shift 8 --model "${FINAL_W}" --dry-run; }
e16_5() { planned_only E16_5 "gated fusion under registration shift" "${PY}" "${ROOT}/scripts/e16_eval_registration_shift.py" --shift 4 --model "${ROOT}/results/val/e12_best/weights/best.pt" --gated --dry-run; }
e17_1() { e0_4; }
e17_2() { planned_only E17_2 "low-contrast evaluation across selected models" "${PY}" "${ROOT}/scripts/e17_eval_low_contrast_models.py" --models E2="${E2_W}" E4="${ROOT}/results/val/e4_late_fusion_wbf_val" E10="${E10_W}" FINAL="${FINAL_W}" --dry-run; }
e17_3() { planned_only E17_3 "report low-contrast AP/Recall/FPPI" "${PY}" "${ROOT}/scripts/e17_report_low_contrast.py" --results-val "${ROOT}/results/val" --dry-run; }
e18() { planned_only E18 "multi-seed stability wrapper" "${PY}" "${ROOT}/scripts/e18_run_multiseed.py" --experiments E5 E6 E10_2 --seeds 1 2 --device "${DEVICE}" --dry-run; }
e19() { planned_only E19 "efficiency/deployment metrics" "${PY}" "${ROOT}/scripts/e19_measure_efficiency.py" --models E2="${E2_W}" E6="${E6_W}" E10="${E10_W}" FINAL="${FINAL_W}" --dry-run; }
e20() { planned_only E20 "per-class/tiny/failure-case analysis" "${PY}" "${ROOT}/scripts/e20_failure_case_analysis.py" --model "${FINAL_W}" --out-dir "${ROOT}/results/val/e20_failure_analysis" --dry-run; }
e21() { planned_only E21 "locked test-set final evaluation; do not run during tuning" "${PY}" "${ROOT}/scripts/e21_locked_test_eval.py" --protocol DroneVehicle-DarkSmall-v1 --models E2 E4 E10 FINAL --dry-run; }
e22_1() { planned_only E22_1 "collect FP from E2/E4/E10-best on dark/low-contrast" "${PY}" "${ROOT}/scripts/e22_hard_negative_mining.py" --step collect-fp --models E2="${E2_W}" E10="${E10_W}" --subsets dark low-contrast --dry-run; }
e22_2() { planned_only E22_2 "classify FP taxonomy" "${PY}" "${ROOT}/scripts/e22_hard_negative_mining.py" --step classify-fp --taxonomy thermal_hotspot lamp edge background_vehicle_like registration_error --dry-run; }
e22_3() { planned_only E22_3 "hard negative 2x sampling" "${PY}" "${ROOT}/scripts/e22_hard_negative_mining.py" --step train --multiplier 2 --out-yaml "${ROOT}/generated_data/e22_3_hard_negative_2x.yaml" --dry-run; }
e22_4() { planned_only E22_4 "hard negative 3x sampling" "${PY}" "${ROOT}/scripts/e22_hard_negative_mining.py" --step train --multiplier 3 --out-yaml "${ROOT}/generated_data/e22_4_hard_negative_3x.yaml" --dry-run; }
e22_5() { planned_only E22_5 "hard negative + CEBS" "${PY}" "${ROOT}/scripts/e22_hard_negative_mining.py" --step train --multiplier 2 --cebs --out-yaml "${ROOT}/generated_data/e22_5_hard_negative_cebs.yaml" --dry-run; }
e23_1() { planned_only E23_1 "object-level evaluator implementation smoke" "${PY}" "${ROOT}/scripts/e23_object_level_subset_eval.py" --subsets dark-small tiny low-contrast --model "${FINAL_W}" --dry-run; }
e23_2() { planned_only E23_2 "export object-level metrics" "${PY}" "${ROOT}/scripts/e23_object_level_subset_eval.py" --subsets dark-small tiny low-contrast --metrics AP Recall --out "${ROOT}/results/val/e23_object_level_metrics.json" --dry-run; }
e23_3() { planned_only E23_3 "compare image-level and object-level subset metrics" "${PY}" "${ROOT}/scripts/e23_compare_metric_scopes.py" --results-val "${ROOT}/results/val" --dry-run; }
e24_1() { planned_only E24_1 "freeze final configs" "${PY}" "${ROOT}/scripts/e24_freeze_repro_config.py" --step freeze-configs --out-dir "${ROOT}/configs/frozen" --dry-run; }
e24_2() { planned_only E24_2 "audit config paths, commit, seed, weights, results" "${PY}" "${ROOT}/scripts/e24_freeze_repro_config.py" --step audit --results-val "${ROOT}/results/val" --dry-run; }
e24_3() { planned_only E24_3 "verify leaderboard vs result dirs" "${PY}" "${ROOT}/scripts/e24_freeze_repro_config.py" --step verify-leaderboard --leaderboard "${ROOT}/results/val/dark_small_experiment_leaderboard.csv" --dry-run; }

list_ids() {
  cat <<'EOF'
E0_1 E0_2 E0_3 E0_4 E0_5 E0_6 E0_7
E1 E2 E3 E4 E5 E6
E7_1 E7_2 E7_3 E7_4 E7_5 E7_6
E8_1 E8_2 E8_3 E8_4
E9_1 E9_2
E10_1 E10_2 E10_3 E10_4
E11_1 E11_2 E11_3
E12_1 E12_1b E12_2 E12_3 E12_4
E13_1 E13_2 E13_3 E13_4 E13_5 E13_6
E14_1 E14_2 E14_3 E14_4
E15_1 E15_2 E15_3 E15_4
E16_1 E16_2 E16_3 E16_4 E16_5
E17_1 E17_2 E17_3
E18 E19 E20 E21
E22_1 E22_2 E22_3 E22_4 E22_5
E23_1 E23_2 E23_3
E24_1 E24_2 E24_3
EOF
}

usage() {
  cat <<'EOF'
Usage: scripts/train/run_dark_small_experiment_demos.sh {list|all|EXPERIMENT_ID...}

Default: dry-run. Set RUN_MODE=run for implemented experiments.
Current route: do not run E21/test-set or strong-model E15 during tuning.
EOF
}

run_id() {
  local id_lc
  id_lc="$(echo "$1" | tr '[:upper:]-' '[:lower:]_')"
  case "${id_lc}" in
    e0_1) e0_1 ;; e0_2) e0_2 ;; e0_3) e0_3 ;; e0_4) e0_4 ;; e0_5) e0_5 ;; e0_6) e0_6 ;; e0_7) e0_7 ;;
    e1) e1 ;; e2) e2 ;; e3) e3 ;; e4) e4 ;; e5) e5 ;; e6) e6 ;;
    e7_1) e7_1 ;; e7_2) e7_2 ;; e7_3) e7_3 ;; e7_4) e7_4 ;; e7_5) e7_5 ;; e7_6) e7_6 ;;
    e8_1) e8_1 ;; e8_2) e8_2 ;; e8_3) e8_3 ;; e8_4) e8_4 ;;
    e9_1) e9_1 ;; e9_2) e9_2 ;;
    e10_1) e10_1 ;; e10_2) e10_2 ;; e10_3) e10_3 ;; e10_4) e10_4 ;;
    e11_1) e11_1 ;; e11_2) e11_2 ;; e11_3) e11_3 ;;
    e12_1) e12_1 ;; e12_1b) e12_1b ;; e12_2) e12_2 ;; e12_3) e12_3 ;; e12_4) e12_4 ;;
    e13_1) e13_1 ;; e13_2) e13_2 ;; e13_3) e13_3 ;; e13_4) e13_4 ;; e13_5) e13_5 ;; e13_6) e13_6 ;;
    e14_1) e14_1 ;; e14_2) e14_2 ;; e14_3) e14_3 ;; e14_4) e14_4 ;;
    e15_1) e15_1 ;; e15_2) e15_2 ;; e15_3) e15_3 ;; e15_4) e15_4 ;;
    e16_1) e16_1 ;; e16_2) e16_2 ;; e16_3) e16_3 ;; e16_4) e16_4 ;; e16_5) e16_5 ;;
    e17_1) e17_1 ;; e17_2) e17_2 ;; e17_3) e17_3 ;;
    e18) e18 ;; e19) e19 ;; e20) e20 ;; e21) e21 ;;
    e22_1) e22_1 ;; e22_2) e22_2 ;; e22_3) e22_3 ;; e22_4) e22_4 ;; e22_5) e22_5 ;;
    e23_1) e23_1 ;; e23_2) e23_2 ;; e23_3) e23_3 ;;
    e24_1) e24_1 ;; e24_2) e24_2 ;; e24_3) e24_3 ;;
    *) echo "Unknown experiment ID: $1" >&2; usage >&2; exit 2 ;;
  esac
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

case "$1" in
  list) list_ids ;;
  all)
    while read -r line; do
      for id in ${line}; do
        echo
        run_id "${id}"
      done
    done < <(list_ids)
    ;;
  *)
    for id in "$@"; do
      run_id "${id}"
    done
    ;;
esac
