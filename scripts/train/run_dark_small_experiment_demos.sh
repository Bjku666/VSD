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
LOG_DIR="${ROOT}/results/S6_5_reliability_calibration/logs"
WORK_DIR="${WORK_DIR:-${ROOT}/results/S6_5_reliability_calibration/work}"
MODEL_N="${MODEL_N:-${ROOT}/weights/pretrained/yolo11n.pt}"
MODEL_S="${MODEL_S:-${ROOT}/weights/pretrained/yolo11s.pt}"
RGB_DATA="${ROOT}/configs/dronevehicle_resplit/dronevehicle_resplit_rgb.yaml"
IR_DATA="${ROOT}/configs/dronevehicle_resplit/dronevehicle_resplit_ir.yaml"
RGB_IR_DATA="${ROOT}/configs/dronevehicle_resplit/dronevehicle_resplit_rgb_ir.yaml"
E1_W="${ROOT}/results/S1_baselines/e1_yolo11n_rgb_only_640_ddp/weights/best.pt"
E2_W="${ROOT}/results/S1_baselines/e2_yolo11n_ir_only_640_ddp/weights/best.pt"
E5_W="${ROOT}/results/S2_fusion_mainline/yolo11n_e5_rgb_ir_640_ddp/weights/best.pt"
E6_W="${ROOT}/results/S2_fusion_mainline/yolo11n_e6_rgb_ir_640_ddp/weights/best.pt"
E10_W="${ROOT}/results/S3_resolution_sampling/e10_2_e6_768/weights/best.pt"
E13_3B_LIGHT_W="${ROOT}/results/S6_object_background_suppression/e13_3b_light_target_center_loss/weights/best.pt"
E14_3_W="${ROOT}/results/S6_object_background_suppression/e14_3_e13_3b_light_cebs_a005/weights/best.pt"
FINAL_W="${FINAL_W:-${E10_W}}"
BATCH_OBJECT="${BATCH_OBJECT:-16}"
WORKERS_OBJECT="${WORKERS_OBJECT:-8}"

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
  if [[ "${RUN_MODE}" == "run" ]]; then
    run_logged "${exp_id}" "${PY}" "${RUNNER}" run "${exp_id}" --work-dir "${WORK_DIR}"
  else
    echo "# log would be: ${LOG_DIR}/${exp_id}_$(date +%Y%m%d_%H%M%S).log"
    print_cmd "${PY}" "${RUNNER}" run "${exp_id}" --dry-run --work-dir "${WORK_DIR}"
  fi
}

object_eval_e6_after() {
  local exp_id="$1"
  local weights="$2"
  local image_metrics="$3"
  local out_dir="$4"
  local device="$5"
  local batch="$6"
  planned_only "${exp_id}_object" "run after unified validation completes" \
    "${PY}" "${ROOT}/scripts/e23_object_level_subset_eval.py" \
    --weights "${weights}" --validator e6 --mode rgb_ir --split val \
    --image-metrics "${image_metrics}" --imgsz 640 --batch "${batch}" \
    --workers "${WORKERS_OBJECT}" --device "${device}" --out-dir "${out_dir}"
}

object_eval_e13_after() {
  local exp_id="$1"
  local weights="$2"
  local image_metrics="$3"
  local out_dir="$4"
  local device="$5"
  planned_only "${exp_id}_object" "run after unified validation completes" \
    "${PY}" "${ROOT}/scripts/e23_object_level_subset_eval.py" \
    --weights "${weights}" --validator e13 --mode rgb_ir --split val \
    --image-metrics "${image_metrics}" --imgsz 640 --batch "${BATCH_OBJECT}" \
    --workers "${WORKERS_OBJECT}" --device "${device}" --out-dir "${out_dir}"
}

object_eval_e14_after() {
  local exp_id="$1"
  local weights="$2"
  local image_metrics="$3"
  local out_dir="$4"
  local device="$5"
  local alpha="$6"
  planned_only "${exp_id}_object" "run after unified validation completes" \
    "${PY}" "${ROOT}/scripts/e23_object_level_subset_eval.py" \
    --weights "${weights}" --validator e14 --mode rgb_ir --split val \
    --image-metrics "${image_metrics}" --imgsz 640 --batch "${BATCH_OBJECT}" \
    --workers "${WORKERS_OBJECT}" --device "${device}" --out-dir "${out_dir}" \
    --cebs-alpha "${alpha}" --dark-threshold 33.50320816040039 \
    --low-contrast-threshold 0.08425217866897583 --contrast-kernel 7 \
    --suppression-temperature 0.08
}

s6_current() {
  for id in E18_check E22_1 E23 E13_3b_light E22_2a E22_2b E14_1 E14_2 E14_3 E14_4 E24_0; do
    echo
    run_id "${id}"
  done
}

stage_tag_for_id() {
  case "$1" in
    E0_*) echo "S0" ;;
    E1|E2|E3|E4) echo "S1" ;;
    E5|E6) echo "S2" ;;
    E7_*|E8_*|E9_*|E10_*) echo "S3" ;;
    E11_*|E12_*|E13_1|E13_2|E13_3|E13_4|E13_5|E13_6) echo "S4" ;;
    E15_*|E16_*|E17_*|E18|E19|E20|E21) echo "S5" ;;
    E13_3b_light|E14_*|E18_check|E22_*|E23|E23_*|E24_*) echo "S6" ;;
    *) echo "S?" ;;
  esac
}

# E0 protocol and dataset audit ------------------------------------------------
e0_1() { echo "# E0_1: RGB/IR pair audit"; run_or_print "${PY}" "${ROOT}/scripts/dataset_resplit_dronevehicle.py" --root "${ROOT}" --audit-pairs; }
e0_2() { echo "# E0_2: official split leakage check"; run_or_print "${PY}" "${ROOT}/scripts/dataset_resplit_dronevehicle.py" --root "${ROOT}" --check-leakage; }
e0_3() { echo "# E0_3: confirm dark/small/dark-small subsets"; run_or_print "${PY}" "${ROOT}/scripts/dataset_resplit_dronevehicle.py" --root "${ROOT}" --build-subsets dark small dark-small; }
e0_4() { echo "# E0_4: build low-contrast subset from train thresholds"; run_or_print "${PY}" "${ROOT}/scripts/dataset_resplit_dronevehicle.py" --root "${ROOT}" --build-subsets low-contrast; }
e0_5() { echo "# E0_5: build tiny/small/medium/large size buckets"; run_or_print "${PY}" "${ROOT}/scripts/dataset_resplit_dronevehicle.py" --root "${ROOT}" --build-size-buckets tiny small medium large; }
e0_6() { echo "# E0_6: aggregate unified metric protocol artifacts"; run_or_print "${PY}" "${ROOT}/scripts/dark_small_experiment_runner.py" aggregate --out-dir "${ROOT}/results"; }
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
e7_6() { echo "# E7_6: skipped duplicate of E4 equal-weight WBF"; run_or_print "${PY}" "${RUNNER}" aggregate --out-dir "${ROOT}/results"; }
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
e13_3b_light() {
  echo "# E13_3b_light: target-scoped center-aware loss, S6 reviewed"
  runner_exp E13_3b_light
  object_eval_e13_after E13_3b_light \
    "${E13_3B_LIGHT_W}" \
    "${ROOT}/results/S6_object_background_suppression/e13_3b_light_target_center_loss_val/required_metrics.json" \
    "${ROOT}/results/S6_object_background_suppression/e23_e13_3b_light_object_level" \
    "${DEVICE_SINGLE}"
}
e13_4() { planned_only E13_4 "dark-small sample weighting" "${PY}" "${ROOT}/scripts/e13_train_tiny_aware_loss.py" --loss dark-small-weight --model "${E6_W}" --name e13_4_darksmall_weight --dry-run; }
e13_5() { planned_only E13_5 "center/location-sensitive bbox loss" "${PY}" "${ROOT}/scripts/e13_train_tiny_aware_loss.py" --loss center-aware --model "${E6_W}" --name e13_5_center_aware --dry-run; }
e13_6() { planned_only E13_6 "WIoU/CIoU comparison" "${PY}" "${ROOT}/scripts/e13_train_tiny_aware_loss.py" --loss wiou-ciou --model "${E6_W}" --name e13_6_wiou_ciou --dry-run; }
e14_1() {
  echo "# E14_1: E6 + CEBS alpha=0.05, S6 reviewed"
  runner_exp E14_1
  object_eval_e14_after E14_1 \
    "${ROOT}/results/S6_object_background_suppression/e14_1_e6_cebs_a005/weights/best.pt" \
    "${ROOT}/results/S6_object_background_suppression/e14_1_e6_cebs_a005_val/required_metrics.json" \
    "${ROOT}/results/S6_object_background_suppression/e23_e14_1_cebs_a005_object_level" \
    "${DEVICE_SINGLE}" 0.05
}
e14_2() {
  echo "# E14_2: E6 + CEBS alpha=0.10, S6 reviewed"
  runner_exp E14_2
  object_eval_e14_after E14_2 \
    "${ROOT}/results/S6_object_background_suppression/e14_2_e6_cebs_a010/weights/best.pt" \
    "${ROOT}/results/S6_object_background_suppression/e14_2_e6_cebs_a010_val/required_metrics.json" \
    "${ROOT}/results/S6_object_background_suppression/e23_e14_2_cebs_a010_object_level" \
    "${DEVICE_SINGLE}" 0.10
}
e14_3() {
  echo "# E14_3: E13_3b-light + CEBS alpha=0.05, S6 reviewed"
  runner_exp E14_3
  object_eval_e14_after E14_3 \
    "${E14_3_W}" \
    "${ROOT}/results/S6_object_background_suppression/e14_3_e13_3b_light_cebs_a005_val/required_metrics.json" \
    "${ROOT}/results/S6_object_background_suppression/e23_e14_3_e13_3b_light_cebs_a005_object_level" \
    "${DEVICE_SINGLE}" 0.05
}
e14_4() { planned_only E14_4 "skipped_not_justified: E14_3 and HN 1.5x/2x did not meet image/object candidate criteria" "${PY}" "${RUNNER}" run E14_4 --dry-run --work-dir "${WORK_DIR}"; }
e15_1() { planned_only E15_1 "YOLOv10n strong model comparison" "${YOLO}" detect train model=yolov10n.pt data="${RGB_IR_DATA}" imgsz=640 epochs=100 batch="${BATCH_SINGLE}" workers="${WORKERS}" device="${DEVICE}" project="${ROOT}/results" name=e15_1_yolov10n; }
e15_2() { planned_only E15_2 "RT-DETR-R18 comparison" "${YOLO}" detect train model=rtdetr-r18.pt data="${RGB_IR_DATA}" imgsz=640 epochs=100 batch=16 workers="${WORKERS}" device="${DEVICE}" project="${ROOT}/results" name=e15_2_rtdetr_r18; }
e15_3() { planned_only E15_3 "YOLO11s modality/model-capacity comparison" "${PY}" "${ROOT}/scripts/e15_train_yolo11s_comparison.py" --modes rgb ir fusion --model "${MODEL_S}" --device "${DEVICE}" --dry-run; }
e15_4() { planned_only E15_4 "final method vs strong models" "${PY}" "${ROOT}/scripts/e15_compare_final_models.py" --final "${FINAL_W}" --baselines e15_1 e15_2 e15_3 --dry-run; }
e16_1() { planned_only E16_1 "normal registration robustness baseline" "${PY}" "${ROOT}/scripts/e16_eval_registration_shift.py" --shift 0 --model "${FINAL_W}" --dry-run; }
e16_2() { planned_only E16_2 "IR random shift +/-2 px" "${PY}" "${ROOT}/scripts/e16_eval_registration_shift.py" --shift 2 --model "${FINAL_W}" --dry-run; }
e16_3() { planned_only E16_3 "IR random shift +/-4 px" "${PY}" "${ROOT}/scripts/e16_eval_registration_shift.py" --shift 4 --model "${FINAL_W}" --dry-run; }
e16_4() { planned_only E16_4 "IR random shift +/-8 px" "${PY}" "${ROOT}/scripts/e16_eval_registration_shift.py" --shift 8 --model "${FINAL_W}" --dry-run; }
e16_5() { planned_only E16_5 "gated fusion under registration shift" "${PY}" "${ROOT}/scripts/e16_eval_registration_shift.py" --shift 4 --model "${ROOT}/results/S4_head_gate_loss/e12_best/weights/best.pt" --gated --dry-run; }
e17_1() { e0_4; }
e17_2() { planned_only E17_2 "low-contrast evaluation across selected models" "${PY}" "${ROOT}/scripts/e17_eval_low_contrast_models.py" --models E2="${E2_W}" E4="${ROOT}/results/S1_baselines/e4_late_fusion_wbf_val" E10="${E10_W}" FINAL="${FINAL_W}" --dry-run; }
e17_3() { planned_only E17_3 "report low-contrast AP/Recall/FPPI" "${PY}" "${ROOT}/scripts/e17_report_low_contrast.py" --results-val "${ROOT}/results" --dry-run; }
e18_check() { echo "# E18_check: E13_3b seed independence audit"; run_logged E18_check "${PY}" "${ROOT}/scripts/e18_run_multiseed.py" --out-dir "${ROOT}/results/S6_object_background_suppression/e18_check_e13_3b_seed_integrity"; }
e18() { planned_only E18 "full multi-seed stability wrapper; not part of S6 current executable set" "${PY}" "${ROOT}/scripts/e18_run_multiseed.py" --dry-run; }
e19() { planned_only E19 "efficiency/deployment metrics" "${PY}" "${ROOT}/scripts/e19_measure_efficiency.py" --models E2="${E2_W}" E6="${E6_W}" E10="${E10_W}" FINAL="${FINAL_W}" --dry-run; }
e20() { planned_only E20 "per-class/tiny/failure-case analysis" "${PY}" "${ROOT}/scripts/e20_failure_case_analysis.py" --model "${FINAL_W}" --out-dir "${ROOT}/results/S5_diagnostic_optimization/e20_failure_analysis" --dry-run; }
e21() { planned_only E21 "locked test-set final evaluation; do not run during tuning" "${PY}" "${ROOT}/scripts/e21_locked_test_eval.py" --protocol DroneVehicle-DarkSmall-v1 --models E2 E4 E10 FINAL --dry-run; }
e22_1() { echo "# E22_1: train-split hard negative list export; background_far only is train-allowed"; run_logged E22_1 "${PY}" "${ROOT}/scripts/e22_hard_negative_mining.py" --mode lists --taxonomy-csv "${ROOT}/results/S5_diagnostic_optimization/e22_0_train_hard_negative_taxonomy/hard_negative_list.csv" --split train --out-dir "${ROOT}/results/S6_object_background_suppression/e22_1_train_hard_negative_lists"; }
e22_2a() {
  echo "# E22_2a: E6 + train background_far HN 1.5x, S6 reviewed"
  runner_exp E22_2a
  object_eval_e6_after E22_2a \
    "${ROOT}/results/S6_object_background_suppression/e22_2a_e6_background_far_hn15_gpu0_b48/weights/best.pt" \
    "${ROOT}/results/S6_object_background_suppression/e22_2a_e6_background_far_hn15_val/required_metrics.json" \
    "${ROOT}/results/S6_object_background_suppression/e23_e22_2a_hn15_object_level" \
    "${DEVICE_SINGLE}" 48
}
e22_2b() {
  echo "# E22_2b: E6 + train background_far HN 2x, S6 reviewed"
  runner_exp E22_2b
  object_eval_e6_after E22_2b \
    "${ROOT}/results/S6_object_background_suppression/e22_2b_e6_background_far_hn2/weights/best.pt" \
    "${ROOT}/results/S6_object_background_suppression/e22_2b_e6_background_far_hn2_val/required_metrics.json" \
    "${ROOT}/results/S6_object_background_suppression/e23_e22_2b_hn2_object_level" \
    "${DEVICE_SINGLE}" 48
}
e22_2() { planned_only E22_2 "legacy alias; S6 uses exact E22_2a/E22_2b entries only" "${PY}" "${RUNNER}" list E22_2a E22_2b; }
e22_3() { planned_only E22_3 "not allowed in S6: no hard-negative 3x/5x/all-HN expansion" "${PY}" "${RUNNER}" list E22_2a E22_2b; }
e22_4() { planned_only E22_4 "not allowed in S6: no hard-negative 3x/5x/all-HN expansion" "${PY}" "${RUNNER}" list E22_2a E22_2b; }
e22_5() { planned_only E22_5 "not allowed in S6: no CEBS + HN combo after E14_3/E22_2a/E22_2b review" "${PY}" "${RUNNER}" list E14_3 E22_2a E22_2b; }
e23() { echo "# E23: E6 object-level evaluator"; run_logged E23 "${PY}" "${ROOT}/scripts/e23_object_level_subset_eval.py" --weights "${E6_W}" --validator e6 --mode rgb_ir --split val --image-metrics "${ROOT}/results/S2_fusion_mainline/e6_feature_fusion_multiscale_val/required_metrics.json" --imgsz 640 --batch "${BATCH_OBJECT}" --workers "${WORKERS_OBJECT}" --device "${DEVICE_SINGLE}" --out-dir "${ROOT}/results/S6_object_background_suppression/e23_object_level_evaluator"; }
e23_1() { echo "# E23_1: object-level subset build smoke"; run_or_print "${PY}" "${ROOT}/scripts/e23_object_level_subset_eval.py" --weights "${E6_W}" --validator e6 --mode rgb_ir --split val --imgsz 640 --batch "${BATCH_OBJECT}" --workers "${WORKERS_OBJECT}" --device "${DEVICE_SINGLE}" --out-dir "${ROOT}/results/S6_object_background_suppression/e23_object_level_evaluator_smoke" --build-only; }
e23_2() { e23; }
e23_3() { echo "# E23_3: compare image-level and object-level metrics"; run_or_print "${PY}" "${ROOT}/scripts/e23_compare_metric_scopes.py" --results-val "${ROOT}/results" --out-dir "${ROOT}/results/S6_object_background_suppression/e23_metric_scope_comparison"; }
e24_0() { planned_only E24_0 "blocked_no_valid_candidate: no image/object-valid candidate to freeze; E24 CLI is still a demo placeholder" "${PY}" "${ROOT}/scripts/e24_freeze_repro_config.py" --step freeze-configs --out-dir "${ROOT}/configs/frozen" --dry-run; }
e24_1() { planned_only E24_1 "freeze final configs" "${PY}" "${ROOT}/scripts/e24_freeze_repro_config.py" --step freeze-configs --out-dir "${ROOT}/configs/frozen" --dry-run; }
e24_2() { planned_only E24_2 "audit config paths, commit, seed, weights, results" "${PY}" "${ROOT}/scripts/e24_freeze_repro_config.py" --step audit --results-val "${ROOT}/results" --dry-run; }
e24_3() { planned_only E24_3 "verify leaderboard vs result dirs" "${PY}" "${ROOT}/scripts/e24_freeze_repro_config.py" --step verify-leaderboard --leaderboard "${ROOT}/results/dark_small_experiment_leaderboard.csv" --dry-run; }

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
E13_1 E13_2 E13_3 E13_3b_light E13_4 E13_5 E13_6
E14_1 E14_2 E14_3 E14_4
E15_1 E15_2 E15_3 E15_4
E16_1 E16_2 E16_3 E16_4 E16_5
E17_1 E17_2 E17_3
E18_check E18 E19 E20 E21
E22_1 E22_2a E22_2b E22_2 E22_3 E22_4 E22_5
E23 E23_1 E23_2 E23_3
E24_0 E24_1 E24_2 E24_3
EOF
}

usage() {
  cat <<'EOF'
Usage: scripts/train/run_dark_small_experiment_demos.sh {list|all|EXPERIMENT_ID...}

Default: dry-run. Set RUN_MODE=run for implemented experiments.
Current route: do not run E21/test-set or strong-model E15 during tuning.
Use S6 to print the current-stage demo sequence.
Each experiment header prints its current stage tag as "Sx / Ex".
EOF
}

run_id() {
  local display_id lookup_id stage_tag
  display_id="${1//-/_}"
  display_id="${display_id/#e/E}"
  display_id="${display_id/#s/S}"
  lookup_id="$(echo "${display_id}" | tr '[:lower:]' '[:upper:]')"
  stage_tag="$(stage_tag_for_id "${lookup_id}")"
  local id_lc
  id_lc="$(echo "$1" | tr '[:upper:]-' '[:lower:]_')"
  if [[ "${id_lc}" != "s6" ]]; then
    echo "# ${stage_tag} / ${display_id}"
  fi
  case "${id_lc}" in
    e0_1) e0_1 ;; e0_2) e0_2 ;; e0_3) e0_3 ;; e0_4) e0_4 ;; e0_5) e0_5 ;; e0_6) e0_6 ;; e0_7) e0_7 ;;
    e1) e1 ;; e2) e2 ;; e3) e3 ;; e4) e4 ;; e5) e5 ;; e6) e6 ;;
    e7_1) e7_1 ;; e7_2) e7_2 ;; e7_3) e7_3 ;; e7_4) e7_4 ;; e7_5) e7_5 ;; e7_6) e7_6 ;;
    e8_1) e8_1 ;; e8_2) e8_2 ;; e8_3) e8_3 ;; e8_4) e8_4 ;;
    e9_1) e9_1 ;; e9_2) e9_2 ;;
    e10_1) e10_1 ;; e10_2) e10_2 ;; e10_3) e10_3 ;; e10_4) e10_4 ;;
    e11_1) e11_1 ;; e11_2) e11_2 ;; e11_3) e11_3 ;;
    e12_1) e12_1 ;; e12_1b) e12_1b ;; e12_2) e12_2 ;; e12_3) e12_3 ;; e12_4) e12_4 ;;
    e13_1) e13_1 ;; e13_2) e13_2 ;; e13_3) e13_3 ;; e13_3b_light) e13_3b_light ;; e13_4) e13_4 ;; e13_5) e13_5 ;; e13_6) e13_6 ;;
    e14_1) e14_1 ;; e14_2) e14_2 ;; e14_3) e14_3 ;; e14_4) e14_4 ;;
    e15_1) e15_1 ;; e15_2) e15_2 ;; e15_3) e15_3 ;; e15_4) e15_4 ;;
    e16_1) e16_1 ;; e16_2) e16_2 ;; e16_3) e16_3 ;; e16_4) e16_4 ;; e16_5) e16_5 ;;
    e17_1) e17_1 ;; e17_2) e17_2 ;; e17_3) e17_3 ;;
    e18_check) e18_check ;; e18) e18 ;; e19) e19 ;; e20) e20 ;; e21) e21 ;;
    e22_1) e22_1 ;; e22_2a) e22_2a ;; e22_2b) e22_2b ;; e22_2) e22_2 ;; e22_3) e22_3 ;; e22_4) e22_4 ;; e22_5) e22_5 ;;
    e23) e23 ;; e23_1) e23_1 ;; e23_2) e23_2 ;; e23_3) e23_3 ;;
    e24_0) e24_0 ;; e24_1) e24_1 ;; e24_2) e24_2 ;; e24_3) e24_3 ;;
    s6) s6_current ;;
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
