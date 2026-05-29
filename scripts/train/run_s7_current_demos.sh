#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/disk2/lhr/VSD}"
RUN_MODE="${RUN_MODE:-dry-run}"

cd "$ROOT"

echo "# S7-A current route"
RUN_MODE="$RUN_MODE" "$ROOT/scripts/train/run_dark_small_experiment_demos.sh" S7_0
echo
RUN_MODE="$RUN_MODE" "$ROOT/scripts/train/run_dark_small_experiment_demos.sh" S7_1a

cat <<'EOF'

# Launch S7_1a in the background with logs:
#   scripts/train/launch_s7_1a_utah_lite_dual4090_background.sh
#
# After S7_1a train finishes, run:
#   RUN_MODE=run scripts/train/run_dark_small_experiment_demos.sh S7_1a_val S7_1a_object S7_1a_export S7_1a_gate
EOF
