#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/disk2/lhr/VSD}"
PY="${PY:-/mnt/disk2/lhr/conda_envs/vsd/bin/python}"
exec "$ROOT/scripts/train/launch_s7_1a_utah_lite_dual4090_background.sh"
