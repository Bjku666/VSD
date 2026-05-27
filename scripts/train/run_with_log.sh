#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "用法: $0 <log-file> <command...>" >&2
  exit 2
fi

LOG_FILE="$1"
shift

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[run-with-log] $(date +%Y-%m-%dT%H:%M:%S) start: $*"
"$@"
echo "[run-with-log] $(date +%Y-%m-%dT%H:%M:%S) done: $*"