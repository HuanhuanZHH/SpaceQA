#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

pid_file="${RUN_DIR}/server.pid"
if [[ ! -s "${pid_file}" ]]; then
  echo "Qwen3-VL server is not running."
  exit 0
fi

pid="$(<"${pid_file}")"
if kill -0 "${pid}" 2>/dev/null; then
  kill "${pid}"
  for _ in {1..30}; do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 1
  done
fi
rm -f "${pid_file}"
echo "Qwen3-VL server stopped."
