#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

pid_file="${RUN_DIR}/server.pid"
log_file="${LOG_DIR}/server.log"

if [[ -s "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
  echo "Qwen3-VL server is already running (PID $(<"${pid_file}"))."
  exit 0
fi

nohup "${DEPLOY_DIR}/serve.sh" >"${log_file}" 2>&1 </dev/null &
pid=$!
echo "${pid}" >"${pid_file}"
echo "Started Qwen3-VL server as PID ${pid}. Log: ${log_file}"
