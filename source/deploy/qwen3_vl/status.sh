#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

QWEN_PORT="${QWEN_PORT:-8001}"
pid_file="${RUN_DIR}/server.pid"

if [[ ! -s "${pid_file}" ]] || ! kill -0 "$(<"${pid_file}")" 2>/dev/null; then
  echo "status=stopped"
  exit 1
fi

echo "status=running pid=$(<"${pid_file}")"
if [[ -s "${API_KEY_FILE}" ]]; then
  curl -fsS \
    -H "Authorization: Bearer $(<"${API_KEY_FILE}")" \
    "http://127.0.0.1:${QWEN_PORT}/v1/models" \
    | "${PYTHON}" -m json.tool
fi
