#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ ! -x "${VENV_DIR}/bin/hf" ]]; then
  echo "Run install.sh before download.sh." >&2
  exit 1
fi

exec "${VENV_DIR}/bin/hf" download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir "${MODEL_DIR}" \
  --max-workers "${HF_MAX_WORKERS:-4}" \
  --format quiet
