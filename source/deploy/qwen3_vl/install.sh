#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

UV_BIN="${UV_BIN:-/staff/tangjiarui/.local/bin/uv}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv was not found at ${UV_BIN}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
  "${UV_BIN}" venv --python 3.10 "${VENV_DIR}"
fi

"${UV_BIN}" pip install \
  --python "${PYTHON}" \
  --index-url "${PYPI_INDEX}" \
  -r "${DEPLOY_DIR}/requirements-serve.txt"

"${PYTHON}" -c "import bitsandbytes, torch, transformers, vllm; print('torch', torch.__version__); print('transformers', transformers.__version__); print('vllm', vllm.__version__); print('bitsandbytes', bitsandbytes.__version__)"
