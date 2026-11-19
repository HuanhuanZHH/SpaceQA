#!/usr/bin/env bash

set -euo pipefail

OPC_ROOT="${OPC_ROOT:-${HOME}/spaceqa-runtime}"
DEPLOY_DIR="${DEPLOY_DIR:-${OPC_ROOT}/spaceqa-agent/deploy/qwen3_vl}"
VENV_DIR="${VENV_DIR:-${OPC_ROOT}/.venv-qwen3vl}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${OPC_ROOT}/.uv-cache}"
HF_HOME="${HF_HOME:-${OPC_ROOT}/.hf-cache}"
MODEL_DIR="${MODEL_DIR:-${OPC_ROOT}/models/Qwen3-VL-8B-Instruct}"
LOG_DIR="${LOG_DIR:-${OPC_ROOT}/logs/qwen3-vl}"
RUN_DIR="${RUN_DIR:-${OPC_ROOT}/run/qwen3-vl}"
API_KEY_FILE="${API_KEY_FILE:-${OPC_ROOT}/.qwen3vl-api-key}"
PYTHON="${PYTHON:-${VENV_DIR}/bin/python}"

mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${MODEL_DIR}" "${LOG_DIR}" "${RUN_DIR}"

export OPC_ROOT DEPLOY_DIR VENV_DIR UV_CACHE_DIR HF_HOME MODEL_DIR LOG_DIR RUN_DIR
export API_KEY_FILE PYTHON
