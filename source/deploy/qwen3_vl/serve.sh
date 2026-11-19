#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/common.sh"

QWEN_GPU="${QWEN_GPU:-0}"
QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8001}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-4096}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.35}"
QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-2}"
QWEN_MIN_FREE_MIB="${QWEN_MIN_FREE_MIB:-14000}"

if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "Model download is incomplete: ${MODEL_DIR}" >&2
  exit 1
fi

for shard in "${MODEL_DIR}"/model-*.safetensors; do
  [[ -f "${shard}" ]] || {
    echo "Model shard set is incomplete: ${MODEL_DIR}" >&2
    exit 1
  }
done

free_mib="$(nvidia-smi --id="${QWEN_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if [[ ! "${free_mib}" =~ ^[0-9]+$ ]] || (( free_mib < QWEN_MIN_FREE_MIB )); then
  echo "GPU ${QWEN_GPU} has ${free_mib:-unknown} MiB free; ${QWEN_MIN_FREE_MIB} MiB is required." >&2
  exit 1
fi

if [[ ! -s "${API_KEY_FILE}" ]]; then
  umask 077
  "${PYTHON}" -c "import secrets; print(secrets.token_urlsafe(32))" > "${API_KEY_FILE}"
fi

export CUDA_VISIBLE_DEVICES="${QWEN_GPU}"
export TOKENIZERS_PARALLELISM=false
export VLLM_API_KEY="$(<"${API_KEY_FILE}")"

exec "${VENV_DIR}/bin/vllm" serve "${MODEL_DIR}" \
  --served-model-name Qwen3-VL-8B-Instruct \
  --host "${QWEN_HOST}" \
  --port "${QWEN_PORT}" \
  --dtype bfloat16 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --tensor-parallel-size 1 \
  --max-model-len "${QWEN_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${QWEN_MAX_NUM_SEQS}" \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --generation-config vllm \
  --enforce-eager \
  --no-enable-log-requests
