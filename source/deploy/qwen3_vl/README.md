# Qwen3-VL optional local service

This directory can run Qwen3-VL-8B-Instruct behind an OpenAI-compatible local endpoint. Model output is a review candidate only; deterministic rules still own scores, ratios, and ranking.

Set these paths for your machine before running the scripts:

```bash
export OPC_ROOT=/absolute/path/to/OPC
export MODEL_DIR="$OPC_ROOT/models/Qwen3-VL-8B-Instruct"
export VENV_DIR="$OPC_ROOT/.venv-qwen3vl"
export LLM_API_KEY_FILE="$OPC_ROOT/.qwen3vl-api-key"
```

Install/download and start:

```bash
./install.sh
./download.sh
QWEN_GPU=0 ./start.sh
./status.sh
./smoke_test.py
```

The service binds to localhost. Use an SSH tunnel for a remote demo. Create candidate crops outside the repository and stop the service with `./stop.sh` when finished. Do not commit model weights, caches, or the API key file.
