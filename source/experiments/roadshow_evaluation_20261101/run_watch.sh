#!/usr/bin/env bash
set -euo pipefail
cd /staff/tangjiarui/OPC/opc-agent/experiments/roadshow_evaluation_20261101
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset OPC_ALL_FAMILY_OUT
exec /staff/tangjiarui/OPC/.venv-image-models/bin/python -u roadshow.py watch
