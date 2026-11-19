#!/bin/sh
set -eu

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

exec python3 "$APP_ROOT/opc_agent.py" scan \
  --input "${INPUT_ROOT:-/input}" \
  --output "${OUTPUT_ROOT:-/output}" \
  --seed "${OPC_SEED:-20261021}"
