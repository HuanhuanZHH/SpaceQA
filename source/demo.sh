#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: ./demo.sh INPUT_PATH [OUTPUT_PATH]" >&2
  exit 2
fi

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INPUT_PATH=$1
STAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_PATH=${2:-"$APP_ROOT/runs/demo-$STAMP"}
SEED=${OPC_SEED:-20261021}

if [ ! -d "$INPUT_PATH" ]; then
  echo "Input directory does not exist: $INPUT_PATH" >&2
  exit 2
fi

if [ -e "$OUTPUT_PATH" ]; then
  echo "Output path already exists; choose a new path: $OUTPUT_PATH" >&2
  exit 2
fi

echo "[1/2] Running OPC evaluation"
python3 "$APP_ROOT/opc_agent.py" scan \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --seed "$SEED"

echo "[2/2] Auditing generated outputs"
python3 "$APP_ROOT/tools/audit_run.py" "$OUTPUT_PATH"

echo "Demo result: $OUTPUT_PATH"
echo "Scorecards: $OUTPUT_PATH/04_评分卡"
echo "Ranking: $OUTPUT_PATH/05_排序表/ranking_global.csv"
echo "Findings: $OUTPUT_PATH/06_问题清单/findings.jsonl"
echo "Evidence: $OUTPUT_PATH/07_证据索引/evidence_index.csv"
