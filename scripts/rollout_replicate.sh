#!/bin/bash
# Replicate original results: deterministic mode (choice_temperature=0, decode_temperature=0).
# Output: eval_logs/<name>/ (default: eval_logs/l1_replicate)
#
# Run: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/rollout_replicate.sh
# Or:  OUTPUT_BASE=eval_logs RUN_PATH=./logs-dd/l1 ./scripts/rollout_replicate.sh

set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

RUN_PATH="${RUN_PATH:-./logs-dd/l1}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_logs/l1_replicate}"
NUM_GPUS="${NUM_GPUS:-6}"

mkdir -p "$OUTPUT_DIR"
echo "=== Replicate (deterministic): choice_temp=0 decode_temp=0 ==="
uv run python src/eval_dd.py \
  --run-path "$RUN_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --num-gpus "$NUM_GPUS" \
  --config.choice-temperature 0 \
  --config.decode-temperature 0

echo "Done. Results in $OUTPUT_DIR"
