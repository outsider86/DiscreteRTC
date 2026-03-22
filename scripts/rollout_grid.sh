#!/bin/bash
# Grid rollout: 4×4 experiments across choice_temperature and decode_temperature in [0, 0.01, 0.1, 1]
# temp=0 → deterministic (argmax / lowest-conf unmasking); temp>0 → sampling with that temperature
#
# Run: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./scripts/rollout_grid.sh

set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

RUN_PATH="${RUN_PATH:-./logs-dd/l1}"
OUTPUT_BASE="${OUTPUT_BASE:-eval_output}"
NUM_GPUS="${NUM_GPUS:-6}"

TEMPS=(0 0.01 0.1 1)

for ct in "${TEMPS[@]}"; do
  for dt in "${TEMPS[@]}"; do
    OUT="${OUTPUT_BASE}/grid_choice${ct}_decode${dt}"
    mkdir -p "$OUT"

    # temp=0 → deterministic; temp>0 → sampling
    echo "=== choice${ct}_decode${dt}: choice_temp=$ct decode_temp=$dt ==="
    uv run python src/eval_dd.py \
      --run-path "$RUN_PATH" \
      --output-dir "$OUT" \
      --num-gpus "$NUM_GPUS" \
      --config.choice-temperature "$ct" \
      --config.decode-temperature "$dt"
  done
done

echo "Done. Results in ${OUTPUT_BASE}/grid_choice*_decode*"
