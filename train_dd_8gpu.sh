#!/usr/bin/env bash
# Train Discrete Diffusion policies on 8 GPUs.
# Number of levels must divide number of GPUs (README); we use 8 levels for 8 GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use GPUs 0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 8 levels so that 8 | 8 (required for level sharding)
LEVEL_PATHS=(
  worlds/l/grasp_easy.json
  worlds/l/catapult.json
  worlds/l/cartpole_thrust.json
  worlds/l/hard_lunar_lander.json
  worlds/l/mjc_half_cheetah.json
  worlds/l/mjc_swimmer.json
  worlds/l/mjc_walker.json
  worlds/l/h17_unicycle.json
)

RUN_PATH="${1:-}"
if [[ -z "$RUN_PATH" ]]; then
  echo "Usage: $0 <run-path>" >&2
  echo "  run-path: path to expert run containing data/ (e.g. ./logs-expert/<wandb-run-name> or rtc-assets/expert)" >&2
  exit 1
fi

uv run src/train_dd.py \
  --config.run-path "$RUN_PATH" \
  --config.level-paths "${LEVEL_PATHS[@]}"
