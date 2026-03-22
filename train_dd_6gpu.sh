#!/usr/bin/env bash
# Train Discrete Diffusion policies on GPUs 0,1,2,3,4,5 (6 GPUs).
# Use 6 levels so that level sharding has one level per GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

LEVEL_PATHS=(
  worlds/l/grasp_easy.json
  worlds/l/catapult.json
  worlds/l/cartpole_thrust.json
  worlds/l/hard_lunar_lander.json
  worlds/l/mjc_half_cheetah.json
  worlds/l/mjc_swimmer.json
)

RUN_PATH="${1:-}"
if [[ -z "$RUN_PATH" ]]; then
  echo "Usage: $0 <run-path>" >&2
  echo "  run-path: path to expert run containing data/ (e.g. ./logs-expert/<wandb-run-name>)" >&2
  exit 1
fi

uv run python -m train_dd \
  --run-path "$RUN_PATH" \
  --level-paths "${LEVEL_PATHS[@]}"
