## Installation

```bash
# Clone Kinetix submodule
git submodule update --init
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install dependencies
uv sync --python
```

## Reproduce results

For steps 1-3 (training experts, generating data, and training continuous flow matching baselines), follow the instructions in the [original RTC-Kinetix repo](https://github.com/Physical-Intelligence/real-time-chunking-kinetix). The expert data and pre-trained continuous baselines from that repo are directly reusable here.

Note that, for all scripts, your number of GPUs must divide the number of levels (default 12) because computation is sharded over levels.

All outputs are stored under `paper/`:
- `paper/checkpoints/` — trained model checkpoints
- `paper/stats/` — evaluation result CSVs

## Pre-trained checkpoints

This branch **ships the pre-trained discrete diffusion policies** used in the paper, so you can skip steps 1–4 and evaluate directly. They live under [`paper/checkpoints/`](paper/checkpoints):

```
paper/checkpoints/
├── config.json
├── model_config.json
└── 31/                       # run name (numeric checkpoint dir)
    └── policies/             # one policy per Kinetix level
        ├── worlds_l_car_launch.pkl
        ├── worlds_l_cartpole_thrust.pkl
        └── ...               # 12 levels total
```

To evaluate them, point `--run-path` at the **parent** directory (`paper/checkpoints`); the script auto-selects the numeric run dir (`31`):

```bash
uv run src/eval_dd.py --run-path paper/checkpoints
```

This sweeps inference delays (0–4) and all methods (naive, discrete RTC, adaptive discrete RTC, BID) with 2048 trials per level, writing results to `paper/stats/results.csv`. Add `--num-gpus N` to shard levels across GPUs.

### Discrete diffusion policy training

4. Train discrete diffusion policies: `uv run src/train_dd.py --config.run-path <path-to-expert-data>`
    - This loads the expert data from step 2 and trains discrete diffusion (MaskGIT-style) policies for each level.
    - Checkpoints are written to `paper/checkpoints/<wandb-run-name>`.

### Evaluation

5. Evaluate discrete diffusion policies: `uv run src/eval_dd.py --run-path paper/checkpoints` (or your own `paper/checkpoints/<wandb-run-name>`)
    - This sweeps over inference delays (0-4) and evaluates all methods (naive, discrete RTC, adaptive discrete RTC, BID, VLASH) with 2048 trials per level.
    - The pre-trained checkpoints shipped in this branch (see [Pre-trained checkpoints](#pre-trained-checkpoints)) work directly with this command.
    - Results are written to `paper/stats/results.csv`.
    - Use `--num-gpus N` to distribute levels across GPUs via separate processes.

6. Evaluate continuous flow baselines: `uv run src/eval_flow.py --run-path <path-to-bc-checkpoints>`
    - Results are written to `paper/stats/results.csv`.
