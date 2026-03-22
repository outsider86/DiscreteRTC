Simulated experiments for discrete diffusion real-time chunking in Kinetix, extending the work from [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339) and [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964).

## Installation

```bash
# Clone Kinetix submodule
git submodule update --init
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install dependencies (Python 3.12 recommended - dm-tree wheels unavailable for 3.13)
uv sync --python 3.12
```

## Reproduce results

For steps 1-3 (training experts, generating data, and training continuous flow matching baselines), follow the instructions in the [original RTC-Kinetix repo](https://github.com/Physical-Intelligence/real-time-chunking-kinetix). The expert data and pre-trained continuous baselines from that repo are directly reusable here.

Note that, for all scripts, your number of GPUs must divide the number of levels (default 12) because computation is sharded over levels.

### Discrete diffusion policy training

4. Train discrete diffusion policies: `uv run src/train_dd.py --run-path ./logs-expert/<wandb-run-name>`
    - This loads the expert data from step 2 and trains discrete diffusion (MaskGIT-style) policies for each level.
    - Checkpoints are written to `./logs-dd/<wandb-run-name>`.

### Evaluation

5. Evaluate discrete diffusion policies: `uv run src/eval_dd.py --run-path ./logs-dd/<wandb-run-name> --output-dir <output-dir>`
    - This sweeps over inference delays (0-4) and evaluates all methods (naive, discrete RTC, adaptive discrete RTC, BID, VLASH) with 2048 trials per level.
    - Results are written to `<output-dir>/results.csv`.
    - Use `--num-gpus N` to distribute levels across GPUs via separate processes.

6. Evaluate continuous flow baselines: `uv run src/eval_flow.py --run-path ./logs-bc/<wandb-run-name> --output-dir <output-dir>`

### Optional: fine-tuning and VLASH training

- Fine-tune with inpainting mask: `uv run src/finetune_dd.py --run-path ./logs-expert/<wandb-run-name> --load-dir ./logs-dd/<wandb-run-name>/<epoch>`
- Self-forcing fine-tuning: `uv run src/finetune_dd_self_forcing.py --run-path ./logs-expert/<wandb-run-name> --load-dir ./logs-dd/<wandb-run-name>/<epoch>`
- VLASH training: `uv run src/train_vlash_dd.py --run-path ./logs-expert/<wandb-run-name>`

## Visualization

Generate comparison plots (continuous vs discrete):
```bash
uv run visualization/plot_comparison_grid.py \
    --continuous-csv <continuous-results.csv> \
    --discrete-csv <discrete-results.csv> \
    --output <output.png>
```
