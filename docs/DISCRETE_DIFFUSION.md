# Discrete Diffusion Policy Implementation

This document describes the discrete diffusion policy implementation for action chunking on the Kinetix benchmark. The implementation mirrors the flow matching pipeline (`model.py`, `train_flow.py`, `eval_flow.py`) with a discrete diffusion formulation.

## Overview

| File | Purpose |
|------|---------|
| `src/model_dd.py` | Discrete diffusion policy model (interface aligned with `model.py`) |
| `src/train_dd.py` | Training script for discrete diffusion policies |
| `src/eval_dd.py` | Evaluation script for discrete diffusion policies |

**Dependencies:** Uses the same expert data and environment setup as the flow pipeline. Requires steps 1–2 from the main README (train expert, generate data) before training DD policies.

---

## 1. Model (`src/model_dd.py`)

### Architecture

- **Discretization:** Maps continuous actions in `[-1, 1]` to `K` bins (default `num_bins=256`).
- **Forward process:** Uniform corruption—each bin is replaced with a random bin with probability `t`.
- **Backbone:** Same MLP-Mixer structure as the flow policy, with time conditioning.
- **Output:** Categorical logits over bins; converted back to continuous actions via bin centers.

### ModelConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `channel_dim` | 256 | Channel dimension |
| `channel_hidden_dim` | 512 | Hidden dimension in mixer blocks |
| `token_hidden_dim` | 64 | Token mixing hidden dim |
| `num_layers` | 4 | Number of MLP-Mixer blocks |
| `action_chunk_size` | 8 | Actions per chunk |
| `num_bins` | 256 | Bins per action dimension |
| `simulated_delay` | `None` | Optional; enables training-time RTC when set |

### Policy Methods

| Method | Description |
|--------|-------------|
| `action(rng, obs, num_steps)` | Iterative denoising; returns continuous action chunk |
| `realtime_action(...)` | Prefix masking for already-executed actions |
| `bid_action(...)` | Best-of-n selection by prefix consistency |
| `loss(rng, obs, action)` | Cross-entropy loss on predicted bins |

---

## 2. Training (`src/train_dd.py`)

### Usage

```bash
# Basic training (requires expert data from steps 1–2)
uv run src/train_dd.py --config.run-path ./logs-expert/<wandb-run-name>
```

### Output

- **WandB project:** `rtc-kinetix-dd`
- **Log directory:** `./logs-dd/<wandb-run-name>/`
- **Checkpoints:** `logs-dd/<run-name>/<epoch>/policies/<level>.pkl`

### Key CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config.run-path` | (required) | Path to expert run containing `data/*.npz` |
| `--config.batch-size` | 512 | Batch size |
| `--config.num-epochs` | 32 | Training epochs |
| `--config.load-dir` | `None` | Path to pretrained DD checkpoints for fine-tuning |
| `--config.learning-rate` | 3e-4 | Learning rate |
| `--config.grad-norm-clip` | 10.0 | Gradient clipping |

### Example: Fine-tuning from Checkpoint

```bash
uv run src/train_dd.py --config.run-path ./logs-expert/<run> --config.load-dir logs-dd/<dd-run>/24 --config.num-epochs 8
```

---

## 3. Evaluation (`src/eval_dd.py`)

### Usage

```bash
uv run src/eval_dd.py --run-path logs-dd/<wandb-run-name> --output-dir <output-dir>
```

### Output

- **Default output directory:** `eval_dd_output`
- **Results file:** `<output-dir>/results.csv`

### Evaluation Methods

| Method | Description |
|--------|-------------|
| `naive` | Standard sampling, no prefix guidance |
| `realtime` | Prefix guidance with exponential schedule |
| `bid` | Best-of-n by prefix consistency |
| `hard_masking` | Realtime with zero-weight prefix schedule |

### Sweep

- **Inference delay:** 0, 1, 2, 3, 4
- **Execute horizon:** 1 to `8 - inference_delay`

### Key CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--run-path` | (required) | Path to DD run (e.g. `logs-dd/<run-name>`) |
| `--output-dir` | `eval_dd_output` | Directory for `results.csv` |
| `--config.step` | -1 | Checkpoint epoch to load (-1 = latest) |
| `--config.num-evals` | 128 | Rollouts per level (use 2048 for paper) |
| `--config.num-flow-steps` | 5 | Denoising steps at inference |
| `--config.weak-step` | `None` | Epoch for weak policy (BID with bid_k) |

---

## 4. Full Workflow

### Prerequisites

1. **Train expert policies**
   ```bash
   uv run src/train_expert.py
   ```

2. **Generate expert data**
   ```bash
   uv run src/generate_data.py --config.run-path ./logs-expert/<wandb-run-name>
   ```

### Discrete Diffusion Pipeline

3. **Train DD policies**
   ```bash
   uv run src/train_dd.py --config.run-path ./logs-expert/<wandb-run-name>
   ```

4. **Evaluate DD policies**
   ```bash
   uv run src/eval_dd.py --run-path logs-dd/<wandb-run-name> --output-dir eval_dd_output
   ```

### Using Pre-trained Data

If you have downloaded `gs://rtc-assets/expert/` (or equivalent):

```bash
uv run src/train_dd.py --config.run-path rtc-assets/expert
uv run src/eval_dd.py --run-path logs-dd/<run-name> --output-dir eval_dd_output
```

---

## 5. File Structure

```
src/
├── model.py        # Flow policy (original)
├── model_dd.py     # Discrete diffusion policy
├── train_flow.py   # Flow training
├── train_dd.py     # DD training
├── eval_flow.py    # Flow evaluation
├── eval_dd.py      # DD evaluation
├── train_expert.py
├── generate_data.py
└── ...

logs-dd/            # DD training outputs
└── <wandb-run-name>/
    └── <epoch>/
        └── policies/
            └── <level>.pkl
```

---

## 6. Comparison with Flow Pipeline

| Aspect | Flow (`model.py`) | Discrete Diffusion (`model_dd.py`) |
|--------|-------------------|-----------------------------------|
| **State representation** | Continuous noise → action | Discrete bins |
| **Forward process** | Linear interpolation | Uniform corruption |
| **Loss** | MSE on velocity | Cross-entropy on bins |
| **Sampling** | ODE integration | Iterative denoising |
| **Interface** | Same `action`, `realtime_action`, `bid_action` | Same (drop-in compatible) |
