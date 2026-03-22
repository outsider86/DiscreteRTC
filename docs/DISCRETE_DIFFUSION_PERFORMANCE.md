# Discrete Diffusion Policy — Performance Improvement Guide

This doc focuses on **model_dd.py**, **train_dd.py**, and **eval_dd.py**. Use it to systematically improve solve rates and robustness.

---

## 1. Pipeline summary

- **Model** (`model_dd.py`): MaskGIT-style discrete diffusion over action bins. Observation + (partially masked) action tokens → MLP-Mixer → logits over bins. Training: cross-entropy on masked positions only; inference: iterative unmasking with schedule + Gumbel top-k.
- **Training** (`train_dd.py`): Level-sharded data, AdamW + warmup, per-epoch eval at horizons 1..action_chunk_size.
- **Eval** (`eval_dd.py`): Naive, Realtime, BID, hard_masking; sweeps inference_delay and execute_horizon.

---

## 2. Model improvements (`model_dd.py`)

| Change | Where | Suggestion |
|--------|--------|------------|
| **Capacity** | `ModelConfig` | Increase `num_layers` (e.g. 6–8), `channel_dim` (e.g. 384), `channel_hidden_dim` (e.g. 768). Start with one at a time. |
| **Action resolution** | `num_bins` | 256 is standard. Try 128 (faster, less overfitting) or 512 (finer control, more capacity). |
| **RTC-style training** | `simulated_delay` | Set to True to never mask the first d steps (d random in 1..chunk_size//2); helps realtime_action at delay > 0. |
 Set to a small value (e.g. 1–2) so some batches train with a “prefix” of known actions; helps realtime_action at delay > 0. |
| **Train mask schedule** | `train_mask_schedule` | Default "cosine" is good; try "linear" for ablations. |
| **No-mask token prob** | `no_mask_token_prob` | Small value (e.g. 0.05–0.1) can reduce overfitting to always-masked; ref uses this. |

---

## 3. Decode / inference improvements

| Change | Where | Suggestion |
|--------|--------|------------|
| **Decode steps** | `eval_dd.EvalConfig.num_flow_steps`, `train_dd.eval_num_flow_steps` | Default 5; try 8–12 for better quality (more iterations). Trade off with latency. |
| **Decode schedule** | `ModelConfig.decode_schedule` | "cosine" vs "linear"; usually cosine is better. |
| **Temperature** | `ModelConfig.choice_temperature` | Lower → more deterministic; try 0.5–1.0. |
| **Remask** | `ModelConfig.use_remask` | True can improve quality by re-masking low-confidence positions; try both. |

---

## 4. Training improvements (`train_dd.py`)

| Change | Where | Suggestion |
|--------|--------|------------|
| **LR schedule** | `Config` + optimizer | Add cosine decay after warmup instead of constant LR (see below). |
| **Warmup** | `lr_warmup_steps` | Increase to 2000–5000 for larger batches or more epochs. |
| **Batch size** | `batch_size` | 512 is solid; try 1024 if memory allows (often more stable). |
| **Epochs** | `num_epochs` | Try 48–64 if not overfitting. |
| **Weight decay** | `weight_decay` | 1e-2 is common; try 1e-3 or 5e-2 for ablations. |
| **Grad clip** | `grad_norm_clip` | 10.0 is reasonable; avoid clipping too aggressively. |
| **Eval cost** | Per-epoch eval | Eval at fewer horizons (e.g. 1, 4, 8) or every N epochs to save time. |

---

## 5. Evaluation improvements (`eval_dd.py`)

- **Methods**: You already compare naive, realtime, BID, hard_masking. For low latency, focus on realtime vs naive at various (inference_delay, execute_horizon).
- **Seeds**: Run multiple seeds (e.g. seed=0,1,2) and report mean ± std for returned_episode_solved.
- **num_flow_steps**: Sweep 5, 8, 12 at eval to see quality–latency tradeoff.

---

## 6. Quick wins to try first

1. **Enable RTC-style training**: Set `simulated_delay=True` in `ModelConfig` (or via `train_dd.Config.eval_model`).
2. **More decode steps at eval**: Set `num_flow_steps=8` or `10` in `EvalConfig` and in training’s `eval_num_flow_steps`.
3. **Cosine LR decay**: Replace constant LR after warmup with cosine decay to 1e-5 or 1e-6 over the rest of training.
4. **Slightly larger model**: e.g. `num_layers=6`, `channel_dim=320` without changing data pipeline.

---

## 7. Suggested experiment order

1. Baseline: current config, log solve rate and loss.
2. Add **cosine LR decay**; re-run same epochs.
3. Add **simulated_delay=True**; compare realtime vs naive at delay 1–2.
4. Increase **num_flow_steps** at eval to 8; compare solve rate vs latency.
5. Ablate **use_remask** True vs False at eval.
6. Scale model (**num_layers** / **channel_dim**) if still underfitting.

### 8. CLI examples (train_dd.py)

- **Cosine LR decay**:  
  `python -m train_dd --run-path ... --use-cosine-decay --lr-min 1e-5`
- **RTC-style training** (prefix masking):  
  Pass a model config with `simulated_delay=True`; e.g. set `eval_model.simulated_delay true` via CLI if supported.
- **More decode steps during training eval**:  
  `--eval-num-flow-steps 8`
