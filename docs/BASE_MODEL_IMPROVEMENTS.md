# Improving Base Discrete Diffusion Policy (No Inpainting)

Focused improvements for the **base model** trained from scratch with `inpainting_mask=False`. Use this when you are not using RTC-style finetuning.

---

## 1. Quick wins (try first)

| Change | Where | Current | Suggested | Why |
|--------|--------|---------|-----------|-----|
| **Decode steps** | `train_dd.Config.eval_num_flow_steps`, `eval_dd.EvalConfig.num_flow_steps` | 5 | **8** or **10** | More unmasking steps → better action quality; small latency cost. |
| **No-mask token prob** | `model_dd.ModelConfig.no_mask_token_prob` | 0.0 | Keep 0.0 | Non-zero (e.g. 0.05) often worsens results; only try if you see clear overfitting. “always predict masked”; |
| **Remask at inference** | `model_dd.ModelConfig.use_remask` | False | Keep False | Re-masking low-confidence positions during decode often hurts; keep off unless you ablate. |

---

## 2. Model capacity (if underfitting)

If training loss is still high or eval success plateaus early:

| Parameter | Current | Try |
|-----------|---------|-----|
| `num_layers` | 4 | **6** |
| `channel_dim` | 256 | **320** or **384** |
| `channel_hidden_dim` | 512 | **768** |

Change one at a time and compare. Larger models need slightly longer warmup.

---

## 3. Action resolution

| Parameter | Current | Notes |
|-----------|---------|--------|
| `num_bins` | 512 | Good default. **256** = faster, sometimes more robust; **512** = finer control. Try 256 if you see overfitting or slow training. |

### Soft labels (encourage nearby bins)

If the ground-truth bin is 100, predicting 99 or 101 should get partial credit. Set **`soft_label_sigma`** &gt; 0 in `ModelConfig` to use a Gaussian over bins instead of one-hot targets:

- **`soft_label_sigma=0`** (default): one-hot targets, only the exact bin is correct.
- **`soft_label_sigma=1.0`** or **1.5**: target distribution is a Gaussian in bin space; neighboring bins get some mass, so the loss encourages "close" predictions (helps smoothness and precision-sensitive tasks like catapult/MJC).

Example: `--config.eval-model.soft-label-sigma 1.0` when training.

---

## 4. Training schedule

| Parameter | Current | Suggestion |
|-----------|---------|------------|
| `num_epochs` | 32 | **48** or **64** if validation/eval success still improving at epoch 32. |
| `batch_size` | 512 | **1024** if GPU memory allows (often more stable). |
| `lr_warmup_steps` | 2000 | Keep or increase to **3000–5000** if you use a larger model or batch. |
| `weight_decay` | 1e-2 | Try **1e-3** if underfitting, **5e-2** if overfitting. |

You already use cosine decay to `lr_min=1e-5`; that’s a good default.

---

## 5. Inference (eval only)

- **Temperature:** `choice_temperature=1.0` is default. Try **0.5–0.8** for more deterministic (sometimes better success), **1.0–1.2** for more diverse rollouts.
- **Decode schedule:** Keep `decode_schedule="cosine"`.
- **Seeds:** Run eval with 2–3 seeds and report mean ± std of solve rate.

---

## 6. Suggested order of experiments

1. **Baseline:** Current config; log train loss and eval solve rate (e.g. naive, delay=0).
2. **More decode steps:** Set `eval_num_flow_steps=8` in both training and eval; re-run and compare.
3. **no_mask_token_prob:** Set `no_mask_token_prob=0.05` in `ModelConfig`; train from scratch and compare.
4. **use_remask:** Keep `False`; trying `True` often worsens results.
5. If still underfitting: **capacity** (e.g. `num_layers=6` or `channel_dim=320`).
6. If overfitting: **weight_decay** up or **num_bins=256**.

---

## 7. CLI examples (base model only)

**Train with improved defaults (more decode steps + no_mask_token_prob):**
```bash
uv run src/train_dd.py --config.run-path ./logs-expert/<run> \
  --config.eval-num-flow-steps 8 \
  --config.eval-model.no-mask-token-prob 0.05
```

**Eval with more steps and remask:**
```bash
uv run src/eval_dd.py --run-path logs-dd/<run> --output-dir eval_base_improved \
  --config.num-flow-steps 8 \
  --config.model.use-remask true
```

(If your CLI exposes `eval_model` / `model` as nested config, use the flags above; otherwise set these in code or config JSON.)
