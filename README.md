<div align="center">

# DiscreteRTC — Real-World Experiments Infrastructure

**Built on [StarVLA](https://github.com/starVLA/starVLA)**

[![Project Page](https://img.shields.io/badge/Project-Page-2575fc.svg)](https://outsider86.github.io/DiscreteRTCSite/)
[![main branch](https://img.shields.io/badge/branch-main-6a11cb.svg)](https://github.com/outsider86/DiscreteRTC/tree/main)
[![kinetix branch](https://img.shields.io/badge/branch-kinetix-1a8c1a.svg)](https://github.com/outsider86/DiscreteRTC/tree/kinetix)
[![StarVLA](https://img.shields.io/badge/⭐_StarVLA-upstream-orange.svg)](https://github.com/starVLA/starVLA)

</div>

This branch documents the **real-world** side of DiscreteRTC (UR5e dynamic manipulation: Dynamic Pick, Dynamic Place, Hockey Defend). For the simulation pipeline see the [`kinetix`](https://github.com/outsider86/DiscreteRTC/tree/kinetix) branch; for the project overview see [`main`](https://github.com/outsider86/DiscreteRTC/tree/main).

---

## 1. Built upon StarVLA

The real-world DiscreteRTC policy is implemented as a **VLA framework on top of [StarVLA](https://github.com/starVLA/starVLA)** — a modular ("Lego-like") codebase where the VLM backbone, action head, dataloader, trainer, and deployment hooks are decoupled, so a new method typically reduces to swapping the action head while reusing the rest.

DiscreteRTC is upstreamed into StarVLA in **[PR #343 — `QwenDiscreteDiffusion` + RTC `predict_action_realtime`](https://github.com/starVLA/starVLA/pull/343)**. We actively maintain the method there. The relevant files in StarVLA:

| Component | Path (in StarVLA) |
|---|---|
| Framework | `starVLA/model/framework/VLM4A/QwenDiscreteDiffusion.py` |
| Action head | `starVLA/model/modules/action_model/LayerwiseDiscreteDiffusion_ActionHeader.py` |
| Binning + MaskGIT helpers | `starVLA/model/modules/action_model/discrete_diffusion/` |
| Sim config (RoboTwin, 14D) | `starVLA/config/training/starvla_train_discrete_diffusion.yaml` |
| Real config (FastUMI, 10D) | `starVLA/config/training/starvla_train_discrete_diffusion_real.yaml` |
| Extension docs | `examples/modelExtensions/DiscreteDiffusion/README.md` |

---

## 2. Architecture

`QwenDiscreteDiffusion` keeps the **same Qwen2.5-VL backbone** as StarVLA's flow-matching `QwenPI`, and the action head (`LayerwiseDiscreteDiffusionActionHead`) mirrors the QwenPI flow-matching head — same DiT block stack, same layer-wise cross-attention against the Qwen-VL hidden states, same future-token / state / position-embedding inputs. Two things change:

1. **Tokens, not noise.** Each continuous action `a ∈ [low, high]^D` is encoded into discrete bins (`num_bins=256`); the head predicts per-position logits over bins and decodes back to bin centers. An auxiliary L1 loss (`l1_loss_weight=0.1`) keeps bin centers calibrated. Both a `bin` (softmax over bins, cross-entropy) and a `bit` (8 sigmoid bits, BCE) representation are supported.
2. **MaskGIT decode replaces Euler integration.** Training masks a schedule-controlled fraction of positions and reconstructs them; inference iteratively unmasks the highest-confidence positions over `num_inference_steps` rounds.

**Real-time chunking (RTC) decode** — `predict_action_realtime(...)` is the deployment entry point. The previous chunk's un-executed tail is encoded into bins and **pinned as a known prefix**; only the trailing `execution_horizon` positions are masked and resampled. Decode steps scale with the masked fraction (or are held fixed via `fixed_steps=True`), and `early_stop=True` exits as soon as every non-prefix slot is unmasked. The flow-matching head exposes the same `prev_action_chunk` / `inference_delay` / `execution_horizon` interface (using ΠGDM), so the two heads are interchangeable from the deployment client's perspective.

> For the full architecture write-up, see [`examples/modelExtensions/DiscreteDiffusion/README.md`](https://github.com/starVLA/starVLA/blob/main/examples/modelExtensions/DiscreteDiffusion/README.md) in StarVLA.

---

## 3. Training scripts (real-world datasets)

Real-world training uses the FastUMI pick-and-place dataset (10D single-arm actions: `[dx, dy, dz, rot6d(6), gripper]`) with the `starvla_train_discrete_diffusion_real.yaml` config:

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_discrete_diffusion_real.yaml \
  --framework.name QwenDiscreteDiffusion \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix fastumi_pickandplace_real_0307 \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --run_id qwendd_fastumi_real
```

The sim/RoboTwin counterpart (14D dual-arm) swaps in `starvla_train_discrete_diffusion.yaml`. Any YAML key can be overridden from the CLI with `--<group>.<subgroup>.<key> <value>`. See the StarVLA extension README for the full set of options.

---

## 4. TODO

- [ ] **Benchmark results:** provide VLM + discrete-diffusion policy-head results on StarVLA-supported benchmarks (LIBERO, SimplerEnv, RoboTwin 2.0, RoboCasa, DOMINO, BEHAVIOR, Calvin, …).
