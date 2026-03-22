# Why MJC Tasks and Catapult Fail More Often

Analysis of why discrete diffusion policies tend to fail on **MJC tasks** (half_cheetah, swimmer, walker) and **catapult** more than on other levels, and what to try.

---

## 1. MJC tasks (half_cheetah, swimmer, walker)

### Task characteristics

- **Articulated locomotion**: 4 motor bindings, 6 joints, 256 steps. Success = sustained, coordinated motion (e.g. run forward, swim, walk).
- **Sensitivity**: Small action errors compound over time; gaits are stable only when torques are smooth and temporally consistent.

### Likely reasons policies fail

| Factor | Explanation |
|--------|-------------|
| **Action binning** | Actions are discretized to 512 bins. Bin centers are used at execution; small but **systematic** rounding error every step can cause drift, phase error, or instability in dynamic gaits. |
| **Chunk commitment** | Fixed `action_chunk_size=8` means the policy commits to 8 steps at once. Locomotion often needs **reactive**, smooth adjustments; committing too far ahead can lock in a bad trajectory. |
| **Temporal coherence** | Chunk boundaries can show small discontinuities (e.g. last step of chunk vs first step of next). For locomotion, these can trigger falls or loss of balance. |
| **Imitation / mode averaging** | Expert data may contain multiple successful gaits or phases. Discrete diffusion may average or pick a “middle” that is not a valid gait. |

### Mitigations to try

1. **More decode steps**  
   Already recommended globally; helps action quality and smoothness (e.g. `num_flow_steps=10` or 12 for MJC).

2. **Larger model for MJC**  
   If MJC is underfitting, try level-specific or global increase: `num_layers=6`, `channel_dim=320` so the model can represent more precise, coordinated actions.

3. **Action chunk size**  
   Experiment with **smaller** `action_chunk_size` (e.g. 4) for MJC-only training or eval: shorter commitment may help reactivity (at the cost of more inference calls).

4. **Smoothing at chunk boundaries**  
   Optionally blend last step of previous chunk with first step of new chunk (e.g. linear interpolation) to reduce discontinuity; needs a small change in `eval_dd` execute loop.

5. **More expert data for MJC**  
   If data is limited, generate more MJC rollouts so the policy sees more diverse successful trajectories.

6. **Compare with flow policy**  
   Run the same levels with the flow policy (`eval_flow`). If flow works much better on MJC, the gap is likely due to **discretization + chunking**; if flow also struggles, the bottleneck may be expert quality or data coverage.

---

## 2. Catapult

### Task characteristics

- **Single critical action**: One main joint (arm) controls launch; success = ball (circle) reaches target. Likely **sparse** or **tight** success condition.
- **Precision**: The “correct” release angle/timing lies in a narrow band; small error → miss.

### Likely reasons policies fail

| Factor | Explanation |
|--------|-------------|
| **Bin resolution** | 512 bins over [-1, 1] → step ~0.004. One **wrong bin** at the critical moment can be enough to miss the target. |
| **Timing alignment** | The good action may only be correct for a small window of states. With chunking, that action might be applied at the **wrong** step (e.g. one step late/early). |
| **Sparse reward** | Many trajectories get no success signal; BC/diffusion gets weak gradient for “almost right” and may not concentrate mass on the narrow success region. |
| **Stochastic decode** | Categorical sampling during decode adds variance; for a one-shot style task, variance directly hurts success rate. |

### Mitigations to try

1. **Deterministic decode at eval**  
   For catapult (or in general), try **argmax** instead of categorical sample when taking the final action from logits (one-off change in `action` / `realtime_action` for eval only). Reduces variance; may help on precision-sensitive levels.

2. **More decode steps**  
   Again, more steps → better convergence of iterative unmasking → more precise actions.

3. **Lower temperature**  
   `choice_temperature` &lt; 1 (e.g. 0.5) makes decode more deterministic; can help on precision-critical tasks.

4. **Check expert solve rate**  
   If experts barely solve catapult (e.g. &lt; 70%), data may be noisy or biased; consider more expert training or higher solve-rate threshold for data generation for that level.

5. **Execute horizon**  
   Try **execute_horizon=1** for catapult so the policy can “correct” every step if your eval already uses that; if you use larger execute_horizon, try reducing it for this level to improve timing.

---

## 3. Summary

| Level type | Main issue | First things to try |
|------------|------------|----------------------|
| **MJC** | Binning + chunking hurt smooth, reactive locomotion; possible underfitting | More decode steps; optionally smaller chunk size or boundary smoothing; compare with flow |
| **Catapult** | Precision and timing; sparse success; decode variance | More decode steps; lower temperature or argmax at eval; check expert quality |

Implementing **argmax at eval** and **boundary smoothing** would require small, localized code changes in `model_dd.py` and `eval_dd.py`; the rest are config or data changes.
