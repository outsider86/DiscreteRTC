# Why WandB success rate can differ from eval_dd results

Training logs **in-epoch evaluation** to WandB; the standalone script `eval_dd.py` runs a **sweep** with different settings. Comparing them only makes sense when the **setting matches**.

## What training logs (train_dd.py)

- **Per level**, for each epoch:
  - `returned_episode_solved_1`, `returned_episode_solved_2`, … `returned_episode_solved_8`
- These correspond to **execute_horizon** = 1, 2, …, 8:
  - **_1**: replan every step (hardest, most comparable to “realtime”).
  - **_8**: use a full 8-step chunk before replanning (easier, policy commits longer).
- **Method**: always **naive** (no prefix, no realtime).
- **inference_delay**: 0.
- **num_evals**: `eval_num_evals` (e.g. 128).

So **~40% in WandB** might be e.g. **returned_episode_solved_8** (easier), while your eval run might be looking at **delay=0, execute_horizon=1** (harder) → lower %.

## What eval_dd.py reports

- Sweep over **inference_delay** (0–4), **execute_horizon** = max(1, delay), and **method** (naive, realtime, discrete_rtc, bid).
- For **delay=0**, it uses **execute_horizon=1** (replan every step).
- **num_evals**: default 2048 (can be 256 etc. via config).

So the “main” naive curve in eval is **delay=0, execute_horizon=1**, i.e. the **same regime as WandB’s returned_episode_solved_1**, not _8.

## How to compare

1. **Apples to apples**: In WandB, use **returned_episode_solved_1** (and, if you care, per-level) and compare to **eval_dd**’s **naive, delay=0, execute_horizon=1** (and same level or same level-average).
2. **Why _8 is higher**: With execute_horizon=8 the policy runs a full chunk before replanning, so behavior is smoother and success rate is often higher than horizon=1.
3. **num_evals**: Training uses 128 by default; eval_dd often uses 2048. Slight differences in reported % are normal (variance).

## Summary

You’re not misreading the metric. The difference comes from **which horizon** (and possibly which level/method) you look at. For a fair comparison to eval_dd’s default naive setting, use **WandB’s returned_episode_solved_1**.
