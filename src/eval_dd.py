"""Evaluation script for Discrete Diffusion policies. Mirrors eval_flow.py with multi-GPU sharding."""

import os
if "EVAL_DD_GPU_ID" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["EVAL_DD_GPU_ID"]

import dataclasses
import functools
import json
import math
import pathlib
import pickle
import subprocess
import sys
import tempfile
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv                       # type: ignore
import kinetix.environment.env_state as kenv_state           # type: ignore
import kinetix.environment.wrappers as wrappers              # type: ignore
import kinetix.render.renderer_pixels as renderer_pixels     # type: ignore
import pandas as pd
import tyro

import model_dd as _model_dd
import train_expert
import compute_robot_indices


@dataclasses.dataclass(frozen=True)
class NaiveMethodConfig:
    pass


@dataclasses.dataclass(frozen=True)
class RealtimeMethodConfig:
    early_stop: bool = False
    adaptive_unmasking: bool = False


@dataclasses.dataclass(frozen=True)
class DiscreteRTCConfig:
    early_stop: bool = True
    adaptive_unmasking: bool = False


@dataclasses.dataclass(frozen=True)
class AdaptiveDiscreteRTCConfig:
    """Discrete RTC with adaptive_unmasking=True (fewer steps when fewer tokens unknown)."""
    early_stop: bool = True
    adaptive_unmasking: bool = True


@dataclasses.dataclass(frozen=True)
class BIDMethodConfig:
    n_samples: int = 16
    bid_k: int | None = None


@dataclasses.dataclass(frozen=True)
class VlashMethodConfig:
    """VLASH-style future-state conditioning for discrete diffusion (no extra noise modelling)."""
    pass


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    step: int = -1
    weak_step: int | None = None
    num_evals: int = 2048
    num_flow_steps: int = 5

    inference_delay: int = 0
    execute_horizon: int = 1
    method: NaiveMethodConfig | RealtimeMethodConfig | DiscreteRTCConfig | AdaptiveDiscreteRTCConfig | BIDMethodConfig | VlashMethodConfig = NaiveMethodConfig()

    # Eval-time decode temperature (not a model parameter); used when sampling which positions to unmask (if deterministic_choice=False).
    # 0.1 gives a good balance; temp=0 implicitly sets deterministic_choice=True.
    choice_temperature: float = 0.0
    # Temperature for token probs during decode: probs = softmax(logits / decode_temperature). inf → uniform, 0 → argmax; 1.0 = standard.
    # temp=0 implicitly sets deterministic_decode=True.
    decode_temperature: float = 1.0

    model: _model_dd.ModelConfig = _model_dd.ModelConfig()


def _eval_config_to_dict(
    config: EvalConfig,
    *,
    run_path: str,
    level_paths: list[str],
    seed: int,
    output_dir: str | None,
) -> dict:
    """Convert EvalConfig and run params to a JSON-serializable dict."""
    d = dataclasses.asdict(config)
    # Add method type name for clarity
    m = config.method
    if isinstance(m, NaiveMethodConfig):
        d["method"] = {"type": "naive", **dataclasses.asdict(m)}
    elif isinstance(m, RealtimeMethodConfig):
        d["method"] = {"type": "realtime", **dataclasses.asdict(m)}
    elif isinstance(m, DiscreteRTCConfig):
        d["method"] = {"type": "discrete_rtc", **dataclasses.asdict(m)}
    elif isinstance(m, AdaptiveDiscreteRTCConfig):
        d["method"] = {"type": "adaptive_discrete_rtc", **dataclasses.asdict(m)}
    elif isinstance(m, BIDMethodConfig):
        d["method"] = {"type": "bid", **dataclasses.asdict(m)}
    else:
        d["method"] = dataclasses.asdict(m)
    d["run_path"] = run_path
    d["level_paths"] = level_paths
    d["seed"] = seed
    d["output_dir"] = output_dir
    return d


def eval(
    config: EvalConfig,
    env: kenv.environment.Environment,
    rng: jax.Array,
    level: kenv_state.EnvState,
    policy: _model_dd.DiscreteDiffusionPolicy,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.EnvParams,
    weak_policy: _model_dd.DiscreteDiffusionPolicy | None = None,
    robot_mask: jax.Array | None = None,
):
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))), config.num_evals
    )
    render_video = train_expert.make_render_video(renderer_pixels.make_render_pixels(env_params, static_env_params))
    assert config.execute_horizon >= config.inference_delay, f"{config.execute_horizon=} {config.inference_delay=}"

    def execute_chunk(carry, _):
        def step(carry, action):
            rng, obs, env_state = carry
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
            return (rng, next_obs, next_env_state), (done, env_state, info)

        rng, obs, env_state, action_chunk, n = carry
        rng, key = jax.random.split(rng)
        ct, dt = config.choice_temperature, config.decode_temperature
        if isinstance(config.method, NaiveMethodConfig):
            next_action_chunk = policy.action(
                key, obs, config.num_flow_steps, choice_temperature=ct, decode_temperature=dt
            )
        elif isinstance(config.method, (RealtimeMethodConfig, DiscreteRTCConfig, AdaptiveDiscreteRTCConfig)):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            assert (
                config.inference_delay <= policy.action_chunk_size
                and prefix_attention_horizon <= policy.action_chunk_size
            ), f"{config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            next_action_chunk = policy.realtime_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                config.execute_horizon,
                config.method.early_stop,
                adaptive_unmasking=config.method.adaptive_unmasking,
                choice_temperature=ct,
                decode_temperature=dt,
            )
        elif isinstance(config.method, BIDMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            if config.method.bid_k is not None:
                assert weak_policy is not None, "weak_policy is required for BID"
            next_action_chunk = policy.bid_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.n_samples,
                bid_k=config.method.bid_k,
                bid_weak_policy=weak_policy if config.method.bid_k is not None else None,
                choice_temperature=ct,
                decode_temperature=dt,
            )
        elif isinstance(config.method, VlashMethodConfig):
            # VLASH: simulate future state for delayed actions, then act on mixed future obs.
            # Use the same RNG stream for simulation as for the first inference_delay env steps
            # so that NoisyActionWrapper noise matches and simulated future == actual future.
            assert robot_mask is not None, "robot_mask is required for VlashMethodConfig"

            if config.inference_delay == 0:
                # Mode 1: no simulation
                obs_for_policy = obs
            elif config.inference_delay == 1:
                # Mode 2: simulate 1 step (no inner scan)
                step_keys = jax.random.split(rng, config.num_evals + 1)[: config.num_evals]  # (num_evals, 2)
                actions_0 = action_chunk[:, 0, :]  # (num_evals, action_dim)

                def one_step(key, state, action):
                    obs_next, next_state, _, _, _ = env._env.step(key, state, action, env_params)
                    return obs_next, next_state

                obs_future, _ = jax.vmap(one_step)(step_keys, env_state, actions_0)
                obs_for_policy = jnp.where(robot_mask, obs_future, obs)
            else:
                # Mode 3: inference_delay > 1 — simulate d steps with a scan (same pattern as eval_vlash_dd.py)
                d = config.inference_delay
                # One RNG per env; inside scan we carry RNG and split each step (no pre-built key array)
                rngs_sim = jax.random.split(rng, config.num_evals)  # (num_evals, 2)

                @jax.vmap
                def sim_steps(rng_sim, state, actions_d):
                    def step(carry, action):
                        s, r = carry
                        r, key = jax.random.split(r)
                        obs_n, s_n, _, _, _ = env._env.step(key, s, action, env_params)
                        return (s_n, r), obs_n

                    (_, _), obs_seq = jax.lax.scan(step, (state, rng_sim), actions_d)
                    return obs_seq[-1]

                obs_future = sim_steps(rngs_sim, env_state, action_chunk[:, :d, :])
                obs_for_policy = jnp.where(robot_mask, obs_future, obs)
            next_action_chunk = policy.action(
                key,
                obs_for_policy,
                config.num_flow_steps,
                choice_temperature=ct,
                decode_temperature=dt,
            )
        else:
            raise ValueError(f"Unknown method: {config.method}")

        # Build action chunk to execute and shift for next iteration
        if isinstance(config.method, VlashMethodConfig):
            # VLASH: next_action_chunk generated with simulated future obs
            shift = config.execute_horizon - config.inference_delay
            action_chunk_to_execute = jnp.concatenate(
                [action_chunk[:, : config.inference_delay], next_action_chunk[:, :shift]],
                axis=1,
            )
            next_action_chunk = jnp.concatenate(
                [
                    next_action_chunk[:, shift:],
                    jnp.zeros((obs.shape[0], shift, policy.action_dim)),
                ],
                axis=1,
            )
            # Advance by execute_horizon steps (we ran that many env steps), not shift
            next_n = jnp.concatenate(
                [n[config.execute_horizon :], jnp.zeros(config.execute_horizon, dtype=jnp.int32)]
            )
        else:
            action_chunk_to_execute = jnp.concatenate(
                [
                    action_chunk[:, : config.inference_delay],
                    next_action_chunk[:, config.inference_delay : config.execute_horizon],
                ],
                axis=1,
            )
            next_action_chunk = jnp.concatenate(
                [
                    next_action_chunk[:, config.execute_horizon :],
                    jnp.zeros((obs.shape[0], config.execute_horizon, policy.action_dim)),
                ],
                axis=1,
            )
            next_n = jnp.concatenate([n[config.execute_horizon :], jnp.zeros(config.execute_horizon, dtype=jnp.int32)])
        (rng, next_obs, next_env_state), (dones, env_states, infos) = jax.lax.scan(
            step, (rng, obs, env_state), action_chunk_to_execute.transpose(1, 0, 2)
        )
        return (rng, next_obs, next_env_state, next_action_chunk, next_n), (dones, env_states, infos)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    action_chunk = policy.action(
        key,
        obs,
        config.num_flow_steps,
        choice_temperature=config.choice_temperature,
        decode_temperature=config.decode_temperature,
    )
    n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)
    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    _, (dones, env_states, infos) = jax.lax.scan(
        execute_chunk,
        (rng, obs, env_state, action_chunk, n),
        None,
        length=scan_length,
    )
    dones, env_states, infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (dones, env_states, infos))
    assert dones.shape[0] >= env_params.max_timesteps, f"{dones.shape=}"
    return_info = {}
    for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        first_done_idx = jnp.argmax(dones, axis=0)
        return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()
    for key in ["match"]:
        if key in infos:
            return_info[key] = jnp.mean(infos[key])
    # Average task completion time (seconds): mean episode length * step duration
    step_duration_s = float(env_params.dt) * int(static_env_params.frame_skip)
    return_info["mean_task_completion_time_s"] = return_info["returned_episode_lengths"] * step_duration_s
    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))
    return return_info, video


def _append_row_to_csv(path: pathlib.Path, row: dict, write_header: bool) -> None:
    """Append a single row to CSV on the fly. write_header=True for first row."""
    mode = "w" if write_header else "a"
    pd.DataFrame([row]).to_csv(path, mode=mode, header=write_header, index=False)


def run_eval_chunk(
    run_path: str,
    config: EvalConfig,
    level_paths: Sequence[str],
    seed: int,
    results_path: pathlib.Path | None = None,
) -> list[dict]:
    """Run full eval sweep for a subset of levels; returns list of row dicts (one per level × delay × method × horizon)."""
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    state_dicts = []
    weak_state_dicts = []
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(run_path).iterdir()))
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        with (log_dirs[config.step] / "policies" / f"{level_name}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
        if config.weak_step is not None:
            with (log_dirs[config.weak_step] / "policies" / f"{level_name}.pkl").open("rb") as f:
                weak_state_dicts.append(pickle.load(f))
    if config.weak_step is None:
        weak_state_dicts = None

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[
        0
    ].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    # Robot masks for VLASH
    robot_masks = jnp.stack(
        [compute_robot_indices.compute_robot_mask(p, obs_dim) for p in level_paths]
    )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _eval_single(
        config: EvalConfig,
        rng: jax.Array,
        level: kenv_state.EnvState,
        state_dict: dict,
        weak_state_dict: dict | None,
        robot_mask: jax.Array,
    ):
        policy = _model_dd.DiscreteDiffusionPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config.model,
            rngs=nnx.Rngs(rng),
        )
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        policy = nnx.merge(graphdef, state)
        if weak_state_dict is not None:
            graphdef, state = nnx.split(policy)
            state.replace_by_pure_dict(weak_state_dict)
            weak_policy = nnx.merge(graphdef, state)
        else:
            weak_policy = None
        eval_info, _ = eval(
            config,
            env,
            rng,
            level,
            policy,
            env_params,
            static_env_params,
            weak_policy,
            robot_mask,
        )
        return eval_info

    rngs = jax.random.split(jax.random.key(seed), len(level_paths))

    def run_config(c: EvalConfig) -> list[dict]:
        out_list = []
        for i in range(len(level_paths)):
            level_i = jax.tree.map(lambda x: x[i], levels)
            w = weak_state_dicts[i] if weak_state_dicts is not None else None
            mask_i = robot_masks[i]
            info = _eval_single(c, rngs[i], level_i, state_dicts[i], w, mask_i)
            out_list.append(jax.device_get(info))
        return out_list

    rows: list[dict] = []
    for inference_delay in [0, 1, 2, 3, 4]:
        execute_horizon = max(1, inference_delay)
        print(f"{inference_delay=} {execute_horizon=}")

        # Naive
        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=NaiveMethodConfig()
        )
        out_list = run_config(c)
        for i in range(len(level_paths)):
            row = {"delay": inference_delay, "method": "naive", "level": level_paths[i], "execute_horizon": execute_horizon, **out_list[i]}
            rows.append(row)
            if results_path is not None:
                _append_row_to_csv(results_path, row, write_header=(len(rows) == 1))

        # Discrete RTC
        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=DiscreteRTCConfig()
        )
        out_list = run_config(c)
        for i in range(len(level_paths)):
            row = {"delay": inference_delay, "method": "discrete_rtc", "level": level_paths[i], "execute_horizon": execute_horizon, **out_list[i]}
            rows.append(row)
            if results_path is not None:
                _append_row_to_csv(results_path, row, write_header=(len(rows) == 1))

        # Adaptive discrete RTC (adaptive_unmasking=True)
        c = dataclasses.replace(
            config,
            inference_delay=inference_delay,
            execute_horizon=execute_horizon,
            method=AdaptiveDiscreteRTCConfig(),
        )
        out_list = run_config(c)
        for i in range(len(level_paths)):
            row = {
                "delay": inference_delay,
                "method": "adaptive_discrete_rtc",
                "level": level_paths[i],
                "execute_horizon": execute_horizon,
                **out_list[i],
            }
            rows.append(row)
            if results_path is not None:
                _append_row_to_csv(results_path, row, write_header=(len(rows) == 1))

        # BID
        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=BIDMethodConfig()
        )
        out_list = run_config(c)
        for i in range(len(level_paths)):
            row = {"delay": inference_delay, "method": "bid", "level": level_paths[i], "execute_horizon": execute_horizon, **out_list[i]}
            rows.append(row)
            if results_path is not None:
                _append_row_to_csv(results_path, row, write_header=(len(rows) == 1))

        # VLASH (future-state-aware chunk generation)
        c = dataclasses.replace(
            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=VlashMethodConfig()
        )
        out_list = run_config(c)
        for i in range(len(level_paths)):
            row = {
                "delay": inference_delay,
                "method": "vlash",
                "level": level_paths[i],
                "execute_horizon": execute_horizon,
                **out_list[i],
            }
            rows.append(row)
            if results_path is not None:
                _append_row_to_csv(results_path, row, write_header=(len(rows) == 1))

        jax.clear_caches()
    return rows


def main(
    run_path: str,
    config: EvalConfig = EvalConfig(),
    level_paths: Sequence[str] = (
        "worlds/l/grasp_easy.json",
        "worlds/l/catapult.json",
        "worlds/l/cartpole_thrust.json",
        "worlds/l/hard_lunar_lander.json",
        "worlds/l/mjc_half_cheetah.json",
        "worlds/l/mjc_swimmer.json",
        "worlds/l/mjc_walker.json",
        "worlds/l/h17_unicycle.json",
        "worlds/l/chain_lander.json",
        "worlds/l/catcher_v3.json",
        "worlds/l/trampoline.json",
        "worlds/l/car_launch.json",
    ),
    seed: int = 0,
    output_dir: str | None = "eval_logs_paper/stats",
    num_gpus: int = 0,
):
    """Run evaluation. Use num_gpus > 1 to distribute levels across GPUs via separate processes (avoids NCCL)."""
    level_paths = list(level_paths)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_dict = _eval_config_to_dict(
        config, run_path=run_path, level_paths=level_paths, seed=seed, output_dir=output_dir
    )
    with (output_path / "eval_config.json").open("w") as f:
        json.dump(config_dict, f, indent=2)

    if num_gpus <= 1:
        results_path = output_path / "results.csv"
        rows = run_eval_chunk(run_path, config, level_paths, seed, results_path=results_path)
    else:
        num_gpus = min(num_gpus, len(level_paths))
        print(f"Using {num_gpus} GPUs (one process per GPU)")
        chunk_size = (len(level_paths) + num_gpus - 1) // num_gpus
        chunks = [level_paths[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus)]
        chunks = [c for c in chunks if c]
        project_root = pathlib.Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "eval_dd.py"
        all_rows: list[dict] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            procs = []
            out_paths = []
            for i, chunk in enumerate(chunks):
                in_path = pathlib.Path(tmpdir) / f"in_{i}.pkl"
                out_paths.append(pathlib.Path(tmpdir) / f"out_{i}.pkl")
                worker_results_path = output_path / f"results_worker{i}.csv"
                with in_path.open("wb") as f:
                    pickle.dump((run_path, config, chunk, seed + i, str(worker_results_path)), f)
                env = {**os.environ, "EVAL_DD_GPU_ID": str(i)}
                procs.append(
                    subprocess.Popen(
                        [sys.executable, str(script_path), "worker", str(in_path), str(out_paths[-1])],
                        cwd=project_root,
                        env=env,
                    )
                )
            for p in procs:
                p.wait()
                if p.returncode != 0:
                    raise RuntimeError(f"Worker process exited with code {p.returncode}")
            for out_path in out_paths:
                with out_path.open("rb") as f:
                    all_rows.extend(pickle.load(f))
        rows = all_rows
        df = pd.DataFrame(rows)
        df.to_csv(output_path / "results.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "worker":
        in_path, out_path = sys.argv[2], sys.argv[3]
        with open(in_path, "rb") as f:
            run_path, config, level_paths, seed, results_path_str = pickle.load(f)
        results_path = pathlib.Path(results_path_str) if results_path_str else None
        rows = run_eval_chunk(run_path, config, level_paths, seed, results_path=results_path)
        with open(out_path, "wb") as f:
            pickle.dump(rows, f)
    else:
        tyro.cli(main)
