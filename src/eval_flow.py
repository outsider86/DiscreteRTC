import collections
import dataclasses
import functools
import logging
import math
import pathlib
import pickle
import time
from typing import Sequence

log = logging.getLogger(__name__)

import flax.nnx as nnx
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import kinetix.environment.env as kenv                       # type: ignore
import kinetix.environment.env_state as kenv_state           # type: ignore
import kinetix.environment.wrappers as wrappers              # type: ignore
import kinetix.render.renderer_pixels as renderer_pixels     # type: ignore
import pandas as pd
import tyro

import model as _model
import train_expert


@dataclasses.dataclass(frozen=True)
class NaiveMethodConfig:
    pass


@dataclasses.dataclass(frozen=True)
class RealtimeMethodConfig:
    prefix_attention_schedule: _model.PrefixAttentionSchedule = "exp"
    max_guidance_weight: float = 5.0


@dataclasses.dataclass(frozen=True)
class BIDMethodConfig:
    n_samples: int = 16
    bid_k: int | None = None


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    step: int = -1
    weak_step: int | None = None
    num_evals: int = 2048  # Lowered from 2048 for ~16GB VRAM; use --config.num-evals 2048 for paper reproducibility
    num_flow_steps: int = 5

    inference_delay: int = 0
    execute_horizon: int = 1
    method: NaiveMethodConfig | RealtimeMethodConfig | BIDMethodConfig = NaiveMethodConfig()

    model: _model.ModelConfig = _model.ModelConfig()


def _process_profile_events(
    t_start: float,
    t_end: float,
    events: list[tuple[float, int, str]],
    method_name: str,
    num_steps: int,
) -> None:
    """Process (timestamp, step_idx, phase) events and log prepare / forward / decode breakdown."""
    if not events:
        return
    events_sorted = sorted(events, key=lambda x: x[0])
    prepare_s = events_sorted[0][0] - t_start
    steps: dict[int, list[tuple[float, str]]] = {}
    for t, step_idx, phase in events_sorted:
        step_idx_int = int(step_idx)
        if step_idx_int >= 0:
            steps.setdefault(step_idx_int, []).append((t, phase))
    forward_times: list[float] = []
    decode_times: list[float] = []
    for step_idx in sorted(steps.keys()):
        step_events = sorted(steps[step_idx], key=lambda x: x[0])
        if len(step_events) >= 3:
            t_start_s, t_fwd, t_dec = step_events[0][0], step_events[1][0], step_events[2][0]
            forward_times.append(t_fwd - t_start_s)
            decode_times.append(t_dec - t_fwd)
    overall_s = t_end - t_start
    mean_fwd = sum(forward_times) / len(forward_times) if forward_times else 0.0
    mean_dec = sum(decode_times) / len(decode_times) if decode_times else 0.0
    log.info(
        "[inference breakdown] method=%s | overall=%.4fs | prepare=%.4fs | per-step forward=%.4fs | per-step decode=%.4fs | num_steps=%d (overall is from one profiled run; prepare/forward/decode are host inter-callback intervals, not device time—do not sum to overall)",
        method_name,
        overall_s,
        prepare_s,
        mean_fwd,
        mean_dec,
        num_steps,
    )


def _benchmark_inference_s(
    policy: _model.FlowPolicy,
    config: EvalConfig,
    obs_dim: int,
    action_dim: int,
    n_warmup: int = 3,
    n_timed: int = 20,
    profile_breakdown: bool = False,
) -> float:
    """Mean wall time (seconds) per action-chunk inference for the current config's method."""
    rng = jax.random.key(0)
    obs = jnp.zeros((config.num_evals, obs_dim))
    action_chunk = jnp.zeros((config.num_evals, policy.action_chunk_size, action_dim))
    prefix_h = policy.action_chunk_size - config.execute_horizon
    method_name = type(config.method).__name__

    def run_naive():
        k, _ = jax.random.split(rng)
        return policy.action(k, obs, config.num_flow_steps)

    def run_realtime():
        k, _ = jax.random.split(rng)
        return policy.realtime_action(
            k,
            obs,
            config.num_flow_steps,
            action_chunk,
            config.inference_delay,
            prefix_h,
            config.method.prefix_attention_schedule,
            config.method.max_guidance_weight,
        )

    def run_bid():
        k, _ = jax.random.split(rng)
        return policy.bid_action(
            k,
            obs,
            config.num_flow_steps,
            action_chunk,
            config.inference_delay,
            prefix_h,
            config.method.n_samples,
            bid_k=config.method.bid_k,
            bid_weak_policy=None,
        )

    if isinstance(config.method, NaiveMethodConfig):
        fn = run_naive
        fn_with_profile = lambda cb: policy.action(
            jax.random.key(0), obs, config.num_flow_steps, _profile_callback=cb
        )
    elif isinstance(config.method, RealtimeMethodConfig):
        fn = run_realtime
        fn_with_profile = lambda cb: policy.realtime_action(
            jax.random.key(0), obs, config.num_flow_steps,
            action_chunk, config.inference_delay, prefix_h,
            config.method.prefix_attention_schedule, config.method.max_guidance_weight,
            _profile_callback=cb,
        )
    elif isinstance(config.method, BIDMethodConfig):
        fn = run_bid
        fn_with_profile = None
    else:
        raise ValueError(type(config.method))
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    start = time.perf_counter()
    for _ in range(n_timed):
        jax.block_until_ready(fn())
    mean_s = (time.perf_counter() - start) / n_timed
    log.info(
        "Inference time: %.4f s per action chunk (num_evals=%d, method=%s, inference_delay=%d, execute_horizon=%d)",
        mean_s,
        config.num_evals,
        method_name,
        config.inference_delay,
        config.execute_horizon,
    )

    if profile_breakdown and fn_with_profile is not None:
        _profile_events: list[tuple[float, int, str]] = []

        def _collect(step_idx: int, phase: str) -> None:
            _profile_events.append((time.perf_counter(), int(step_idx), phase))

        t0 = time.perf_counter()
        out = fn_with_profile(_collect)
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        _process_profile_events(t0, t1, _profile_events, method_name, config.num_flow_steps)

    return mean_s


def eval(
    config: EvalConfig,
    env: kenv.environment.Environment,
    rng: jax.Array,
    level: kenv_state.EnvState,
    policy: _model.FlowPolicy,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.EnvParams,
    weak_policy: _model.FlowPolicy | None = None,
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
        if isinstance(config.method, NaiveMethodConfig):
            next_action_chunk = policy.action(key, obs, config.num_flow_steps)
        elif isinstance(config.method, RealtimeMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            assert (
                config.inference_delay <= policy.action_chunk_size
                and prefix_attention_horizon <= policy.action_chunk_size
            ), f"{config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            print(
                f"{config.execute_horizon=} {config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            )
            next_action_chunk = policy.realtime_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.prefix_attention_schedule,
                config.method.max_guidance_weight,
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
            )
        else:
            raise ValueError(f"Unknown method: {config.method}")

        # we execute `inference_delay` actions from the *previously generated* action chunk, and then the remaining
        # `execute_horizon - inference_delay` actions from the newly generated action chunk
        action_chunk_to_execute = jnp.concatenate(
            [
                action_chunk[:, : config.inference_delay],
                next_action_chunk[:, config.inference_delay : config.execute_horizon],
            ],
            axis=1,
        )
        # throw away the first `execute_horizon` actions from the newly generated action chunk, to align it with the
        # correct frame of reference for the next scan iteration
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
        # if config.inference_delay > 0:
        #     infos["match"] = jnp.mean(jnp.abs(fixed_prefix - action_chunk_to_execute))
        return (rng, next_obs, next_env_state, next_action_chunk, next_n), (dones, env_states, infos)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, config.num_flow_steps)  # [batch, horizon, action_dim]
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
        # only consider the first episode of each rollout
        first_done_idx = jnp.argmax(dones, axis=0)
        return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()
    for key in ["match"]:
        if key in infos:
            return_info[key] = jnp.mean(infos[key])
    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))
    return return_info, video


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
    output_dir: str | None = "eval_output",
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    # load policies from best checkpoints by solve rate
    state_dicts = []
    weak_state_dicts = []
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(run_path).iterdir()))
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        # load policy
        with (log_dirs[config.step] / "policies" / f"{level_name}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
        if config.weak_step is not None:
            with (log_dirs[config.weak_step] / "policies" / f"{level_name}.pkl").open("rb") as f:
                weak_state_dicts.append(pickle.load(f))
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))
    if config.weak_step is not None:
        weak_state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *weak_state_dicts))
    else:
        weak_state_dicts = None

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[
        0
    ].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    # Build a single policy for inference benchmarking (first level's state)
    benchmark_policy = _model.FlowPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config.model,
        rngs=nnx.Rngs(jax.random.key(0)),
    )
    graphdef, state = nnx.split(benchmark_policy)
    state.replace_by_pure_dict(jax.tree.map(lambda x: x[0], state_dicts))
    benchmark_policy = nnx.merge(graphdef, state)

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec, pspec), out_specs=pspec)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
    def _eval(config: EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict, weak_state_dict):
        policy = _model.FlowPolicy(
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
        eval_info, _ = eval(config, env, rng, level, policy, env_params, static_env_params, weak_policy)
        return eval_info

    rngs = jax.random.split(jax.random.key(seed), len(level_paths))
    results = collections.defaultdict(list)
    for inference_delay in [0, 1, 2, 3, 4]:
        execute_horizon_min = max(1, inference_delay)
        # execute_horizon_min = action_chunk_size - inference_delay
        # execute_horizon_max = action_chunk_size - inference_delay
        execute_horizon_max = execute_horizon_min
        for execute_horizon in range(execute_horizon_min, execute_horizon_max + 1):
            print(f"{inference_delay=} {execute_horizon=}")
            c = dataclasses.replace(
                config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=NaiveMethodConfig()
            )
            mean_inference_s = _benchmark_inference_s(benchmark_policy, c, obs_dim, action_dim)
            out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
            for i in range(len(level_paths)):
                for k, v in out.items():
                    results[k].append(v[i])
                results["delay"].append(inference_delay)
                results["method"].append("naive")
                results["level"].append(level_paths[i])
                results["execute_horizon"].append(execute_horizon)
                results["mean_inference_s"].append(mean_inference_s)

            c = dataclasses.replace(
                config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=RealtimeMethodConfig()
            )
            mean_inference_s = _benchmark_inference_s(benchmark_policy, c, obs_dim, action_dim)
            out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
            for i in range(len(level_paths)):
                for k, v in out.items():
                    results[k].append(v[i])
                results["delay"].append(inference_delay)
                results["method"].append("realtime")
                results["level"].append(level_paths[i])
                results["execute_horizon"].append(execute_horizon)
                results["mean_inference_s"].append(mean_inference_s)

            c = dataclasses.replace(
                config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=BIDMethodConfig()
            )
            mean_inference_s = _benchmark_inference_s(benchmark_policy, c, obs_dim, action_dim)
            out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
            for i in range(len(level_paths)):
                for k, v in out.items():
                    results[k].append(v[i])
                results["delay"].append(inference_delay)
                results["method"].append("bid")
                results["level"].append(level_paths[i])
                results["execute_horizon"].append(execute_horizon)
                results["mean_inference_s"].append(mean_inference_s)

            c = dataclasses.replace(
                config,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=RealtimeMethodConfig(prefix_attention_schedule="zeros"),
            )
            mean_inference_s = _benchmark_inference_s(benchmark_policy, c, obs_dim, action_dim)
            out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
            for i in range(len(level_paths)):
                for k, v in out.items():
                    results[k].append(v[i])
                results["delay"].append(inference_delay)
                results["method"].append("hard_masking")
                results["level"].append(level_paths[i])
                results["execute_horizon"].append(execute_horizon)
                results["mean_inference_s"].append(mean_inference_s)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = pathlib.Path(output_dir)
    df = pd.DataFrame(results)
    df.to_csv(out_path / "results.csv", index=False)

    # Write mean inference time summary to a .txt file
    inference_df = df[["method", "delay", "execute_horizon", "mean_inference_s"]].drop_duplicates()
    with (out_path / "inference_time_summary.txt").open("w") as f:
        f.write("Mean inference time (s) per action chunk\n")
        f.write("========================================\n")
        for _, row in inference_df.sort_values(["method", "delay", "execute_horizon"]).iterrows():
            f.write(
                f"  method={row['method']}, delay={row['delay']}, execute_horizon={row['execute_horizon']}: "
                f"{row['mean_inference_s']:.4f} s\n"
            )
        f.write(f"\nOverall mean: {df['mean_inference_s'].mean():.4f} s\n")


if __name__ == "__main__":
    tyro.cli(main)
