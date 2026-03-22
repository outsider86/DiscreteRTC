import collections
import dataclasses
import functools
import math
import pathlib
import pickle
from typing import Sequence

import flax.nnx as nnx
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import kinetix.environment.env as kenv                          # type: ignore
import kinetix.environment.env_state as kenv_state              # type: ignore
import kinetix.environment.wrappers as wrappers                 # type: ignore
import kinetix.render.renderer_pixels as renderer_pixels        # type: ignore
import pandas as pd                                        
from tqdm import tqdm
import tyro

import model_dd as _model_dd
import train_expert
import compute_robot_indices


@dataclasses.dataclass(frozen=True)
class NaiveMethodConfig:
    pass


@dataclasses.dataclass(frozen=True)
class RealtimeMethodConfig:
    """Realtime decode for discrete diffusion (uses DD realtime_action; fields kept for CLI compatibility)."""
    early_stop: bool = False
    prefix_attention_schedule: str = "exp"
    max_guidance_weight: float = 5.0


@dataclasses.dataclass(frozen=True)
class DiscreteRTCConfig:
    """Discrete RTC method (realtime_action with early_stop=True)."""
    early_stop: bool = True


@dataclasses.dataclass(frozen=True)
class BIDMethodConfig:
    n_samples: int = 16
    bid_k: int | None = None


@dataclasses.dataclass(frozen=True)
class VLASHMethodConfig:
    """Visual Lag, Actual State Hybrid - simulate robot state after executing delay actions"""
    with_noise: bool = False  # If True, use different RNG for simulation to introduce prediction error


@dataclasses.dataclass(frozen=True)
class OracleMethodConfig:
    """Oracle - use fully simulated future observation (no delay, perfect information)"""
    pass


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    step: int = -1
    weak_step: int | None = None
    num_evals: int = 2048
    num_flow_steps: int = 5

    inference_delay: int = 0
    execute_horizon: int = 1
    method: NaiveMethodConfig | RealtimeMethodConfig | DiscreteRTCConfig | BIDMethodConfig | VLASHMethodConfig | OracleMethodConfig = NaiveMethodConfig()

    # Discrete diffusion model config
    model: _model_dd.ModelConfig = _model_dd.ModelConfig()
    # Decode-time temperatures for discrete diffusion
    choice_temperature: float = 0.0
    decode_temperature: float = 1.0


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
    ct, dt = config.choice_temperature, config.decode_temperature

    def execute_chunk(carry, _):
        def step(carry, action):
            rng, obs, env_state = carry
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
            return (rng, next_obs, next_env_state), (done, env_state, info)

        rng, obs, env_state, action_chunk, n = carry
        rng, key = jax.random.split(rng)
        if isinstance(config.method, NaiveMethodConfig):
            next_action_chunk = policy.action(
                key,
                obs,
                config.num_flow_steps,
                choice_temperature=ct,
                decode_temperature=dt,
            )
        elif isinstance(config.method, (RealtimeMethodConfig, DiscreteRTCConfig)):
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
        elif isinstance(config.method, (VLASHMethodConfig, OracleMethodConfig)):
            # VLASH: mix env@current + robot@simulated_future
            # Oracle: use fully simulated future observation
            is_vlash = isinstance(config.method, VLASHMethodConfig)
            if is_vlash:
                assert robot_mask is not None, "robot_mask is required for VLASH"
            
            @jax.vmap
            def simulate_future_state(rng_sim, obs_current, env_state_wrapped, prev_actions):
                if config.inference_delay > 0:
                    def sim_step(carry, action):
                        state_wrapped, rng_carry = carry
                        rng_next, key_step = jax.random.split(rng_carry)
                        obs_next, next_state_wrapped, _, _, _ = env._env.step(
                            key_step, state_wrapped, action, env_params
                        )
                        return (next_state_wrapped, rng_next), obs_next
                    
                    (_, _), obs_sequence = jax.lax.scan(
                        sim_step, (env_state_wrapped, rng_sim), prev_actions[:config.inference_delay]
                    )
                    obs_future = obs_sequence[-1]
                else:
                    obs_future = obs_current
                
                # VLASH: mix using robot_mask; Oracle: use full future
                if is_vlash:
                    return jnp.where(robot_mask, obs_future, obs_current)
                return obs_future
            
            # VLASH with_noise=True: use different RNG to introduce prediction error
            # VLASH with_noise=False / Oracle: use same RNG for perfect noise prediction
            use_noise = is_vlash and config.method.with_noise
            if use_noise:
                rng, rng_for_sim = jax.random.split(rng)
            else:
                rng_for_sim = rng
            obs_for_policy = simulate_future_state(
                jax.random.split(rng_for_sim, config.num_evals), obs, env_state, action_chunk
            )
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
        if isinstance(config.method, (VLASHMethodConfig, OracleMethodConfig)):
            # VLASH/Oracle: next_action_chunk generated with simulated future obs
            shift = config.execute_horizon - config.inference_delay
            action_chunk_to_execute = jnp.concatenate(
                [action_chunk[:, :config.inference_delay], next_action_chunk[:, :shift]], axis=1
            )
            next_action_chunk = jnp.concatenate(
                [next_action_chunk[:, shift:], jnp.zeros((obs.shape[0], shift, policy.action_dim))], axis=1
            )
            next_n = jnp.concatenate([n[shift:], jnp.zeros(shift, dtype=jnp.int32)])
        else:
            # Other methods: next_action_chunk generated with current obs
            action_chunk_to_execute = jnp.concatenate(
                [action_chunk[:, :config.inference_delay], next_action_chunk[:, config.inference_delay:config.execute_horizon]], axis=1
            )
            next_action_chunk = jnp.concatenate(
                [next_action_chunk[:, config.execute_horizon:], jnp.zeros((obs.shape[0], config.execute_horizon, policy.action_dim))], axis=1
            )
            next_n = jnp.concatenate([n[config.execute_horizon:], jnp.zeros(config.execute_horizon, dtype=jnp.int32)])
        (rng, next_obs, next_env_state), (dones, env_states, infos) = jax.lax.scan(
            step, (rng, obs, env_state), action_chunk_to_execute.transpose(1, 0, 2)
        )
        # if config.inference_delay > 0:
        #     infos["match"] = jnp.mean(jnp.abs(fixed_prefix - action_chunk_to_execute))
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
    )  # [batch, horizon, action_dim]
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
    # Average task completion time (seconds): mean episode length * step duration (discrete diffusion eval compatibility)
    step_duration_s = float(env_params.dt) * int(static_env_params.frame_skip)
    return_info["mean_task_completion_time_s"] = return_info["returned_episode_lengths"] * step_duration_s
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
    parallel_index: int | None = None,
    parallel_total: int | None = None,
):
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

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    # Compute robot masks for VLASH/Oracle
    robot_masks = jnp.stack([compute_robot_indices.compute_robot_mask(p, obs_dim) for p in level_paths])
    robot_masks = jax.device_put(robot_masks, sharding)

    @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec, pspec, pspec), out_specs=pspec)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
    def _eval(config: EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict, weak_state_dict, robot_mask):
        policy = _model_dd.DiscreteDiffusionPolicy(
            obs_dim=obs_dim, action_dim=action_dim, config=config.model, rngs=nnx.Rngs(rng),
        )
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        policy = nnx.merge(graphdef, state)
        if weak_state_dict is not None:
            weak_graphdef, weak_state = nnx.split(policy)
            weak_state.replace_by_pure_dict(weak_state_dict)
            weak_policy = nnx.merge(weak_graphdef, weak_state)
        else:
            weak_policy = None
        eval_info, _ = eval(config, env, rng, level, policy, env_params, static_env_params, weak_policy, robot_mask)
        return eval_info

    # Build evaluation tasks
    # methods = ["naive", "realtime", "bid", "vlash", "vlash_w_noise", "oracle"]
    methods = ["vlash", "vlash_w_noise", "oracle"]
    delay_horizon_pairs = [(d, max(1, d)) for d in range(8)] + [(1, h) for h in range(2, 9)]
    all_tasks = [(d, h, m) for d, h in delay_horizon_pairs for m in methods]
    
    if parallel_index is not None and parallel_total is not None:
        all_tasks = [t for i, t in enumerate(all_tasks) if i % parallel_total == parallel_index]
        print(f"Parallel {parallel_index}/{parallel_total}: {len(all_tasks)} tasks")

    rngs = jax.random.split(jax.random.key(seed), len(level_paths))
    results = collections.defaultdict(list)
    
    desc = f"GPU{parallel_index}" if parallel_index is not None else "Eval"
    for delay, horizon, method_name in tqdm(all_tasks, desc=desc, ncols=80):
        method_config = {
            "naive": NaiveMethodConfig(),
            "realtime": RealtimeMethodConfig(early_stop=False),
            "discrete_rtc": DiscreteRTCConfig(early_stop=True),
            "bid": BIDMethodConfig(),
            "vlash": VLASHMethodConfig(with_noise=False),
            "vlash_w_noise": VLASHMethodConfig(with_noise=True),
            "oracle": OracleMethodConfig(),
        }[method_name]
        
        c = dataclasses.replace(config, inference_delay=delay, execute_horizon=horizon, method=method_config)
        out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts, robot_masks))
        
        for i in range(len(level_paths)):
            for k, v in out.items():
                results[k].append(v[i])
            results["delay"].append(delay)
            results["method"].append(method_name)
            results["level"].append(level_paths[i])
            results["execute_horizon"].append(horizon)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"results_part_{parallel_index}.csv" if parallel_index is not None else "results.csv"
    pd.DataFrame(results).to_csv(pathlib.Path(output_dir) / output_file, index=False)


if __name__ == "__main__":
    tyro.cli(main)