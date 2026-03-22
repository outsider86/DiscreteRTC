"""Self-forcing fine-tuning for Discrete Diffusion policies.

Trains on trajectory segments: (a) generate from scratch at first obs, supervise with GT;
(b) use model-generated prefix (cut from previous chunk) for inpainting at later obs, supervise with GT.
This exposes the policy to its own output distribution, improving robustness at inference.
"""

import concurrent.futures
import dataclasses
import functools
import json
import pathlib
import pickle
from typing import Sequence

import einops
from flax import struct
import flax.nnx as nnx
import imageio
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv                  # type: ignore
import kinetix.environment.env_state as kenv_state      # type: ignore
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import eval_flow as _eval
import generate_data
import model_dd as _model_dd
from model_dd import IGNORE_TOKEN, continuous_to_bins
import train_expert

WANDB_PROJECT = "rtc-kinetix-dd-self-forcing"


@dataclasses.dataclass(frozen=True)
class Config:
    run_path: str
    load_dir: str  # Required: path to pre-trained checkpoint (e.g. logs-dd/basemodel/31)
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
    )
    batch_size: int = 256  # Number of trajectory segments per batch
    num_epochs: int = 16
    seed: int = 0

    # Self-forcing: trajectory structure (matches eval_dd inference loop)
    inference_delay: int = 2
    execute_horizon: int = 2
    sample_delay_horizon: bool = True  # If True, sample (d,h) per epoch from inference_delays
    inference_delays: tuple[int, ...] = (1, 2, 3, 4)  # d ~ inference_delays, h ~ [d, chunk_size-d]
    num_decision_points: int = 4  # Decision points per segment (first=from scratch, rest=inpainting)

    # Decode params for generation (deterministic for reproducible prefix)
    num_flow_steps: int = 5
    choice_temperature: float = 0.0
    decode_temperature: float = 0.0

    # Eval during training
    eval_num_evals: int = 128
    eval_num_flow_steps: int = 5
    eval_inference_delay: int = 0
    eval_execute_horizon: int = 1
    eval_model: _model_dd.ModelConfig = dataclasses.field(default_factory=_model_dd.ModelConfig)

    log_dir: str = "logs-dd-self-forcing"  # Output directory for checkpoints

    # Fine-tuning LR
    learning_rate: float = 3e-5
    grad_norm_clip: float = 10.0
    weight_decay: float = 1e-5
    lr_warmup_steps: int = 500
    use_cosine_decay: bool = True
    lr_min: float = 1e-6


@struct.dataclass
class EpochCarry:
    rng: jax.Array
    train_state: nnx.State
    graphdef: nnx.GraphDef[tuple[_model_dd.DiscreteDiffusionPolicy, nnx.Optimizer]]


def _make_eval_config(config: Config, execute_horizon: int) -> _eval.EvalConfig:
    return _eval.EvalConfig(
        num_evals=config.eval_num_evals,
        num_flow_steps=config.eval_num_flow_steps,
        inference_delay=config.eval_inference_delay,
        execute_horizon=execute_horizon,
        method=_eval.NaiveMethodConfig(),
        model=config.eval_model,  # type: ignore[arg-type]
    )


def _segment_len_steps(
    num_decision_points: int,
    execute_horizon: int,
    action_chunk_size: int,
) -> int:
    """Number of env steps needed for a segment with num_decision_points decisions."""
    return (num_decision_points - 1) * execute_horizon + action_chunk_size


def _sample_delay_horizon_py(
    rng: jax.Array,
    inference_delays: tuple[int, ...],
    action_chunk_size: int,
) -> tuple[jax.Array, int, int]:
    """Sample (d, h) at Python level. Returns (new_rng, d, h) with concrete ints."""
    rng, d_key, h_key = jax.random.split(rng, 3)
    d_idx = int(jax.device_get(jax.random.randint(d_key, (), 0, len(inference_delays))))
    d = inference_delays[d_idx]
    h = int(jax.device_get(jax.random.randint(h_key, (), d, action_chunk_size - d + 1)))
    return rng, d, h


def main(config: Config):
    static_env_params = kenv_state.StaticEnvParams(
        **train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP
    )
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(config.level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params
    )

    mesh = jax.make_mesh((jax.local_device_count(),), ("level",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("level"))

    action_chunk_size = config.eval_model.action_chunk_size
    if config.sample_delay_horizon:
        h_max = action_chunk_size - 1
        segment_len = _segment_len_steps(
            config.num_decision_points, h_max, action_chunk_size
        )
    else:
        segment_len = _segment_len_steps(
            config.num_decision_points, config.execute_horizon, action_chunk_size
        )

    # Load data (same structure as train_dd)
    def load_data(level_path: str):
        level_name = level_path.replace("/", "_").replace(".json", "")
        print("Loading data for level:", level_name)
        return dict(np.load(pathlib.Path(config.run_path) / "data" / f"{level_name}.npz"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = list(executor.map(load_data, config.level_paths))
    with jax.default_device(jax.devices("cpu")[0]):
        data = jax.tree.map(
            lambda *x: einops.rearrange(jnp.stack(x), "l s e ... -> l (e s) ..."), *data
        )
        total_steps = data["obs"].shape[1]
        # Need segment_len + action_chunk_size - 1 for the last gt_chunk
        required = segment_len + action_chunk_size - 1
        if total_steps < required:
            raise ValueError(
                f"Data has {total_steps} steps but need at least {required} for "
                f"segment_len={segment_len}, chunk_size={action_chunk_size}"
            )
        valid_starts = total_steps - segment_len - action_chunk_size + 1
        if valid_starts < 1:
            valid_starts = 1
        # Truncate to batch-aligned
        num_batches = max(1, valid_starts // config.batch_size)
        truncate = num_batches * config.batch_size + segment_len + action_chunk_size - 1
        data = jax.tree.map(lambda x: x[:, :truncate], data)
        data = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                x.shape,
                sharding,
                [
                    jax.device_put(y, d)
                    for y, d in zip(
                        jnp.split(x, jax.local_device_count()),
                        jax.local_devices(),
                        strict=True,
                    )
                ],
            ),
            data,
        )

    data: generate_data.Data = generate_data.Data(**data)
    print(
        f"Self-forcing: segment_len={segment_len} steps, "
        f"num_decision_points={config.num_decision_points}, "
        f"sample_delay_horizon={config.sample_delay_horizon}, "
        f"valid_starts={valid_starts}, num_batches={num_batches}"
    )

    obs_dim = data.obs.shape[-1]
    action_dim = env.action_space(env_params).shape[0]
    num_bins = config.eval_model.num_bins

    # Load pre-trained checkpoints
    load_path = pathlib.Path(config.load_dir)
    assert load_path.exists(), f"load_dir does not exist: {config.load_dir}"
    state_dicts_raw = []
    for level_path in config.level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        pkl_path = load_path / "policies" / f"{level_name}.pkl"
        assert pkl_path.exists(), f"Missing policy: {pkl_path}"
        with pkl_path.open("rb") as f:
            state_dicts_raw.append(pickle.load(f))
    state_dicts = jax.device_put(
        jax.tree.map(lambda *x: jnp.array(x), *state_dicts_raw)
    )

    def _init_body(rng: jax.Array, state_dict: dict, total_steps: int) -> EpochCarry:
        rng, key = jax.random.split(rng)
        policy = _model_dd.DiscreteDiffusionPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config.eval_model,
            rngs=nnx.Rngs(key),
        )
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        policy = nnx.merge(graphdef, state)
        total_params = sum(
            x.size for x in jax.tree.leaves(nnx.state(policy, nnx.Param))
        )
        print(f"Total params: {total_params:,}")
        if config.use_cosine_decay:
            decay_steps = max(1, total_steps - config.lr_warmup_steps)
            lr_schedule = optax.warmup_cosine_decay_schedule(
                0.0,
                config.learning_rate,
                config.lr_warmup_steps,
                decay_steps,
                config.lr_min,
            )
        else:
            lr_schedule = optax.warmup_constant_schedule(
                0, config.learning_rate, config.lr_warmup_steps
            )
        optimizer = nnx.Optimizer(
            policy,
            optax.chain(
                optax.clip_by_global_norm(config.grad_norm_clip),
                optax.adamw(lr_schedule, weight_decay=config.weight_decay),
            ),
        )
        graphdef, train_state = nnx.split((policy, optimizer))
        return EpochCarry(rng, train_state, graphdef)

    @functools.partial(
        jax.jit,
        in_shardings=(sharding, sharding),
        out_shardings=sharding,
        static_argnums=(2,),
    )
    @functools.partial(jax.vmap, in_axes=(0, 0, None))
    def init_load(rng: jax.Array, state_dict: dict, total_steps: int) -> EpochCarry:
        return _init_body(rng, state_dict, total_steps)

    def _self_forcing_loss_fn(
        policy: _model_dd.DiscreteDiffusionPolicy,
        rng: jax.Array,
        segment_starts: jax.Array,
        data_obs: jax.Array,
        data_action: jax.Array,
        data_done: jax.Array,
        d: int,
        h: int,
    ) -> tuple[jax.Array, dict]:
        """Compute self-forcing loss over one batch of segments.

        For each segment: scan over decision points. Step 0: generate from scratch, loss vs GT.
        Step k>0: prefix from prev generated chunk, inpaint rest, loss vs GT.
        """
        B = segment_starts.shape[0]
        K = config.num_decision_points
        assert h + d <= action_chunk_size, (
            f"execute_horizon({h}) + inference_delay({d}) must be <= action_chunk_size({action_chunk_size})"
        )

        def step_fn(carry, k):
            prev_chunk, rng_state = carry
            rng_state, gen_key, loss_key = jax.random.split(rng_state, 3)

            # Observation and GT chunk at decision point k
            obs_idx = segment_starts[:, None] + k * h  # [B]
            obs = data_obs[segment_starts + k * h]  # [B, obs_dim]
            gt_chunk = data_action[
                segment_starts[:, None] + k * h + jnp.arange(action_chunk_size)[None, :]
            ]  # [B, chunk, action_dim]
            done_chunk = data_done[
                segment_starts[:, None] + k * h + jnp.arange(action_chunk_size)[None, :]
            ]
            done_idxs = jnp.where(
                jnp.any(done_chunk, axis=-1),
                jnp.argmax(done_chunk.astype(jnp.int32), axis=-1),
                action_chunk_size,
            )
            gt_chunk = jnp.where(
                jnp.arange(action_chunk_size)[None, :, None] >= done_idxs[:, None, None],
                0.0,
                gt_chunk,
            )

            def first_step(_):
                # (a) Generate from scratch, standard loss
                loss_total, loss_info = policy.loss(loss_key, obs, gt_chunk)
                next_chunk = policy.action(
                    gen_key, obs, config.num_flow_steps,
                    choice_temperature=config.choice_temperature,
                    decode_temperature=config.decode_temperature,
                )
                return (next_chunk, rng_state), (loss_total, loss_info)

            def later_step(_):
                # (b) Inpainting: prefix = positions [execute_horizon : execute_horizon+d] of prev chunk.
                # In eval, after shift, action_chunk[:, :d] = prev_full_chunk[:, h:h+d] (see eval_dd index flow).
                prefix = prev_chunk[:, h : h + d, :]  # [B, d, action_dim]
                prefix_bins = continuous_to_bins(prefix, num_bins)
                input_tokens = jnp.full(
                    (B, action_chunk_size, action_dim), IGNORE_TOKEN, dtype=jnp.int32
                )
                input_tokens = input_tokens.at[:, :d, :].set(prefix_bins)
                input_tokens = jax.lax.stop_gradient(input_tokens)
                loss_total, loss_info = policy.loss(
                    loss_key, obs, gt_chunk, input_tokens=input_tokens
                )
                next_chunk = policy.realtime_action(
                    gen_key,
                    obs,
                    config.num_flow_steps,
                    prev_chunk,
                    d,
                    h,
                    early_stop=False,
                    adaptive_unmasking=False,
                    choice_temperature=config.choice_temperature,
                    decode_temperature=config.decode_temperature,
                )
                return (next_chunk, rng_state), (loss_total, loss_info)

            return jax.lax.cond(k == 0, first_step, later_step, None)

        init_chunk = jnp.zeros((B, action_chunk_size, action_dim), dtype=jnp.float32)
        (_, _), (losses, infos) = jax.lax.scan(
            step_fn,
            (init_chunk, rng),
            jnp.arange(K),
        )
        loss_mean = jnp.mean(losses)
        # infos is dict of [K] arrays from scan
        ce_mean = jnp.mean(infos["ce_loss"])
        l1_mean = jnp.mean(infos["l1_loss"])
        return loss_mean, {"ce_loss": ce_mean, "l1_loss": l1_mean}

    @functools.partial(
        jax.jit,
        donate_argnums=(0,),
        in_shardings=(sharding, sharding, sharding),
        out_shardings=sharding,
        static_argnums=(3, 4),  # d, h
    )
    @functools.partial(jax.vmap, in_axes=(0, 0, 0, None, None))  # d, h broadcast
    def train_epoch(
        epoch_carry: EpochCarry,
        level: kenv_state.EnvState,
        data: generate_data.Data,
        d: int,
        h: int,
    ):
        def train_minibatch(carry, batch_starts):
            rng, train_state = carry
            policy, optimizer = nnx.merge(epoch_carry.graphdef, train_state)
            rng, key = jax.random.split(rng)

            def loss_fn(policy):
                return _self_forcing_loss_fn(
                    policy, key, batch_starts, data.obs, data.action, data.done, d, h
                )

            (loss, loss_info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(policy)
            info = {
                "loss": loss,
                "loss_ce": loss_info["ce_loss"],
                "loss_l1": loss_info["l1_loss"],
                "grad_norm": optax.global_norm(grads),
            }
            optimizer.update(grads)
            _, train_state = nnx.split((policy, optimizer))
            return (rng, train_state), info

        rng, key = jax.random.split(epoch_carry.rng)
        valid_starts = data.obs.shape[0] - segment_len - action_chunk_size + 1
        valid_starts = max(1, valid_starts)
        permutation = jax.random.permutation(key, valid_starts)
        n_batches = valid_starts // config.batch_size
        n_batches = max(1, n_batches)
        batch_starts = permutation[: n_batches * config.batch_size].reshape(
            n_batches, config.batch_size
        )
        (rng, train_state), train_info = jax.lax.scan(
            train_minibatch, (epoch_carry.rng, epoch_carry.train_state), batch_starts
        )
        train_info = jax.tree.map(lambda x: x.mean(), train_info)

        rng, key = jax.random.split(rng)
        eval_policy, _ = nnx.merge(epoch_carry.graphdef, train_state)
        eval_info = {}
        for horizon in range(1, action_chunk_size + 1):
            eval_config = _make_eval_config(config, horizon)
            info, _ = _eval.eval(
                eval_config,
                env,
                key,
                level,
                eval_policy,
                env_params,
                static_env_params,  # type: ignore[arg-type]
            )
            eval_info.update({f"{k}_{horizon}": v for k, v in info.items()})
        video = None
        return EpochCarry(rng, train_state, epoch_carry.graphdef), (
            {**train_info, **eval_info},
            video,
        )

    total_steps = config.num_epochs * num_batches
    wandb.init(project=WANDB_PROJECT)
    run_dir = pathlib.Path(config.log_dir) / wandb.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
    with (run_dir / "model_config.json").open("w") as f:
        json.dump(dataclasses.asdict(config.eval_model), f, indent=2)

    rng = jax.random.key(config.seed)
    rngs = jax.random.split(rng, len(config.level_paths))
    epoch_carry = init_load(rngs, state_dicts, total_steps)
    rng_dh = jax.random.key(config.seed + 999)

    for epoch_idx in tqdm.tqdm(range(config.num_epochs)):
        if config.sample_delay_horizon:
            rng_dh, d, h = _sample_delay_horizon_py(
                rng_dh, config.inference_delays, action_chunk_size
            )
        else:
            d = config.inference_delay
            h = config.execute_horizon
        epoch_carry, (info, video) = train_epoch(
            epoch_carry, levels, data, d, h
        )
        info = jax.device_get(info)
        video = jax.device_get(video) if video is not None else None
        train_state_host = jax.device_get(epoch_carry.train_state)

        for i in range(len(config.level_paths)):
            level_name = config.level_paths[i].replace("/", "_").replace(".json", "")
            wandb.log({f"{level_name}/{k}": v[i] for k, v in info.items()}, step=epoch_idx)
            log_dir = pathlib.Path(config.log_dir) / wandb.run.name / str(epoch_idx)
            if video is not None:
                video_dir = log_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(video_dir / f"{level_name}.mp4", video[i], fps=15)
            policy_dir = log_dir / "policies"
            policy_dir.mkdir(parents=True, exist_ok=True)
            level_train_state = jax.tree.map(lambda x: x[i], train_state_host)
            with (policy_dir / f"{level_name}.pkl").open("wb") as f:
                policy, _ = nnx.merge(epoch_carry.graphdef, level_train_state)
                pickle.dump(nnx.state(policy).to_pure_dict(), f)


if __name__ == "__main__":
    tyro.cli(main)
