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
import kinetix.environment.env as kenv                          # type: ignore
import kinetix.environment.env_state as kenv_state              # type: ignore
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import eval_flow as _eval
import generate_data
import model_dd as _model_dd
import train_expert
import compute_robot_indices

WANDB_PROJECT = "rtc-kinetix-vlash"
LOG_DIR = pathlib.Path("logs-vlash-dd")


@dataclasses.dataclass(frozen=True)
class Config:
    run_path: str
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
    batch_size: int = 512
    num_epochs: int = 32
    seed: int = 0
    output_dir: str | None = None  # If None, use LOG_DIR / wandb.run.name

    # Eval during training (same structure as train_dd)
    eval_num_evals: int = 128
    eval_num_flow_steps: int = 5
    eval_inference_delay: int = 0
    eval_execute_horizon: int = 1
    eval_model: _model_dd.ModelConfig = dataclasses.field(default_factory=_model_dd.ModelConfig)

    learning_rate: float = 3e-4
    grad_norm_clip: float = 10.0
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 2000
    use_cosine_decay: bool = True
    lr_min: float = 1e-5

    load_dir: str | None = None
    # Async-augmented (VLASH) training: if > 0, randomly delay obs by [0, async_interval) and mix robot@future + env@current
    async_interval: int = 0


@struct.dataclass
class EpochCarry:
    rng: jax.Array
    train_state: nnx.State
    graphdef: nnx.GraphDef[tuple[_model_dd.DiscreteDiffusionPolicy, nnx.Optimizer]]


def _make_eval_config(config: Config, execute_horizon: int) -> _eval.EvalConfig:
    """Build eval_flow.EvalConfig for a given execute_horizon (same as train_dd)."""
    return _eval.EvalConfig(
        num_evals=config.eval_num_evals,
        num_flow_steps=config.eval_num_flow_steps,
        inference_delay=config.eval_inference_delay,
        execute_horizon=execute_horizon,
        method=_eval.NaiveMethodConfig(),
        model=config.eval_model,  # type: ignore[arg-type]
    )


def main(config: Config):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(config.level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    mesh = jax.make_mesh((jax.local_device_count(),), ("level",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("level"))

    action_chunk_size = config.eval_model.action_chunk_size

    # load data (same as train_dd, with extra reserve for async_interval)
    def load_data(level_path: str):
        level_name = level_path.replace("/", "_").replace(".json", "")
        print("Loading data for level:", level_name)
        return dict(np.load(pathlib.Path(config.run_path) / "data" / f"{level_name}.npz"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = list(executor.map(load_data, config.level_paths))
    with jax.default_device(jax.devices("cpu")[0]):
        data = jax.tree.map(lambda *x: einops.rearrange(jnp.stack(x), "l s e ... -> l (e s) ..."), *data)
        max_async = max(0, config.async_interval - 1) if config.async_interval > 0 else 0
        valid_steps = data["obs"].shape[1] - action_chunk_size - max_async + 1
        data = jax.tree.map(
            lambda x: x[:, : (valid_steps // config.batch_size) * config.batch_size + action_chunk_size + max_async - 1],
            data,
        )
        data = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                x.shape,
                sharding,
                [
                    jax.device_put(y, d)
                    for y, d in zip(jnp.split(x, jax.local_device_count()), jax.local_devices(), strict=True)
                ],
            ),
            data,
        )

    data: generate_data.Data = generate_data.Data(**data)
    print(f"Truncated data to {data.obs.shape[1]:_} steps ({valid_steps // config.batch_size:_} batches)")

    obs_dim = data.obs.shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    num_batches = valid_steps // config.batch_size
    total_steps = config.num_epochs * num_batches

    if config.load_dir is not None:
        state_dicts = []
        for level_path in config.level_paths:
            level_name = level_path.replace("/", "_").replace(".json", "")
            with (pathlib.Path(config.load_dir) / "policies" / f"{level_name}.pkl").open("rb") as f:
                state_dicts.append(pickle.load(f))
        state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))
    else:
        state_dicts = None

    # Compute robot masks for async-augmented (VLASH) training
    robot_masks = None
    if config.async_interval > 0:
        print(f"Async-augmented (VLASH) training: delay in [0, {config.async_interval})")
        robot_masks = jnp.stack([compute_robot_indices.compute_robot_mask(p, obs_dim) for p in config.level_paths])
        robot_masks = jax.device_put(robot_masks, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("level")))

    def _init_body(rng: jax.Array, state_dict: dict | None, total_steps: int) -> EpochCarry:
        rng, key = jax.random.split(rng)
        policy = _model_dd.DiscreteDiffusionPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config.eval_model,
            rngs=nnx.Rngs(key),
        )
        if state_dict is not None:
            graphdef, state = nnx.split(policy)
            state.replace_by_pure_dict(state_dict)
            policy = nnx.merge(graphdef, state)
        total_params = sum(x.size for x in jax.tree.leaves(nnx.state(policy, nnx.Param)))
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
        in_shardings=(sharding, None),
        out_shardings=sharding,
        static_argnums=(2,),
    )
    @functools.partial(jax.vmap, in_axes=(0, None, None))
    def init_no_load(rng: jax.Array, state_dict: dict | None, total_steps: int) -> EpochCarry:
        return _init_body(rng, state_dict, total_steps)

    @functools.partial(
        jax.jit,
        in_shardings=(sharding, sharding),
        out_shardings=sharding,
        static_argnums=(2,),
    )
    @functools.partial(jax.vmap, in_axes=(0, 0, None))
    def init_load(rng: jax.Array, state_dict: dict, total_steps: int) -> EpochCarry:
        return _init_body(rng, state_dict, total_steps)

    @functools.partial(jax.jit, donate_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def train_epoch(epoch_carry: EpochCarry, level: kenv_state.EnvState, data: generate_data.Data, robot_mask):
        def train_minibatch(carry: tuple[jax.Array, nnx.State], batch_idxs: jax.Array):
            rng, train_state = carry
            policy, optimizer = nnx.merge(epoch_carry.graphdef, train_state)

            rng, key = jax.random.split(rng)

            def loss_fn(policy: _model_dd.DiscreteDiffusionPolicy):
                obs_current = data.obs[batch_idxs]
                if config.async_interval > 0:
                    rng_local, key_delay = jax.random.split(key)
                    delays = jax.random.randint(key_delay, (batch_idxs.shape[0],), 0, config.async_interval)
                    obs_future = data.obs[batch_idxs + delays]
                    obs = jnp.where(robot_mask[None, :], obs_future, obs_current)
                    action_indices = batch_idxs[:, None] + delays[:, None] + jnp.arange(action_chunk_size)[None, :]
                else:
                    rng_local = key
                    obs = obs_current
                    action_indices = batch_idxs[:, None] + jnp.arange(action_chunk_size)[None, :]
                action_chunks = data.action[action_indices]
                done_chunks = data.done[action_indices]
                done_idxs = jnp.where(
                    jnp.any(done_chunks, axis=-1),
                    jnp.argmax(done_chunks, axis=-1),
                    action_chunk_size,
                )
                action_chunks = jnp.where(
                    jnp.arange(action_chunk_size)[None, :, None] >= done_idxs[:, None, None],
                    0.0,
                    action_chunks,
                )
                loss_total, loss_info = policy.loss(rng_local, obs, action_chunks)
                return loss_total, loss_info

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

        # shuffle
        rng, key = jax.random.split(epoch_carry.rng)
        max_async = max(0, config.async_interval - 1) if config.async_interval > 0 else 0
        permutation = jax.random.permutation(key, data.obs.shape[0] - action_chunk_size - max_async + 1)
        # batch
        permutation = permutation.reshape(-1, config.batch_size)
        # train
        (rng, train_state), train_info = jax.lax.scan(
            train_minibatch, (epoch_carry.rng, epoch_carry.train_state), permutation
        )
        train_info = jax.tree.map(lambda x: x.mean(), train_info)
        # eval
        rng, key = jax.random.split(rng)
        eval_policy, _ = nnx.merge(epoch_carry.graphdef, train_state)
        eval_info = {}
        for horizon in range(1, action_chunk_size + 1):
            eval_config = _make_eval_config(config, horizon)
            info, _ = _eval.eval(eval_config, env, key, level, eval_policy, env_params, static_env_params)
            eval_info.update({f"{k}_{horizon}": v for k, v in info.items()})
        video = None
        return EpochCarry(rng, train_state, epoch_carry.graphdef), ({**train_info, **eval_info}, video)

    wandb.init(project=WANDB_PROJECT)
    run_dir = pathlib.Path(config.output_dir) if config.output_dir is not None else LOG_DIR / wandb.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
    with (run_dir / "model_config.json").open("w") as f:
        json.dump(dataclasses.asdict(config.eval_model), f, indent=2)

    rng = jax.random.key(config.seed)
    rngs = jax.random.split(rng, len(config.level_paths))
    if state_dicts is None:
        epoch_carry = init_no_load(rngs, None, total_steps)
    else:
        epoch_carry = init_load(rngs, state_dicts, total_steps)

    dummy_masks = jnp.zeros((len(config.level_paths), obs_dim), dtype=bool)
    masks = robot_masks if robot_masks is not None else dummy_masks

    for epoch_idx in tqdm.tqdm(range(config.num_epochs)):
        epoch_carry, (info, video) = train_epoch(epoch_carry, levels, data, masks)
        info = jax.device_get(info)
        video = jax.device_get(video) if video is not None else None
        train_state_host = jax.device_get(epoch_carry.train_state)

        for i in range(len(config.level_paths)):
            level_name = config.level_paths[i].replace("/", "_").replace(".json", "")
            wandb.log({f"{level_name}/{k}": v[i] for k, v in info.items()}, step=epoch_idx)

            log_dir = run_dir / str(epoch_idx)

            if video is not None:
                video_dir = log_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(video_dir / f"{level_name}.mp4", video[i], fps=15)

            policy_dir = log_dir / "policies"
            policy_dir.mkdir(parents=True, exist_ok=True)
            level_train_state = jax.tree.map(lambda x: x[i], train_state_host)
            with (policy_dir / f"{level_name}.pkl").open("wb") as f:
                policy, _ = nnx.merge(epoch_carry.graphdef, level_train_state)
                state_dict = nnx.state(policy).to_pure_dict()
                pickle.dump(state_dict, f)


if __name__ == "__main__":
    tyro.cli(main)