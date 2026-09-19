"""Train a waypoint-navigation PPO policy on SailBench."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from functools import partial
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

# Ensure the repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sailbench.rl.envs.waypoint_env import WaypointEnv, WaypointEnvConfig
from sailbench.rl.live_vis import LiveTrainingVisServer


def _sb3_waypoint_monitor_env(env_dict: dict[str, Any]) -> Monitor:
    """Top-level factory for SubprocVecEnv workers (must be picklable)."""
    cfg = WaypointEnvConfig(**env_dict)
    return Monitor(WaypointEnv(config=cfg))


class SeededEvalCallback(EvalCallback):
    """`EvalCallback` that re-seeds its environment before every evaluation.

    Left unseeded, each evaluation draws a fresh set of start poses and
    waypoints, and the waypoint radius alone spans 12-25 m. The score then moves
    with the draw as much as with the policy -- successive evaluations on one run
    came back at -348, -1391 and -633 with no trend -- and `best_model` is
    selected partly by which evaluation happened to draw short legs.

    Seeding once per evaluation is enough to fix the whole set: `reset(seed=...)`
    re-seeds the env's RNG, and the auto-resets that carry it through the
    remaining episodes continue that one stream deterministically.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        *,
        eval_seed: int,
        best_model_save_path: str,
        log_path: str,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool,
    ) -> None:
        """Wrap `EvalCallback`, remembering the seed to restart each eval from."""
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
        )
        self.eval_seed = int(eval_seed)

    def _on_step(self) -> bool:
        """Re-seed the eval env on the steps where the parent would evaluate."""
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.seed(self.eval_seed)
        return bool(super()._on_step())


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return dict(yaml.safe_load(file) or {})


def _build_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"waypoint_ppo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PPO for SailBench waypoint navigation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl_waypoint_sb3.yaml"),
        help="Path to RL training config YAML.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs"),
        help="Base directory for training outputs.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a saved PPO checkpoint .zip to continue training from.",
    )
    parser.add_argument(
        "--watch-web",
        action="store_true",
        help="Broadcast env-0 training state over websocket for the browser visualizer.",
    )
    parser.add_argument(
        "--watch-host",
        default="127.0.0.1",
        help="Host for training visualization websocket server.",
    )
    parser.add_argument(
        "--watch-port",
        type=int,
        default=8765,
        help="Port for training visualization websocket server.",
    )
    parser.add_argument(
        "--watch-stride",
        type=int,
        default=None,
        help="Publish every N env steps (default from config or 2).",
    )
    args = parser.parse_args(argv)

    full_cfg = _load_yaml(args.config)
    env_cfg = WaypointEnvConfig(**full_cfg.get("env", {}))
    train_cfg = full_cfg.get("train", {})

    run_dir = _build_run_dir(args.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"
    eval_logs_dir = run_dir / "eval_logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_logs_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(full_cfg, file, sort_keys=False)

    seed = int(train_cfg.get("seed", 7))
    device = str(train_cfg.get("device", "cpu"))
    n_envs = int(train_cfg.get("n_envs", 4))
    total_timesteps = int(train_cfg.get("total_timesteps", 200_000))
    watch_enabled = bool(args.watch_web or train_cfg.get("watch_web", False))
    watch_stride = int(args.watch_stride if args.watch_stride is not None else train_cfg.get("watch_stride", 2))

    vis_server: LiveTrainingVisServer | None = None
    if watch_enabled:
        vis_server = LiveTrainingVisServer(host=args.watch_host, port=args.watch_port)
        vis_server.start()

    vec_env_kind = str(train_cfg.get("vec_env", "subproc")).lower()
    if watch_enabled:
        if vec_env_kind == "subproc":
            print(
                "watch_web: using DummyVecEnv (live visualization must run in-process; "
                "not picklable for SubprocVecEnv workers).",
            )
        vec_env_kind = "dummy"

    if vec_env_kind == "subproc":
        env_dict = dict(full_cfg.get("env", {}))
        train_env = SubprocVecEnv([partial(_sb3_waypoint_monitor_env, env_dict) for _ in range(n_envs)])
        print(f"Training with SubprocVecEnv: {n_envs} parallel environment processes.")
    else:
        def make_env(env_idx: int) -> Monitor:
            cfg = env_cfg
            if watch_enabled and env_idx == 0 and vis_server is not None:
                cfg = replace(
                    env_cfg,
                    vis_callback=vis_server.publish,
                    vis_stride_steps=max(watch_stride, 1),
                )
            return Monitor(WaypointEnv(config=cfg))

        train_env = DummyVecEnv([lambda i=i: make_env(i) for i in range(n_envs)])
    eval_env = DummyVecEnv([lambda: Monitor(WaypointEnv(config=env_cfg))])

    use_norm_obs = bool(train_cfg.get("normalize_observation", False))
    use_norm_reward = bool(train_cfg.get("normalize_reward", False))
    if use_norm_obs or use_norm_reward:
        train_env = VecNormalize(
            train_env,
            norm_obs=use_norm_obs,
            norm_reward=use_norm_reward,
            clip_obs=10.0,
            training=True,
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=use_norm_obs,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
        )
        if args.resume_from is not None:
            inferred_vec_path = args.resume_from.with_name(f"{args.resume_from.stem}_vecnormalize.pkl")
            if inferred_vec_path.exists():
                train_env = VecNormalize.load(str(inferred_vec_path), train_env)
                eval_env = VecNormalize.load(str(inferred_vec_path), eval_env)
                train_env.training = True
                train_env.norm_reward = use_norm_reward
                eval_env.training = False
                eval_env.norm_reward = False
            else:
                print(
                    f"WARNING: normalization enabled but could not find {inferred_vec_path}; "
                    "continuing without loading saved VecNormalize statistics.",
                )

    tensorboard_log = train_cfg.get("tensorboard_log", "runs/tensorboard")
    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume_from}")
        # On resume, keep model weights from checkpoint but allow fine-tune overrides from config.
        resume_overrides: dict[str, Any] = {
            "learning_rate": float(train_cfg.get("learning_rate", 3e-4)),
            "ent_coef": float(train_cfg.get("ent_coef", 0.0)),
        }
        model = PPO.load(
            str(args.resume_from),
            env=train_env,
            tensorboard_log=tensorboard_log,
            device=device,
            custom_objects=resume_overrides,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
            n_steps=int(train_cfg.get("n_steps", 1024)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            n_epochs=int(train_cfg.get("n_epochs", 10)),
            gamma=float(train_cfg.get("gamma", 0.99)),
            gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
            clip_range=float(train_cfg.get("clip_range", 0.2)),
            ent_coef=float(train_cfg.get("ent_coef", 0.0)),
            vf_coef=float(train_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
            use_sde=bool(train_cfg.get("use_sde", False)),
            seed=seed,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=device,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(train_cfg.get("checkpoint_freq", 25_000)) // n_envs, 1),
        save_path=str(checkpoints_dir),
        name_prefix="ppo_waypoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    eval_callback = SeededEvalCallback(
        eval_seed=seed,
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_logs_dir),
        eval_freq=max(int(train_cfg.get("eval_freq", 10_000)) // n_envs, 1),
        n_eval_episodes=int(train_cfg.get("eval_episodes", 10)),
        deterministic=bool(train_cfg.get("deterministic_eval", True)),
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=args.resume_from is None,
        )

        final_model_path = run_dir / "final_model.zip"
        model.save(str(final_model_path))
        if isinstance(train_env, VecNormalize):
            train_env.save(str(run_dir / "vecnormalize.pkl"))

        summary = {
            "run_dir": str(run_dir),
            "seed": seed,
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "parallel_rollout_envs": n_envs if vec_env_kind == "subproc" else 1,
            "vec_env": vec_env_kind,
            "final_model_path": str(final_model_path),
            "resumed_from": str(args.resume_from) if args.resume_from is not None else None,
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
    finally:
        train_env.close()
        eval_env.close()
        if vis_server is not None:
            vis_server.close()


if __name__ == "__main__":
    main()

