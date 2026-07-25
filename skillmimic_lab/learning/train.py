#!/usr/bin/env python3
"""Self-contained RL-Games training entry point for migrated SkillMimic tasks."""

from __future__ import annotations

import argparse
import copy
import math
import os
import traceback
from datetime import datetime

from omni.isaac.lab.app import AppLauncher

from skillmimic_lab.kit_runtime import configure_kit_runtime


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="Train a migrated SkillMimic task with RL-Games")
parser.add_argument("--task", choices=("ballplay", "circling", "heading", "throwing", "scoring"), default="ballplay")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--minibatch_size", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", default=None, help="Resume an Isaac Lab-era RL-Games checkpoint")
parser.add_argument("--motion_path", default=None)
parser.add_argument("--llc_checkpoint", default=None)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

configure_kit_runtime(disable_ngx=args.headless)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _record_run_status(exit_code: int) -> None:
    """Persist failures before SimulationApp shutdown can change the exit code."""

    # torchrun propagates failures from every worker. Only rank zero should
    # update the wrapper's shared status file.
    if os.environ.get("RANK", "0") != "0":
        return
    status_file = os.environ.get("SKILLMIMIC_RUN_STATUS_FILE")
    if status_file is None:
        return
    try:
        with open(status_file, "w", encoding="utf-8") as stream:
            stream.write(f"{exit_code}\n")
    except OSError as exc:
        print(f"[SkillMimic Lab] Could not write training status to {status_file}: {exc}", flush=True)

import gymnasium as gym
import gym as legacy_gym
import torch
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

import skillmimic_lab  # noqa: F401
from skillmimic_lab.env.tasks.skillmimic import SkillMimicBallPlayEnvCfg
from skillmimic_lab.env.tasks.hrl_base import (
    SkillMimicCirclingEnvCfg,
    SkillMimicHeadingEnvCfg,
    SkillMimicScoringEnvCfg,
    SkillMimicThrowingEnvCfg,
)
from skillmimic_lab.learning.rl_games_wrapper import RlGamesGpuEnv, RlGamesVecEnvWrapper


TASKS = {
    "ballplay": ("SkillMimic-BallPlay-Direct-v0", SkillMimicBallPlayEnvCfg, "rl_games_ppo_cfg.yaml"),
    "circling": ("SkillMimic-Circling-Direct-v0", SkillMimicCirclingEnvCfg, "rl_games_hlc_ppo_cfg.yaml"),
    "heading": ("SkillMimic-Heading-Direct-v0", SkillMimicHeadingEnvCfg, "rl_games_hlc_ppo_cfg.yaml"),
    "throwing": ("SkillMimic-Throwing-Direct-v0", SkillMimicThrowingEnvCfg, "rl_games_hlc_ppo_cfg.yaml"),
    "scoring": ("SkillMimic-Scoring-Direct-v0", SkillMimicScoringEnvCfg, "rl_games_hlc_ppo_cfg.yaml"),
}


class SkillMimicAlgoObserver(IsaacAlgoObserver):
    """Print the same rolling episode statistics that RL-Games logs to TensorBoard."""

    REWARD_TERMS = (
        "position",
        "rotation",
        "angular_velocity",
        "smoothness",
        "object_position",
        "object_velocity",
        "interaction",
        "body_contact",
        "object_contact",
        "body",
        "object",
        "total",
    )

    def after_init(self, algo) -> None:
        super().after_init(algo)
        self._reward_term_sums = {name: 0.0 for name in self.REWARD_TERMS}
        self._reward_term_counts = {name: 0 for name in self.REWARD_TERMS}

    def process_infos(self, infos, done_indices) -> None:
        super().process_infos(infos, done_indices)
        if not isinstance(infos, dict):
            return
        for name in self.REWARD_TERMS:
            value = infos.get(f"skillmimic/reward/{name}")
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            else:
                value = float(value)
            if math.isfinite(value):
                self._reward_term_sums[name] += value
                self._reward_term_counts[name] += 1

    def _print_reward_terms(self, epoch_num: int) -> None:
        values = []
        for name in self.REWARD_TERMS:
            count = self._reward_term_counts[name]
            if count > 0:
                values.append(f"{name}={self._reward_term_sums[name] / count:.6f}")
            self._reward_term_sums[name] = 0.0
            self._reward_term_counts[name] = 0
        if values:
            print(
                f"[SkillMimic Lab][reward-components] epoch={epoch_num} " + " ".join(values),
                flush=True,
            )

    def after_print_stats(self, frame: int, epoch_num: int, total_time: float) -> None:
        super().after_print_stats(frame, epoch_num, total_time)
        if self.algo.game_rewards.current_size <= 0:
            print(
                f"[SkillMimic Lab] epoch={epoch_num} frames={frame} "
                "mean_reward=<waiting-for-completed-episode>",
                flush=True,
            )
            self._print_reward_terms(epoch_num)
            return

        mean_reward = self.algo.game_rewards.get_mean().reshape(-1)[0].item()
        mean_length = self.algo.game_lengths.get_mean().item()
        print(
            f"[SkillMimic Lab] epoch={epoch_num} frames={frame} "
            f"mean_reward={mean_reward:.8f} mean_episode_length={mean_length:.2f}",
            flush=True,
        )
        self._print_reward_terms(epoch_num)


class SkillMimicRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
    """Adds the discrete action-space bridge missing from Isaac Lab 1.1's wrapper."""

    def __init__(self, env, rl_device: str, clip_obs: float, clip_actions: float):
        super().__init__(env, rl_device, clip_obs, clip_actions)
        if isinstance(self.unwrapped.single_action_space, gym.spaces.Discrete):
            self._clip_actions = self.unwrapped.single_action_space.n - 1

    @property
    def action_space(self):
        action_space = self.unwrapped.single_action_space
        if isinstance(action_space, gym.spaces.Discrete):
            return legacy_gym.spaces.Discrete(action_space.n)
        return super().action_space

def main() -> None:
    task_id, cfg_type, agent_filename = TASKS[args.task]
    env_cfg = cfg_type()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed + (app_launcher.global_rank if args.distributed else 0)
    # Isaac Lab v1.1 exposes --device_id, while newer releases may expose
    # --device. Keep the simulator and RL policy on the same selected device.
    if args.distributed:
        device = f"cuda:{app_launcher.local_rank}"
    else:
        device = getattr(args, "device", None)
        if device is None:
            device = f"cuda:{getattr(args, 'device_id', 0)}"
    env_cfg.sim.device = device
    if args.motion_path is not None:
        env_cfg.motion_path = os.path.abspath(args.motion_path)
    if args.task != "ballplay" and args.llc_checkpoint is not None:
        env_cfg.llc_checkpoint = os.path.abspath(args.llc_checkpoint)

    agent_path = os.path.join(PROJECT_ROOT, "skillmimic_lab", "agents", agent_filename)
    with open(agent_path, encoding="utf-8") as stream:
        agent_cfg = yaml.safe_load(stream)
    agent_cfg = copy.deepcopy(agent_cfg)
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["config"]["multi_gpu"] = args.distributed
    if args.max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = args.max_iterations
    if args.minibatch_size is not None:
        if args.minibatch_size <= 0:
            raise ValueError("--minibatch_size must be positive")
        agent_cfg["params"]["config"]["minibatch_size"] = args.minibatch_size
    horizon_length = agent_cfg["params"]["config"]["horizon_length"]
    batch_size = args.num_envs * horizon_length
    minibatch_size = agent_cfg["params"]["config"]["minibatch_size"]
    if batch_size % minibatch_size != 0:
        suggested_minibatch_size = args.num_envs * 8
        raise ValueError(
            f"RL-Games batch_size={batch_size} (num_envs={args.num_envs} * "
            f"horizon_length={horizon_length}) must be divisible by "
            f"minibatch_size={minibatch_size}. Try --minibatch_size "
            f"{suggested_minibatch_size}."
        )
    rl_device = device
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device

    log_root = os.path.join(PROJECT_ROOT, "logs", "rl_games", f"{args.task}_isaaclab")
    # The shell launcher supplies one name shared by all torchrun workers.
    run_name = os.environ.get("SKILLMIMIC_RUN_NAME") or datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    agent_cfg["params"]["config"]["train_dir"] = log_root
    agent_cfg["params"]["config"]["full_experiment_name"] = run_name
    if args.checkpoint is not None:
        checkpoint = os.path.abspath(args.checkpoint)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = checkpoint

    env = gym.make(task_id, cfg=env_cfg)
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = SkillMimicRlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    if args.distributed:
        print(
            f"[SkillMimic Lab] Distributed rank={app_launcher.global_rank} "
            f"local_rank={app_launcher.local_rank} world_size={os.environ.get('WORLD_SIZE', '1')} device={device}"
        )
    print(f"[SkillMimic Lab] Training task={args.task} envs={args.num_envs} logs={log_root}/{run_name}")
    runner = Runner(SkillMimicAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    try:
        runner.run({"train": True, "play": False})
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        _record_run_status(1)
        traceback.print_exc()
        raise
    else:
        _record_run_status(0)
    finally:
        simulation_app.close()
