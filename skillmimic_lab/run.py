#!/usr/bin/env python3
"""Run a smoke test or a legacy SkillMimic policy in Isaac Lab."""

from __future__ import annotations

import argparse
import os
import traceback

from isaaclab.app import AppLauncher

from skillmimic_lab.kit_runtime import configure_kit_runtime


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "skillmimic", "data", "models", "mixedskills", "nn", "skillmimic_llc.pth"
)
DEFAULT_MOTIONS = os.path.join(PROJECT_ROOT, "skillmimic", "data", "motions", "BallPlay-M")
DEFAULT_LLC_CHECKPOINT = DEFAULT_CHECKPOINT
HLC_CHECKPOINTS = {
    "circling": os.path.join(PROJECT_ROOT, "skillmimic", "data", "models", "hlc_circling", "nn", "SkillMimic.pth"),
    "heading": os.path.join(PROJECT_ROOT, "skillmimic", "data", "models", "hlc_heading", "nn", "SkillMimic.pth"),
    "throwing": os.path.join(PROJECT_ROOT, "skillmimic", "data", "models", "hlc_throwing", "nn", "SkillMimic.pth"),
    "scoring": os.path.join(PROJECT_ROOT, "skillmimic", "data", "models", "hlc_scoring", "nn", "SkillMimic.pth"),
}

parser = argparse.ArgumentParser(description="Run the Isaac Lab SkillMimic BallPlay migration")
parser.add_argument("--mode", choices=("smoke", "play", "reference"), default="smoke")
parser.add_argument("--task", choices=("ballplay", "circling", "heading", "throwing", "scoring"), default="ballplay")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--llc_checkpoint", default=DEFAULT_LLC_CHECKPOINT)
parser.add_argument("--motion_path", default=None)
parser.add_argument("--state_init", type=int, default=None, help="-1 for random, or a deterministic frame >= 2")
parser.add_argument("--episode_length", type=int, default=None, help="Legacy episode length in 60 Hz control steps")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

configure_kit_runtime(disable_ngx=args.headless)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _log_phase(message: str) -> None:
    print(f"[SkillMimic Lab] phase={message}", flush=True)


def _record_run_status(exit_code: int) -> None:
    status_file = os.environ.get("SKILLMIMIC_RUN_STATUS_FILE")
    if status_file is None:
        return
    try:
        with open(status_file, "w", encoding="utf-8") as stream:
            stream.write(f"{exit_code}\n")
    except OSError as exc:
        _log_phase(f"status_write_failed path={status_file!r} value={exc!r}")


try:
    _log_phase("simulation_app_ready; importing_torch")
    import torch

    _log_phase("torch_imported; importing_ballplay_task")
    from skillmimic_lab.env.tasks.skillmimic import SkillMimicBallPlayEnv, SkillMimicBallPlayEnvCfg

    _log_phase("ballplay_task_imported; importing_hrl_tasks")
    from skillmimic_lab.env.tasks.hrl_base import (
        SkillMimicCirclingEnvCfg,
        SkillMimicHLCEnv,
        SkillMimicHeadingEnvCfg,
        SkillMimicScoringEnvCfg,
        SkillMimicThrowingEnvCfg,
    )

    _log_phase("hrl_tasks_imported; importing_policies")
    from skillmimic_lab.learning.policy import LegacyHLCPolicy, LegacySkillMimicPolicy

    _log_phase("imports_complete")
except BaseException as exc:
    _log_phase(f"import_failed type={type(exc).__name__} value={exc!r}")
    traceback.print_exc()
    simulation_app.close()
    raise


HLC_CONFIGS = {
    "circling": SkillMimicCirclingEnvCfg,
    "heading": SkillMimicHeadingEnvCfg,
    "throwing": SkillMimicThrowingEnvCfg,
    "scoring": SkillMimicScoringEnvCfg,
}


def main() -> None:
    _log_phase("building_config")
    cfg = SkillMimicBallPlayEnvCfg() if args.task == "ballplay" else HLC_CONFIGS[args.task]()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if args.motion_path is not None:
        cfg.motion_path = os.path.abspath(args.motion_path)
    if args.state_init is not None:
        if args.state_init in (0, 1) or args.state_init < -1:
            raise ValueError("--state_init must be -1 or a frame index >= 2")
        cfg.state_init = args.state_init
    if args.episode_length is not None:
        if args.episode_length <= 0:
            raise ValueError("--episode_length must be positive")
        cfg.episode_length_s = args.episode_length / 60.0
    if args.task != "ballplay":
        if args.mode == "reference":
            raise ValueError("--mode reference is only available for the ballplay task")
        cfg.llc_checkpoint = os.path.abspath(args.llc_checkpoint)
    elif args.mode == "reference":
        # A fixed horizon makes the actuator probe comparable across runs and
        # lets it measure tracking after a fall instead of immediately sampling
        # a new pose. Natural motion timeouts still reset the environment.
        cfg.early_termination = False
    # Keep the environment tensors on the same GPU selected by AppLauncher for
    # PhysX and rendering.
    cfg.sim.device = args.device

    _log_phase(f"creating_environment task={args.task} num_envs={args.num_envs} device={cfg.sim.device}")
    env = SkillMimicBallPlayEnv(cfg=cfg) if args.task == "ballplay" else SkillMimicHLCEnv(cfg=cfg)
    _log_phase("environment_created")
    if args.mode == "reference":
        env.print_physics_parameter_report()
    policy = None
    if args.mode == "play":
        checkpoint = args.checkpoint or (DEFAULT_CHECKPOINT if args.task == "ballplay" else HLC_CHECKPOINTS[args.task])
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        if args.task == "ballplay":
            policy = LegacySkillMimicPolicy.from_checkpoint(checkpoint, str(env.device))
        else:
            policy = LegacyHLCPolicy.from_checkpoint(checkpoint, str(env.device))
        print(f"[SkillMimic Lab] Loaded legacy policy: {checkpoint}")

    try:
        _log_phase(f"resetting_environment seed={args.seed}")
        observations, _ = env.reset(seed=args.seed)
        _log_phase("environment_reset_complete; stepping")
        total_reward = torch.zeros(env.num_envs, device=env.device)
        tracking_squared_error = torch.zeros(156, device=env.device)
        tracking_absolute_error = torch.zeros(156, device=env.device)
        tracking_samples = 0
        completed_steps = 0
        while simulation_app.is_running() and completed_steps < args.steps:
            policy_obs = observations["policy"]
            if not torch.isfinite(policy_obs).all():
                raise FloatingPointError(f"Non-finite policy observation at step {completed_steps}")
            if args.mode == "reference":
                actions = env.reference_actions()
            elif policy is None:
                if args.task == "ballplay":
                    actions = torch.zeros((env.num_envs, 156), device=env.device)
                else:
                    actions = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
            else:
                actions = policy(policy_obs)
                if args.task == "ballplay":
                    actions = torch.clamp(actions, -1.0, 1.0)
            observations, reward, terminated, truncated, _ = env.step(actions)
            if not torch.isfinite(reward).all():
                raise FloatingPointError(f"Non-finite reward at step {completed_steps}")
            total_reward += reward
            completed_steps += 1
            if args.mode == "reference":
                tracking_error = env.reference_tracking_error()
                tracking_squared_error += tracking_error.square().sum(dim=0)
                tracking_absolute_error += tracking_error.abs().sum(dim=0)
                tracking_samples += tracking_error.shape[0]
            if completed_steps == 1 or completed_steps % 100 == 0:
                resets = int(torch.count_nonzero(terminated | truncated))
                print(
                    f"[SkillMimic Lab] step={completed_steps} "
                    f"reward_mean={reward.mean().item():.6f} resets={resets}"
                )
        if args.mode == "reference" and tracking_samples > 0:
            joint_rmse = torch.sqrt(tracking_squared_error / tracking_samples)
            joint_mae = tracking_absolute_error / tracking_samples
            overall_rmse = torch.sqrt(tracking_squared_error.sum() / (tracking_samples * 156))
            worst_ids = torch.topk(joint_rmse, k=10).indices
            joint_names = [
                env.robot.data.joint_names[env._joint_ids_legacy_order[index]]
                for index in worst_ids.cpu().tolist()
            ]
            worst = ", ".join(
                f"{name}:rmse={joint_rmse[index].item():.6f},mae={joint_mae[index].item():.6f}rad"
                for name, index in zip(joint_names, worst_ids.cpu().tolist())
            )
            print(
                "[SkillMimic Lab][reference-tracking] "
                f"samples={tracking_samples} overall_rmse={overall_rmse.item():.6f}rad "
                f"worst=[{worst}]",
                flush=True,
            )
        print(
            f"[SkillMimic Lab] PASS mode={args.mode} task={args.task} "
            f"envs={env.num_envs} steps={completed_steps} "
            f"mean_return={total_reward.mean().item():.6f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        _log_phase("entering_main")
        main()
        _log_phase("main_complete")
    except BaseException as exc:
        _record_run_status(1)
        _log_phase(f"main_failed type={type(exc).__name__} value={exc!r}")
        traceback.print_exc()
        raise
    else:
        _record_run_status(0)
    finally:
        _log_phase("closing_simulation_app")
        simulation_app.close()
