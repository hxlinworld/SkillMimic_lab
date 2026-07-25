"""Isaac Lab ports of SkillMimic's four high-level controller tasks."""

from __future__ import annotations

import os

import gymnasium as gym
import torch

from omni.isaac.lab.utils import configclass

from skillmimic_lab.utils import torch_utils as math_utils
from .skillmimic import (
    PROJECT_ROOT,
    SkillMimicBallPlayEnv,
    SkillMimicBallPlayEnvCfg,
    compute_humanoid_observations,
    compute_object_observations,
)
from skillmimic_lab.learning.policy import LegacySkillMimicPolicy


LLC_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "skillmimic", "data", "models", "mixedskills", "nn", "skillmimic_llc.pth"
)
RUN_MOTIONS = os.path.join(PROJECT_ROOT, "skillmimic", "data", "motions", "BallPlay-M", "run")
THROW_MOTIONS = os.path.join(PROJECT_ROOT, "skillmimic", "data", "motions", "BallPlay-M", "turnhook")


@configclass
class SkillMimicHLCEnvCfg(SkillMimicBallPlayEnvCfg):
    episode_length_s = 800.0 / 60.0
    decimation = 3
    num_observations = 838
    num_actions = 3
    state_init = 2
    task_name = "throwing"
    task_observation_size = 0
    control_mapping = (34, 1, 13)
    llc_checkpoint = LLC_CHECKPOINT
    motion_path = THROW_MOTIONS
    # Configclass turns mutable defaults into instance factories, so the
    # parent's simulation config is available on an instance, not the class.
    sim = SkillMimicBallPlayEnvCfg().sim.replace(render_interval=decimation)


@configclass
class SkillMimicCirclingEnvCfg(SkillMimicHLCEnvCfg):
    num_observations = 843
    num_actions = 3
    task_name = "circling"
    task_observation_size = 5
    control_mapping = (12, 13, 11)
    motion_path = RUN_MOTIONS


@configclass
class SkillMimicHeadingEnvCfg(SkillMimicHLCEnvCfg):
    num_observations = 842
    num_actions = 3
    task_name = "heading"
    task_observation_size = 4
    control_mapping = (12, 13, 11)
    motion_path = RUN_MOTIONS


@configclass
class SkillMimicThrowingEnvCfg(SkillMimicHLCEnvCfg):
    num_observations = 838
    num_actions = 3
    task_name = "throwing"
    task_observation_size = 0
    control_mapping = (34, 1, 13)
    motion_path = THROW_MOTIONS


@configclass
class SkillMimicScoringEnvCfg(SkillMimicHLCEnvCfg):
    num_observations = 843
    num_actions = 7
    task_name = "scoring"
    task_observation_size = 5
    control_mapping = (31, 1, 2, 12, 13, 11, 31)
    motion_path = RUN_MOTIONS


class SkillMimicHLCEnv(SkillMimicBallPlayEnv):
    cfg: SkillMimicHLCEnvCfg

    def __init__(self, cfg: SkillMimicHLCEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if not os.path.isfile(self.cfg.llc_checkpoint):
            raise FileNotFoundError(f"LLC checkpoint does not exist: {self.cfg.llc_checkpoint}")
        self.llc_policy = LegacySkillMimicPolicy.from_checkpoint(self.cfg.llc_checkpoint, str(self.device))
        self.control_mapping = torch.tensor(self.cfg.control_mapping, device=self.device, dtype=torch.long)
        self.high_level_actions = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.goal_position = torch.zeros((self.num_envs, 2), device=self.device)
        self.goal_radius = torch.zeros((self.num_envs, 1), device=self.device)
        self.reached_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _configure_gym_env_spaces(self) -> None:
        """Expose the legacy HLC as a discrete space on Isaac Lab 1.1."""

        super()._configure_gym_env_spaces()
        self.single_action_space = gym.spaces.Discrete(len(self.cfg.control_mapping))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _physical_observations(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        body_pos, body_rot, body_vel, body_ang_vel = self._local_body_state()
        humanoid = compute_humanoid_observations(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            self.robot_contact_sensor.data.net_forces_w,
            self._contact_obs_ids,
        )
        ball_state = self._local_ball_state()
        object_obs = compute_object_observations(body_pos[:, 0], body_rot[:, 0], ball_state)
        return torch.cat((humanoid, object_obs), dim=-1), body_pos, body_rot, ball_state

    def _get_observations(self) -> dict[str, torch.Tensor]:
        physical, body_pos, body_rot, _ = self._physical_observations()
        if self.cfg.task_name == "circling":
            task = compute_goal_observations(
                body_pos[:, 0], body_rot[:, 0], self.goal_position, self.goal_radius
            )
        elif self.cfg.task_name == "heading":
            task = compute_goal_observations(body_pos[:, 0], body_rot[:, 0], self.goal_position)
        elif self.cfg.task_name == "scoring":
            task = compute_goal_observations(
                body_pos[:, 0], body_rot[:, 0], self.goal_position, self.reached_target.float().unsqueeze(-1)
            )
        else:
            task = physical[:, :0]
        observation = torch.cat((physical, task), dim=-1)
        if observation.shape[-1] != self.cfg.num_observations:
            raise RuntimeError(
                f"{self.cfg.task_name} observation must be {self.cfg.num_observations}-D, got {observation.shape}"
            )
        return {"policy": observation}

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.high_level_actions = actions.long().reshape(-1)
        if torch.any(self.high_level_actions < 0) or torch.any(self.high_level_actions >= len(self.control_mapping)):
            raise ValueError(f"HLC action is outside [0, {len(self.control_mapping) - 1}]")

    def _apply_action(self) -> None:
        physical, _, _, _ = self._physical_observations()
        skill_ids = self.control_mapping[self.high_level_actions]
        condition = torch.zeros((self.num_envs, 64), device=self.device)
        condition[torch.arange(self.num_envs, device=self.device), skill_ids] = 1.0
        llc_observation = torch.cat((physical, condition), dim=-1)
        llc_actions = torch.clamp(self.llc_policy(llc_observation), -1.0, 1.0)
        targets = self.action_offset + self.action_scale * llc_actions
        self.robot.set_joint_position_target(targets)

    def _get_rewards(self) -> torch.Tensor:
        _, body_pos, _, ball_state = self._physical_observations()
        root_pos = body_pos[:, 0]
        root_vel = self.robot.data.body_lin_vel_w[:, self._legacy_root_body_id]
        ball_pos = ball_state[:, :3]
        ball_vel = ball_state[:, 7:10]

        if self.cfg.task_name == "circling":
            distance = torch.linalg.vector_norm(ball_pos[:, :2] - self.goal_position, dim=-1)
            radial_error = torch.linalg.vector_norm(distance.unsqueeze(-1) - self.goal_radius, dim=-1)
            speed_penalty = torch.where(torch.linalg.vector_norm(ball_vel, dim=-1) < 0.5, 0.1, 1.0)
            reward = torch.exp(-radial_error) * speed_penalty
            self.reached_target |= distance < 0.3
        elif self.cfg.task_name == "heading":
            goal_3d = torch.cat((self.goal_position, torch.ones_like(self.goal_position[:, :1])), dim=-1)
            distance = torch.linalg.vector_norm(ball_pos - goal_3d, dim=-1)
            reward = torch.exp(-distance)
            self.reached_target |= distance < 0.5
        elif self.cfg.task_name == "throwing":
            reward = torch.exp(-torch.abs(ball_pos[:, 2] - 2.5))
        elif self.cfg.task_name == "scoring":
            goal_3d = torch.cat((self.goal_position, torch.full_like(self.goal_position[:, :1], 2.5)), dim=-1)
            distance = torch.linalg.vector_norm(ball_pos - goal_3d, dim=-1)
            landing_xy = calculate_landing_position(ball_vel, ball_pos, 2.0)
            landing_distance = torch.linalg.vector_norm(landing_xy - self.goal_position, dim=-1)
            ball_contact = torch.any(torch.abs(self.ball_contact_sensor.data.net_forces_w[:, 0]) > 0.1, dim=-1)
            self.reached_target |= (landing_distance < 0.3) & (ball_pos[:, 2] > 2.0) & (~ball_contact)
            reached_reward = self.reached_target.float()
            position_reward = torch.exp(-0.5 * distance)
            height_reward = torch.exp(-torch.abs(ball_pos[:, 2] - 2.5))
            speed_penalty = torch.where(torch.linalg.vector_norm(root_vel, dim=-1) < 0.5, 0.1, 1.0)
            reward = speed_penalty * (position_reward + reached_reward + 0.2 * height_reward)
        else:
            raise ValueError(f"Unknown HLC task: {self.cfg.task_name}")
        self.extras[f"skillmimic/{self.cfg.task_name}_reward"] = reward.mean()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_height = self.robot.data.body_pos_w[:, self._legacy_root_body_id, 2]
        if self.cfg.early_termination:
            terminated = (root_height < self.cfg.termination_height) & (self.episode_length_buf > 1)
        else:
            terminated = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        count = len(env_ids)
        root_xy = (
            self.robot.data.body_pos_w[env_ids, self._legacy_root_body_id, :2]
            - self.scene.env_origins[env_ids, :2]
        )
        if self.cfg.task_name == "circling":
            self.goal_position[env_ids] = root_xy
            self.goal_radius[env_ids] = 2.0 + 3.0 * torch.rand((count, 1), device=self.device)
        elif self.cfg.task_name == "heading":
            self.goal_position[env_ids] = root_xy + 2.0 + 6.0 * torch.rand((count, 2), device=self.device)
        elif self.cfg.task_name == "scoring":
            distance = 2.0 + 6.0 * torch.rand(count, device=self.device)
            angle = 2.0 * torch.pi * torch.rand(count, device=self.device)
            offset = torch.stack((torch.sin(angle) * distance, torch.cos(angle) * distance), dim=-1)
            self.goal_position[env_ids] = root_xy + offset
        self.reached_target[env_ids] = False
        self.high_level_actions[env_ids] = 0


def compute_goal_observations(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    goal_pos: torch.Tensor,
    extra: torch.Tensor | None = None,
) -> torch.Tensor:
    heading_inv = math_utils.calc_heading_quat_inv(root_rot)
    facing = torch.zeros_like(root_pos)
    facing[:, 0] = 1.0
    local_facing = math_utils.quat_rotate(heading_inv, facing)
    target_delta = goal_pos - root_pos[:, :2]
    target_delta_3d = torch.cat((target_delta, torch.zeros_like(target_delta[:, :1])), dim=-1)
    local_target = math_utils.quat_rotate(heading_inv, target_delta_3d)[:, :2]
    angle = torch.atan2(local_target[:, 1], local_target[:, 0]) - torch.atan2(
        local_facing[:, 1], local_facing[:, 0]
    )
    angle = math_utils.normalize_angle(angle)
    result = torch.cat((local_target, torch.cos(angle).unsqueeze(-1), torch.sin(angle).unsqueeze(-1)), dim=-1)
    return torch.cat((result, extra), dim=-1) if extra is not None else result


def calculate_landing_position(velocity: torch.Tensor, position: torch.Tensor, height: float) -> torch.Tensor:
    gravity = 9.8
    max_height = position[:, 2] + velocity[:, 2].square() / (2.0 * gravity)
    target_height = torch.where(max_height < height, torch.zeros_like(max_height), height)
    time = (
        torch.sqrt(torch.clamp(velocity[:, 2].square() + 2.0 * gravity * (position[:, 2] - target_height), min=0.0))
        - velocity[:, 2]
    ) / gravity
    return position[:, :2] + velocity[:, :2] * time.unsqueeze(-1)
