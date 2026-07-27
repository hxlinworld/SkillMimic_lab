"""Minimal Isaac Lab to RL-Games vector-environment adapter.

This project registers only its own DirectRLEnv tasks, so importing the full
``isaaclab_tasks`` package is unnecessary. That import
also eagerly imports every bundled task and all of their optional agent
libraries.  Keeping the small adapter local avoids unrelated dependencies such
as RSL-RL while preserving the interface expected by RL-Games 1.6.1.
"""

from __future__ import annotations

import gym as legacy_gym
import gymnasium as gym
import torch

from rl_games.common import env_configurations
from rl_games.common.vecenv import IVecEnv

from isaaclab.envs import DirectRLEnv


class RlGamesVecEnvWrapper(IVecEnv):
    """Expose an Isaac Lab ``DirectRLEnv`` through RL-Games' ``IVecEnv`` API."""

    def __init__(self, env, rl_device: str, clip_obs: float, clip_actions: float):
        if not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(f"Expected an Isaac Lab DirectRLEnv, got {type(env.unwrapped)}")
        self.env = env
        self._rl_device = rl_device
        self._sim_device = env.unwrapped.device
        self._clip_obs = clip_obs
        self._clip_actions = clip_actions
        self.rlg_num_states = 0 if self.state_space is None else self.state_space.shape[0]

    @property
    def render_mode(self):
        return self.env.render_mode

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def num_envs(self) -> int:
        return self.unwrapped.num_envs

    @property
    def device(self):
        return self.unwrapped.device

    @property
    def observation_space(self):
        policy_space = self.unwrapped.single_observation_space["policy"]
        if not isinstance(policy_space, gym.spaces.Box):
            raise NotImplementedError(f"Unsupported policy observation space: {type(policy_space)}")
        return legacy_gym.spaces.Box(-self._clip_obs, self._clip_obs, policy_space.shape)

    @property
    def action_space(self):
        action_space = self.unwrapped.single_action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise NotImplementedError(f"Unsupported action space: {type(action_space)}")
        return legacy_gym.spaces.Box(-self._clip_actions, self._clip_actions, action_space.shape)

    @property
    def state_space(self):
        state_space = self.unwrapped.single_observation_space.get("critic")
        if state_space is None:
            return None
        if not isinstance(state_space, gym.spaces.Box):
            raise NotImplementedError(f"Unsupported critic observation space: {type(state_space)}")
        return legacy_gym.spaces.Box(-self._clip_obs, self._clip_obs, state_space.shape)

    def get_number_of_agents(self) -> int:
        return getattr(self, "num_agents", 1)

    def get_env_info(self) -> dict:
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }

    def seed(self, seed: int = -1):
        return self.unwrapped.seed(seed)

    def reset(self):
        observations, _ = self.env.reset()
        return self._process_observations(observations)

    def step(self, actions):
        actions = actions.detach().clone().to(device=self._sim_device)
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated.to(device=self._rl_device)
        observations = self._process_observations(observations)
        rewards = rewards.to(device=self._rl_device)
        dones = (terminated | truncated).to(device=self._rl_device)
        extras = {
            key: value.to(device=self._rl_device, non_blocking=True) if hasattr(value, "to") else value
            for key, value in extras.items()
        }
        if "log" in extras:
            extras["episode"] = extras.pop("log")
        return observations, rewards, dones, extras

    def close(self):
        return self.env.close()

    def _process_observations(self, observations):
        policy = torch.clamp(observations["policy"], -self._clip_obs, self._clip_obs)
        policy = policy.to(device=self._rl_device).clone()
        if self.rlg_num_states == 0:
            return policy
        critic = torch.clamp(observations["critic"], -self._clip_obs, self._clip_obs)
        return {"obs": policy, "states": critic.to(device=self._rl_device).clone()}


class RlGamesGpuEnv(IVecEnv):
    """Return the already-created vector environment from RL-Games' registry."""

    def __init__(self, config_name: str, num_actors: int, **kwargs):
        del num_actors
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def get_number_of_agents(self) -> int:
        return self.env.get_number_of_agents()

    def get_env_info(self) -> dict:
        return self.env.get_env_info()
