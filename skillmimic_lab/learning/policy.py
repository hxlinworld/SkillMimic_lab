"""Policy loaders for legacy SkillMimic RL-Games checkpoints."""

from __future__ import annotations

import torch
from torch import nn


class LegacySkillMimicPolicy(nn.Module):
    observation_dim = 902
    action_dim = 156

    def __init__(self) -> None:
        super().__init__()
        self.actor_mlp = nn.Sequential(
            nn.Linear(self.observation_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.mu = nn.Linear(512, self.action_dim)
        self.register_buffer("running_mean", torch.zeros(self.observation_dim))
        self.register_buffer("running_var", torch.ones(self.observation_dim))

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str) -> "LegacySkillMimicPolicy":
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint.get("model", checkpoint)
        policy = cls().to(device)
        own_state = policy.state_dict()
        mapping = {
            "actor_mlp.0.weight": "a2c_network.actor_mlp.0.weight",
            "actor_mlp.0.bias": "a2c_network.actor_mlp.0.bias",
            "actor_mlp.2.weight": "a2c_network.actor_mlp.2.weight",
            "actor_mlp.2.bias": "a2c_network.actor_mlp.2.bias",
            "actor_mlp.4.weight": "a2c_network.actor_mlp.4.weight",
            "actor_mlp.4.bias": "a2c_network.actor_mlp.4.bias",
            "mu.weight": "a2c_network.mu.weight",
            "mu.bias": "a2c_network.mu.bias",
        }
        for destination, source in mapping.items():
            if source not in model_state:
                raise KeyError(f"Legacy checkpoint is missing {source}")
            own_state[destination].copy_(model_state[source])

        normalizer = checkpoint.get("running_mean_std")
        if normalizer is None:
            raise KeyError("Legacy checkpoint has no running_mean_std")
        policy.running_mean.copy_(normalizer["running_mean"].float())
        policy.running_var.copy_(normalizer["running_var"].float())
        policy.eval()
        return policy

    @torch.inference_mode()
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - self.running_mean) / torch.sqrt(self.running_var + 1.0e-5)
        normalized = torch.clamp(normalized, -5.0, 5.0)
        return self.mu(self.actor_mlp(normalized))


class LegacyHLCPolicy(nn.Module):
    """Deterministic discrete actor for the four released HLC checkpoints."""

    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.actor_mlp = nn.Sequential(
            nn.Linear(observation_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.logits = nn.Linear(512, action_dim)
        self.register_buffer("running_mean", torch.zeros(observation_dim))
        self.register_buffer("running_var", torch.ones(observation_dim))

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str) -> "LegacyHLCPolicy":
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint.get("model", checkpoint)
        first_weight = model_state["a2c_network.actor_mlp.0.weight"]
        logits_weight = model_state["a2c_network.logits.weight"]
        policy = cls(first_weight.shape[1], logits_weight.shape[0]).to(device)
        own_state = policy.state_dict()
        mapping = {
            "actor_mlp.0.weight": "a2c_network.actor_mlp.0.weight",
            "actor_mlp.0.bias": "a2c_network.actor_mlp.0.bias",
            "actor_mlp.2.weight": "a2c_network.actor_mlp.2.weight",
            "actor_mlp.2.bias": "a2c_network.actor_mlp.2.bias",
            "logits.weight": "a2c_network.logits.weight",
            "logits.bias": "a2c_network.logits.bias",
        }
        for destination, source in mapping.items():
            own_state[destination].copy_(model_state[source])
        normalizer = checkpoint["running_mean_std"]
        policy.running_mean.copy_(normalizer["running_mean"].float())
        policy.running_var.copy_(normalizer["running_var"].float())
        policy.eval()
        return policy

    @torch.inference_mode()
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"HLC checkpoint expects {self.observation_dim} observations, got {observation.shape[-1]}"
            )
        normalized = (observation - self.running_mean) / torch.sqrt(self.running_var + 1.0e-5)
        normalized = torch.clamp(normalized, -5.0, 5.0)
        return torch.argmax(self.logits(self.actor_mlp(normalized)), dim=-1)
