"""Loader for the original SkillMimic HOI tensors."""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from . import torch_utils as math_utils


class MotionDataHandler:
    """Loads legacy `.pt` motions without importing Isaac Gym."""

    def __init__(
        self,
        motion_path: str,
        device: str,
        key_body_ids: list[int],
        num_envs: int,
        max_episode_length: int,
        reward_weights: dict[str, float],
        data_fps: float = 60.0,
        data_frames_scale: float = 1.0,
        init_vel: bool = False,
    ):
        self.device = torch.device(device)
        self.key_body_ids = key_body_ids
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.data_fps = data_fps * data_frames_scale
        self.init_vel = init_vel
        self.hoi_data_dict: dict[int, dict[str, torch.Tensor | str]] = {}

        self._load_motion(motion_path)
        self.envid2motid = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.envid2startframe = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.envid2episode_lengths = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.reward_weights = {
            name: torch.full((num_envs,), float(value), device=self.device)
            for name, value in reward_weights.items()
        }
        self.reward_weights_default = reward_weights

    def _load_motion(self, motion_path: str) -> None:
        paths = [motion_path] if os.path.isfile(motion_path) else glob.glob(
            os.path.join(motion_path, "**", "*.pt"), recursive=True
        )
        paths.sort(key=self._sort_key)
        if not paths:
            raise FileNotFoundError(f"No SkillMimic motion .pt files found under: {motion_path}")
        self.motion_paths = paths

        self.motion_lengths = torch.zeros(len(paths), device=self.device, dtype=torch.long)
        self.motion_class = torch.zeros(len(paths), device=self.device, dtype=torch.long)
        for motion_id, path in enumerate(paths):
            sequence = self._process_sequence(path)
            self.hoi_data_dict[motion_id] = sequence
            self.motion_lengths[motion_id] = sequence["hoi_data"].shape[0]
            self.motion_class[motion_id] = int(sequence["hoi_data_text"])
        self._compute_motion_weights()
        print(f"[SkillMimic Lab] Loaded {len(paths)} motions from {motion_path}")

    @staticmethod
    def _sort_key(filename: str) -> int:
        match = re.search(r"\d+\.pt$", filename)
        return int(match.group()[:-3]) if match else -1

    def _process_sequence(self, path: str) -> dict[str, torch.Tensor | str]:
        raw = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(raw, torch.Tensor) or raw.ndim != 2 or raw.shape[1] < 337:
            raise ValueError(f"Unexpected motion tensor in {path}: {getattr(raw, 'shape', type(raw))}")
        raw = raw.detach().float()
        frames = raw.shape[0]

        root_pos = raw[:, 0:3].clone()
        root_rot_3d = raw[:, 3:6].clone()
        root_rot = math_utils.exp_map_to_quat(root_rot_3d)
        self._smooth_quat_seq(root_rot)
        q_diff = math_utils.quat_mul(math_utils.quat_conjugate(root_rot[:-1]), root_rot[1:])
        root_rot_vel = self._prepend_zero(math_utils.quat_to_exp_map(q_diff) * self.data_fps)
        dof_pos = raw[:, 9:165].clone()
        body_pos = raw[:, 165 : 165 + 53 * 3].clone().view(frames, 53, 3)
        obj_pos = raw[:, 324:327].clone()
        # Preserve the legacy SkillMimic convention: object angular velocity is
        # the finite difference of the 3-D exponential-map rotation, while the
        # simulator pose receives the corresponding 4-D quaternion.
        obj_rot_exp_map = -raw[:, 327:330].clone()
        obj_rot = math_utils.exp_map_to_quat(obj_rot_exp_map).clone()
        obj_pos_vel = self._velocity(obj_pos)
        if self.init_vel and frames > 1:
            obj_pos_vel[0] = obj_pos_vel[1]
        key_body_pos = body_pos[:, self.key_body_ids].reshape(frames, -1).clone()
        contact = torch.round(raw[:, 336:337].clone())

        result = {
            "hoi_data_text": os.path.basename(path)[:3],
            "root_pos": root_pos,
            "root_rot": root_rot,
            "root_pos_vel": self._velocity(root_pos),
            "root_rot_vel": root_rot_vel,
            "dof_pos": dof_pos,
            # Isaac Gym stores every three rotational DOFs of this humanoid as
            # one exponential-map rotation.  PhysX spherical-joint velocity is
            # therefore the logarithm of the relative quaternion, not a
            # component-wise finite difference of the exponential map.
            "dof_pos_vel": self._dof_angular_velocity(dof_pos),
            "body_pos": body_pos,
            "key_body_pos": key_body_pos,
            "obj_pos": obj_pos,
            "obj_pos_vel": obj_pos_vel,
            "obj_rot": obj_rot,
            "obj_rot_vel": self._velocity(obj_rot_exp_map),
            "contact": contact,
        }
        result["hoi_data"] = torch.cat(
            (
                root_pos,
                root_rot_3d,
                dof_pos,
                result["dof_pos_vel"],
                obj_pos,
                obj_rot,
                obj_pos_vel,
                key_body_pos,
                contact,
            ),
            dim=-1,
        )
        if result["hoi_data"].shape[1] != 380:
            raise RuntimeError(f"Legacy HOI observation must be 380-D, got {result['hoi_data'].shape[1]}")
        return result

    def _velocity(self, values: torch.Tensor) -> torch.Tensor:
        return self._prepend_zero((values[1:] - values[:-1]) * self.data_fps)

    def _dof_angular_velocity(self, exp_maps: torch.Tensor) -> torch.Tensor:
        if exp_maps.shape[-1] % 3 != 0:
            raise ValueError(
                f"Rotational DOFs must be grouped in threes, got {exp_maps.shape}"
            )
        rotations = exp_maps.reshape(exp_maps.shape[0], -1, 3)
        quaternions = math_utils.exp_map_to_quat(rotations)
        relative = math_utils.quat_mul(
            math_utils.quat_conjugate(quaternions[:-1]), quaternions[1:]
        )
        angular_velocity = math_utils.quat_to_exp_map(relative) * self.data_fps
        angular_velocity = self._prepend_zero(angular_velocity)
        return angular_velocity.reshape_as(exp_maps)

    @staticmethod
    def _prepend_zero(values: torch.Tensor) -> torch.Tensor:
        return torch.cat((torch.zeros_like(values[:1]), values), dim=0)

    @staticmethod
    def _smooth_quat_seq(quaternions: torch.Tensor) -> None:
        for index in range(1, quaternions.shape[0]):
            if torch.dot(quaternions[index - 1], quaternions[index]) < 0:
                quaternions[index] *= -1

    def _compute_motion_weights(self) -> None:
        classes = self.motion_class.cpu().numpy()
        unique, counts = np.unique(classes, return_counts=True)
        class_weights = {int(label): 1.0 / count for label, count in zip(unique, counts)}
        if 1 in class_weights:
            class_weights[1] *= 2.0
        weights = [class_weights[int(label)] for label in classes]
        self.motion_weights = torch.tensor(weights, device=self.device, dtype=torch.float)

    def sample_motions(self, count: int) -> torch.Tensor:
        return torch.multinomial(self.motion_weights, num_samples=count, replacement=True)

    def sample_time(self, motion_ids: torch.Tensor, deterministic_frame: int | None = None) -> torch.Tensor:
        if deterministic_frame is not None:
            frames = torch.full_like(motion_ids, deterministic_frame)
            return torch.minimum(frames, self.motion_lengths[motion_ids] - 2)
        start = torch.full_like(motion_ids, 2)
        end = self.motion_lengths[motion_ids] - 2
        if torch.any(end <= start):
            raise ValueError("Every motion must contain at least five frames")
        random = torch.rand(motion_ids.shape, device=self.device)
        return start + torch.floor(random * (end - start + 1)).long()

    def get_initial_state(self, env_ids: torch.Tensor, motion_ids: torch.Tensor, start_frames: torch.Tensor):
        valid_lengths = self.motion_lengths[motion_ids] - start_frames
        self.envid2episode_lengths[env_ids] = torch.minimum(
            valid_lengths, torch.full_like(valid_lengths, self.max_episode_length)
        )

        fields: dict[str, list[torch.Tensor]] = {
            key: []
            for key in (
                "hoi_data", "root_pos", "root_rot", "root_pos_vel", "root_rot_vel",
                "dof_pos", "dof_pos_vel", "obj_pos", "obj_pos_vel", "obj_rot", "obj_rot_vel",
            )
        }
        for row, env_id in enumerate(env_ids.tolist()):
            motion_id = int(motion_ids[row])
            start = int(start_frames[row])
            length = int(self.envid2episode_lengths[env_id])
            motion = self.hoi_data_dict[motion_id]
            self.envid2motid[env_id] = motion_id
            self.envid2startframe[env_id] = start

            reference = F.pad(motion["hoi_data"][start : start + length], (0, 0, 0, self.max_episode_length - length))
            fields["hoi_data"].append(reference)
            for name in fields:
                if name != "hoi_data":
                    fields[name].append(motion[name][start])

            if motion["hoi_data_text"] == "000":
                fields["obj_pos"][-1] = torch.rand(3, device=self.device) * 10.0 - 5.0
                fields["obj_pos_vel"][-1] = torch.rand(3, device=self.device) * 5.0
                random_quat = torch.randn(4, device=self.device)
                fields["obj_rot"][-1] = torch.nn.functional.normalize(random_quat, dim=0)
                fields["obj_rot_vel"][-1] = torch.rand(3, device=self.device) * 0.1
                for key in ("op", "ig", "cg1", "cg2"):
                    self.reward_weights[key][env_id] = 0.0
            else:
                for key, value in self.reward_weights_default.items():
                    self.reward_weights[key][env_id] = float(value)

        return tuple(torch.stack(fields[name], dim=0) for name in fields)
