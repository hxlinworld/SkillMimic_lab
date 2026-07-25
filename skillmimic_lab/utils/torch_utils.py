"""Legacy quaternion helpers preserving the original XYZW convention.

Isaac Lab exposes quaternions as WXYZ. SkillMimic motion data and checkpoints
were produced with Isaac Gym's XYZW tensors, so conversion is intentionally
limited to the simulator boundary.
"""

from __future__ import annotations

import torch


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 1:4], q[..., 0:1]), dim=-1)


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 3:4], q[..., 0:3]), dim=-1)


def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[..., :3]
    uv = torch.cross(q_vec, v, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return v + 2.0 * (q[..., 3:4] * uv + uuv)


def retarget_root_pose(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    link_pos: torch.Tensor,
    link_rot: torch.Tensor,
    target_link_pos: torch.Tensor,
    target_link_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move an articulation root so one of its links reaches a target pose.

    All quaternions use the legacy XYZW convention.  The root-to-link
    transform is measured from the current kinematic state, so this also works
    when an importer chooses a different floating link than the source asset.
    """

    root_rot_inv = quat_conjugate(root_rot)
    root_to_link_pos = quat_rotate(root_rot_inv, link_pos - root_pos)
    root_to_link_rot = quat_mul(root_rot_inv, link_rot)
    target_root_rot = quat_mul(target_link_rot, quat_conjugate(root_to_link_rot))
    target_root_pos = target_link_pos - quat_rotate(target_root_rot, root_to_link_pos)
    return target_root_pos, target_root_rot


def retarget_root_velocity(
    root_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    link_pos: torch.Tensor,
    link_lin_vel: torch.Tensor,
    link_ang_vel: torch.Tensor,
    target_link_lin_vel: torch.Tensor,
    target_link_ang_vel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjust a root twist so a link reaches a target world-frame twist."""

    angular_correction = target_link_ang_vel - link_ang_vel
    target_root_ang_vel = root_ang_vel + angular_correction
    target_root_lin_vel = (
        root_lin_vel
        + target_link_lin_vel
        - link_lin_vel
        - torch.cross(angular_correction, link_pos - root_pos, dim=-1)
    )
    return target_root_lin_vel, target_root_ang_vel


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    half_angle = 0.5 * angle
    xyz = axis * torch.sin(half_angle).unsqueeze(-1)
    return torch.cat((xyz, torch.cos(half_angle).unsqueeze(-1)), dim=-1)


def exp_map_to_quat(exp_map: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(exp_map, dim=-1)
    safe_angle = torch.clamp(angle, min=1.0e-8)
    axis = exp_map / safe_angle.unsqueeze(-1)
    default_axis = torch.zeros_like(axis)
    default_axis[..., 2] = 1.0
    axis = torch.where((angle > 1.0e-5).unsqueeze(-1), axis, default_axis)
    clean_angle = torch.where(angle > 1.0e-5, normalize_angle(angle), torch.zeros_like(angle))
    return quat_from_angle_axis(clean_angle, axis)


def quat_to_exp_map(q: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(q, dim=-1)
    sin_theta = torch.sqrt(torch.clamp(1.0 - q[..., 3].square(), min=0.0))
    angle = normalize_angle(2.0 * torch.acos(torch.clamp(q[..., 3], -1.0, 1.0)))
    axis = q[..., :3] / torch.clamp(sin_theta, min=1.0e-8).unsqueeze(-1)
    default_axis = torch.zeros_like(axis)
    default_axis[..., 2] = 1.0
    valid = sin_theta.abs() > 1.0e-5
    axis = torch.where(valid.unsqueeze(-1), axis, default_axis)
    angle = torch.where(valid, angle, torch.zeros_like(angle))
    return angle.unsqueeze(-1) * axis


def calc_heading(q: torch.Tensor) -> torch.Tensor:
    ref_dir = torch.zeros_like(q[..., :3])
    ref_dir[..., 0] = 1.0
    rot_dir = quat_rotate(q, ref_dir)
    return torch.atan2(rot_dir[..., 1], rot_dir[..., 0])


def calc_heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
    axis = torch.zeros_like(q[..., :3])
    axis[..., 2] = 1.0
    return quat_from_angle_axis(-calc_heading(q), axis)


def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    tangent = torch.zeros_like(q[..., :3])
    tangent[..., 0] = 1.0
    normal = torch.zeros_like(q[..., :3])
    normal[..., 2] = 1.0
    return torch.cat((quat_rotate(q, tangent), quat_rotate(q, normal)), dim=-1)
