"""CPU-only checks for migrated tensor/data/checkpoint compatibility."""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import gymnasium  # noqa: F401
except ImportError:
    # The legacy CPU test environment predates Gymnasium. Package registration
    # needs only these two attributes; simulator tests still require Isaac Lab.
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.registry = {}
    gymnasium.register = lambda **kwargs: gymnasium.registry.update({kwargs["id"]: kwargs})
    sys.modules["gymnasium"] = gymnasium

import torch

from skillmimic_lab.utils import torch_utils as math_utils
from skillmimic_lab.utils.motion_data_handler import MotionDataHandler
from skillmimic_lab.learning.policy import LegacyHLCPolicy, LegacySkillMimicPolicy
from skillmimic_lab.kit_runtime import configure_kit_runtime


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestIsaacLabCompatibility(unittest.TestCase):
    def test_kit_runtime_isolated_by_local_rank(self):
        with tempfile.TemporaryDirectory() as runtime_root, tempfile.TemporaryDirectory() as log_dir:
            environment = {
                "SKILLMIMIC_KIT_RUNTIME_ROOT": runtime_root,
                "SKILLMIMIC_KIT_LOG_DIR": log_dir,
                "LOCAL_RANK": "3",
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                sys, "argv", ["train.py"]
            ):
                worker_root = configure_kit_runtime(disable_ngx=True)
                self.assertEqual(worker_root, os.path.join(runtime_root, "rank-3"))
                self.assertTrue(os.path.isdir(worker_root))
                self.assertEqual(sys.argv[1:3], ["--portable-root", worker_root])
                self.assertEqual(sys.argv[3], "--/ngx/enabled=false")
                self.assertEqual(sys.argv[4], f"--/log/file={os.path.join(log_dir, 'rank-3.log')}")

    def test_quaternion_boundary_round_trip(self):
        quaternion = torch.nn.functional.normalize(torch.randn(128, 4), dim=-1)
        converted = math_utils.wxyz_to_xyzw(math_utils.xyzw_to_wxyz(quaternion))
        torch.testing.assert_close(converted, quaternion)

        vector = torch.randn(128, 3)
        identity = torch.zeros(128, 4)
        identity[:, 3] = 1.0
        torch.testing.assert_close(math_utils.quat_rotate(identity, vector), vector)

    def test_retarget_root_pose_and_velocity(self):
        batch = 64
        root_pos = torch.randn(batch, 3)
        root_rot = torch.nn.functional.normalize(torch.randn(batch, 4), dim=-1)
        root_to_link_pos = torch.randn(batch, 3)
        root_to_link_rot = torch.nn.functional.normalize(torch.randn(batch, 4), dim=-1)
        link_pos = root_pos + math_utils.quat_rotate(root_rot, root_to_link_pos)
        link_rot = math_utils.quat_mul(root_rot, root_to_link_rot)

        target_link_pos = torch.randn(batch, 3)
        target_link_rot = torch.nn.functional.normalize(torch.randn(batch, 4), dim=-1)
        target_root_pos, target_root_rot = math_utils.retarget_root_pose(
            root_pos,
            root_rot,
            link_pos,
            link_rot,
            target_link_pos,
            target_link_rot,
        )
        reconstructed_link_pos = target_root_pos + math_utils.quat_rotate(
            target_root_rot, root_to_link_pos
        )
        reconstructed_link_rot = math_utils.quat_mul(target_root_rot, root_to_link_rot)
        torch.testing.assert_close(reconstructed_link_pos, target_link_pos)
        rotation_agreement = torch.sum(reconstructed_link_rot * target_link_rot, dim=-1).abs()
        torch.testing.assert_close(rotation_agreement, torch.ones_like(rotation_agreement))

        root_lin_vel = torch.randn(batch, 3)
        root_ang_vel = torch.randn(batch, 3)
        relative_link_lin_vel = torch.randn(batch, 3)
        relative_link_ang_vel = torch.randn(batch, 3)
        root_to_link_world = link_pos - root_pos
        link_ang_vel = root_ang_vel + relative_link_ang_vel
        link_lin_vel = (
            root_lin_vel
            + torch.cross(root_ang_vel, root_to_link_world, dim=-1)
            + relative_link_lin_vel
        )
        target_link_lin_vel = torch.randn(batch, 3)
        target_link_ang_vel = torch.randn(batch, 3)
        target_root_lin_vel, target_root_ang_vel = math_utils.retarget_root_velocity(
            root_pos,
            root_lin_vel,
            root_ang_vel,
            link_pos,
            link_lin_vel,
            link_ang_vel,
            target_link_lin_vel,
            target_link_ang_vel,
        )
        reconstructed_link_ang_vel = target_root_ang_vel + relative_link_ang_vel
        reconstructed_link_lin_vel = (
            target_root_lin_vel
            + torch.cross(target_root_ang_vel, root_to_link_world, dim=-1)
            + relative_link_lin_vel
        )
        torch.testing.assert_close(reconstructed_link_ang_vel, target_link_ang_vel)
        torch.testing.assert_close(reconstructed_link_lin_vel, target_link_lin_vel)

    def test_spherical_dof_velocity_reconstructs_next_rotation(self):
        handler = MotionDataHandler.__new__(MotionDataHandler)
        handler.data_fps = 60.0
        exp_maps = torch.tensor(
            [
                [0.7, -0.2, 0.4, -0.1, 0.3, 0.2],
                [-0.4, 0.8, 0.1, 0.5, -0.2, 0.6],
            ]
        )

        velocity = handler._dof_angular_velocity(exp_maps)
        torch.testing.assert_close(velocity[0], torch.zeros_like(velocity[0]))

        first_quat = math_utils.exp_map_to_quat(exp_maps[0].reshape(-1, 3))
        delta_quat = math_utils.exp_map_to_quat(
            velocity[1].reshape(-1, 3) / handler.data_fps
        )
        reconstructed = math_utils.quat_mul(first_quat, delta_quat)
        expected = math_utils.exp_map_to_quat(exp_maps[1].reshape(-1, 3))
        # Quaternions q and -q encode the same rotation.
        agreement = torch.sum(reconstructed * expected, dim=-1).abs()
        torch.testing.assert_close(agreement, torch.ones_like(agreement), atol=1.0e-5, rtol=1.0e-5)

    def test_single_motion_layout(self):
        body_names = [
            "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
            "Torso", "Spine", "Spine2", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow",
            "L_Wrist", "L_Index1", "L_Index2", "L_Index3", "L_Middle1", "L_Middle2", "L_Middle3",
            "L_Pinky1", "L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2",
            "L_Thumb3", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Index1", "R_Index2", "R_Index3",
            "R_Middle1", "R_Middle2", "R_Middle3", "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1",
            "R_Ring2", "R_Ring3", "R_Thumb1", "R_Thumb2", "R_Thumb3",
        ]
        key_names = [
            "Head", "L_Knee", "R_Knee", "L_Elbow", "R_Elbow", "L_Ankle", "R_Ankle", "L_Index3",
            "L_Middle3", "L_Pinky3", "L_Ring3", "L_Thumb3", "R_Index3", "R_Middle3", "R_Pinky3",
            "R_Ring3", "R_Thumb3",
        ]
        key_ids = [body_names.index(name) for name in key_names]
        path = os.path.join(
            PROJECT_ROOT, "skillmimic", "data", "motions", "BallPlay-M", "run", "013_018pickle_run_3.pt"
        )
        handler = MotionDataHandler(path, "cpu", key_ids, 2, 60, {
            "p": 20.0, "r": 20.0, "pv": 0.0, "rv": 0.0, "op": 1.0, "or": 0.0,
            "opv": 0.0, "orv": 0.0, "ig": 20.0, "cg1": 5.0, "cg2": 5.0,
        })
        self.assertEqual(handler.hoi_data_dict[0]["hoi_data"].shape[1], 380)
        self.assertEqual(handler.hoi_data_dict[0]["dof_pos"].shape[1], 156)
        self.assertEqual(handler.hoi_data_dict[0]["body_pos"].shape[1:], (53, 3))

    def test_released_checkpoint_shapes(self):
        llc_path = os.path.join(
            PROJECT_ROOT, "skillmimic", "data", "models", "mixedskills", "nn", "skillmimic_llc.pth"
        )
        llc = LegacySkillMimicPolicy.from_checkpoint(llc_path, "cpu")
        action = llc(torch.zeros(2, 902))
        self.assertEqual(action.shape, (2, 156))
        self.assertTrue(torch.isfinite(action).all())

        tasks = {
            "hlc_circling": 843,
            "hlc_heading": 842,
            "hlc_throwing": 838,
            "hlc_scoring": 843,
        }
        for directory, observation_dim in tasks.items():
            checkpoint = os.path.join(PROJECT_ROOT, "skillmimic", "data", "models", directory, "nn", "SkillMimic.pth")
            policy = LegacyHLCPolicy.from_checkpoint(checkpoint, "cpu")
            action = policy(torch.zeros(2, observation_dim))
            self.assertEqual(action.shape, (2,))
            self.assertTrue(torch.all((action >= 0) & (action < policy.action_dim)))


if __name__ == "__main__":
    unittest.main()
