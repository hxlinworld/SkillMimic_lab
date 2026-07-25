"""Isaac Lab DirectRLEnv port of the SkillMimic BallPlay task."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import torch
import torch.nn.functional as F

import omni.kit.commands
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.utils import configclass

from skillmimic_lab.utils import torch_utils as math_utils
from skillmimic_lab.utils.motion_data_handler import MotionDataHandler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HUMANOID_MJCF = os.path.join(PROJECT_ROOT, "skillmimic", "data", "assets", "mjcf", "mocap_humanoid.xml")
DEFAULT_MOTION_PATH = os.path.join(PROJECT_ROOT, "skillmimic", "data", "motions", "BallPlay-M")
ROBOT_SOURCE_PRIM_PATH = "/World/envs/env_0/Robot"

KEY_BODY_NAMES = [
    "Head", "L_Knee", "R_Knee", "L_Elbow", "R_Elbow", "L_Ankle", "R_Ankle",
    "L_Index3", "L_Middle3", "L_Pinky3", "L_Ring3", "L_Thumb3",
    "R_Index3", "R_Middle3", "R_Pinky3", "R_Ring3", "R_Thumb3",
]
FINGERTIP_NAMES = [
    "L_Index3", "L_Middle3", "L_Pinky3", "L_Ring3", "L_Thumb3",
    "R_Index3", "R_Middle3", "R_Pinky3", "R_Ring3", "R_Thumb3",
]
UNDESIRED_CONTACT_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "R_Hip", "R_Knee", "Torso", "Spine", "Spine2",
    "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "R_Thorax",
    "R_Shoulder", "R_Elbow",
]

REWARD_WEIGHTS = {
    "p": 20.0,
    "r": 20.0,
    "pv": 0.0,
    "rv": 0.0,
    "op": 1.0,
    "or": 0.0,
    "opv": 0.0,
    "orv": 0.0,
    "ig": 20.0,
    "cg1": 5.0,
    "cg2": 5.0,
}


def _mjcf_names(path: str, tag: str) -> list[str]:
    return [element.attrib["name"] for element in ET.parse(path).iter(tag) if "name" in element.attrib]


def _import_humanoid_mjcf(path: str, prim_path: str) -> None:
    """Import the humanoid with Isaac Sim 4.1's bundled MJCF importer.

    Isaac Lab 1.1 predates ``MjcfFileCfg``.  Importing into the source
    environment before cloning provides the equivalent behavior without a
    generated USD file in the repository.
    """

    enable_extension("omni.importer.mjcf")
    status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
    if not status or import_config is None:
        raise RuntimeError("Isaac Sim 4.1 could not create the MJCF import configuration")

    import_config.set_fix_base(False)
    import_config.set_make_default_prim(False)
    import_config.set_create_physics_scene(False)
    import_config.set_import_inertia_tensor(True)
    import_config.set_import_sites(False)
    import_config.set_self_collision(True)
    import_config.set_merge_fixed_joints(False)

    status, _ = omni.kit.commands.execute(
        "MJCFCreateAsset",
        mjcf_path=path,
        import_config=import_config,
        prim_path=prim_path,
    )
    if not status:
        raise RuntimeError(f"Isaac Sim 4.1 failed to import the humanoid MJCF: {path}")

    # The source MJCF contains a world-level floor geom. The 4.1 importer turns
    # it into a second articulation named ``worldBody`` next to the humanoid.
    # We provide one global local ground below, so remove this duplicate before
    # cloning environments and target only the Pelvis articulation.
    stage = omni.usd.get_context().get_stage()
    stage.RemovePrim(f"{prim_path}/worldBody")
    articulation_prim_path = f"{prim_path}/Pelvis"

    sim_utils.modify_rigid_body_properties(
        articulation_prim_path,
        sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            angular_damping=0.01,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
        ),
    )
    sim_utils.modify_articulation_root_properties(
        articulation_prim_path,
        sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    )
    sim_utils.activate_contact_sensors(articulation_prim_path)


@configclass
class SkillMimicBallPlayEnvCfg(DirectRLEnvCfg):
    """Configuration matching `skillmimic/data/cfg/skillmimic.yaml`."""

    episode_length_s = 1.0
    # The legacy Isaac Gym task used two PhysX substeps for every 60 Hz
    # control step. Isaac Lab expresses that as 120 Hz physics with a
    # decimation of two.
    decimation = 2
    num_actions = 156
    num_observations = 902
    num_states = 0

    condition_size = 64
    motion_path = DEFAULT_MOTION_PATH
    state_init = -1
    data_fps = 60.0
    data_frames_scale = 1.0
    init_vel = False
    early_termination = True
    termination_height = 0.25

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.8,
        ),
        physx=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=4,
            max_position_iteration_count=4,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=2.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot/Pelvis",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.89),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
                velocity_limit=100.0,
            )
        },
    )

    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.12,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_angular_velocity=100.0,
                max_depenetration_velocity=10.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.81,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.45, 0.05)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    robot_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Pelvis/.*", update_period=0.0
    )
    ball_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Ball", update_period=0.0
    )


class SkillMimicBallPlayEnv(DirectRLEnv):
    cfg: SkillMimicBallPlayEnvCfg

    def __init__(self, cfg: SkillMimicBallPlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._validate_asset_layout()

        legacy_body_names = _mjcf_names(HUMANOID_MJCF, "body")
        self._key_body_ids = [legacy_body_names.index(name) for name in KEY_BODY_NAMES]
        self._contact_obs_ids = [legacy_body_names.index(name) for name in FINGERTIP_NAMES]
        self._undesired_contact_ids = [legacy_body_names.index(name) for name in UNDESIRED_CONTACT_NAMES]

        lower = self.robot.data.soft_joint_pos_limits[0, self._joint_ids_legacy_order, 0]
        upper = self.robot.data.soft_joint_pos_limits[0, self._joint_ids_legacy_order, 1]
        self.action_offset = 0.5 * (upper + lower)
        self.action_scale = 0.5 * (upper - lower)
        self.joint_velocity_limit = self.robot.data.soft_joint_vel_limits[
            0, self._joint_ids_legacy_order
        ].clone()
        self.actions = torch.zeros((self.num_envs, 156), device=self.device)
        self._reset_joint_targets = torch.zeros_like(self.actions)
        self._reset_target_substeps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.motion_data = MotionDataHandler(
            motion_path=self.cfg.motion_path,
            device=self.device,
            key_body_ids=self._key_body_ids,
            num_envs=self.num_envs,
            max_episode_length=self.max_episode_length,
            reward_weights=REWARD_WEIGHTS,
            data_fps=self.cfg.data_fps,
            data_frames_scale=self.cfg.data_frames_scale,
            init_vel=self.cfg.init_vel,
        )
        self.reference_hoi = torch.zeros(
            (self.num_envs, self.max_episode_length, 380), device=self.device
        )
        self.current_hoi = torch.zeros((self.num_envs, 380), device=self.device)
        self.previous_hoi = torch.zeros_like(self.current_hoi)
        self.condition = torch.zeros((self.num_envs, self.cfg.condition_size), device=self.device)
        self._non_finite_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._warning_counts: dict[str, int] = {}
        # Capture the first post-reset simulation state once.  Comparing it
        # with the exact pose written at reset distinguishes a joint
        # coordinate/import mismatch from a controller that immediately pulls
        # the character away from the reference pose.
        self._initial_joint_pos = torch.zeros_like(self.actions)
        self._joint_diagnostic_pending = True

    def _setup_scene(self) -> None:
        _import_humanoid_mjcf(HUMANOID_MJCF, ROBOT_SOURCE_PRIM_PATH)
        self.robot = Articulation(self.cfg.robot)
        self.ball = RigidObject(self.cfg.ball)
        self.robot_contact_sensor = ContactSensor(self.cfg.robot_contact_sensor)
        self.ball_contact_sensor = ContactSensor(self.cfg.ball_contact_sensor)

        # GroundPlaneCfg references NVIDIA's remote default_environment.usd.
        # Use an equivalent local static collider so compute nodes can run
        # without outbound network access.
        ground_cfg = sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.8,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        )
        ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, -0.05))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball
        self.scene.sensors["robot_contact"] = self.robot_contact_sensor
        self.scene.sensors["ball_contact"] = self.ball_contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _validate_asset_layout(self) -> None:
        expected_bodies = _mjcf_names(HUMANOID_MJCF, "body")
        expected_joints = _mjcf_names(HUMANOID_MJCF, "joint")
        actual_bodies = list(self.robot.data.body_names)
        actual_joints = list(self.robot.data.joint_names)
        if len(actual_bodies) != len(expected_bodies) or set(actual_bodies) != set(expected_bodies):
            raise RuntimeError(
                "MJCF body names changed during import; legacy observations would be invalid. "
                f"Expected {len(expected_bodies)} bodies {expected_bodies}, "
                f"got {len(actual_bodies)} {actual_bodies}."
            )
        if len(actual_joints) != 156 or set(actual_joints) != set(expected_joints):
            raise RuntimeError(
                "MJCF joint names changed during conversion; legacy actions would be invalid. "
                f"Expected 156 joints {expected_joints}, got {len(actual_joints)} {actual_joints}."
            )
        self._body_ids_legacy_order = [actual_bodies.index(name) for name in expected_bodies]
        self._legacy_root_body_id = actual_bodies.index(expected_bodies[0])
        self._joint_ids_legacy_order = [actual_joints.index(name) for name in expected_joints]
        self._joint_ids_sim_order = [expected_joints.index(name) for name in actual_joints]

        contact_bodies = list(self.robot_contact_sensor.body_names)
        if not set(expected_bodies).issubset(contact_bodies):
            raise RuntimeError(
                "Contact sensors do not cover every legacy body. "
                f"Expected at least {expected_bodies}, got {contact_bodies}."
            )
        self._contact_force_order = [contact_bodies.index(name) for name in expected_bodies]

    def _robot_contact_forces(self) -> torch.Tensor:
        return self.robot_contact_sensor.data.net_forces_w[:, self._contact_force_order]

    def _legacy_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.robot.data.joint_pos[:, self._joint_ids_legacy_order],
            self.robot.data.joint_vel[:, self._joint_ids_legacy_order],
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        targets_legacy = self.action_offset + self.action_scale * self.actions
        hold_initial_target = self._reset_target_substeps > 0
        if torch.any(hold_initial_target):
            targets_legacy[hold_initial_target] = self._reset_joint_targets[hold_initial_target]
            self._reset_target_substeps[hold_initial_target] -= 1
        self.robot.set_joint_position_target(targets_legacy[:, self._joint_ids_sim_order])

    def _local_body_state(self):
        body_pos = self.robot.data.body_pos_w[:, self._body_ids_legacy_order]
        body_pos = body_pos - self.scene.env_origins.unsqueeze(1)
        body_rot = math_utils.wxyz_to_xyzw(
            self.robot.data.body_quat_w[:, self._body_ids_legacy_order]
        )
        body_vel = self.robot.data.body_lin_vel_w[:, self._body_ids_legacy_order]
        body_ang_vel = self.robot.data.body_ang_vel_w[:, self._body_ids_legacy_order]
        return body_pos, body_rot, body_vel, body_ang_vel

    def _local_ball_state(self) -> torch.Tensor:
        state = self.ball.data.root_state_w.clone()
        state[:, :3] -= self.scene.env_origins
        state[:, 3:7] = math_utils.wxyz_to_xyzw(state[:, 3:7])
        return state

    def _update_articulation_kinematics(self) -> None:
        """Refresh link state after direct root or joint tensor writes."""

        self.robot._physics_sim_view.update_articulations_kinematic()
        self.robot.data._body_state_w.timestamp = -1.0

    def _write_legacy_root_state_to_sim(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_rot: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> None:
        """Write a legacy Pelvis root state through PhysX's imported root.

        Isaac Gym treated the MJCF ``Pelvis`` free joint as the actor root.
        Isaac Sim 4.1's MJCF importer instead exposes ``Chest`` as the PhysX
        articulation root for this asset.  Root tensor writes therefore need
        to preserve the current root-to-Pelvis transform rather than assigning
        the legacy Pelvis pose directly to the imported root.
        """

        target_pelvis_pos = root_pos + self.scene.env_origins[env_ids]
        seed_pose = torch.cat((target_pelvis_pos, math_utils.xyzw_to_wxyz(root_rot)), dim=-1)
        seed_velocity = torch.cat((root_lin_vel, root_ang_vel), dim=-1)
        self.robot.write_root_pose_to_sim(seed_pose, env_ids)
        self.robot.write_root_velocity_to_sim(seed_velocity, env_ids)
        self._update_articulation_kinematics()

        articulation_state = self.robot.data.root_state_w[env_ids]
        pelvis_state = self.robot.data.body_state_w[env_ids, self._legacy_root_body_id]
        corrected_pos, corrected_rot = math_utils.retarget_root_pose(
            articulation_state[:, :3],
            math_utils.wxyz_to_xyzw(articulation_state[:, 3:7]),
            pelvis_state[:, :3],
            math_utils.wxyz_to_xyzw(pelvis_state[:, 3:7]),
            target_pelvis_pos,
            root_rot,
        )
        corrected_pose = torch.cat(
            (corrected_pos, math_utils.xyzw_to_wxyz(corrected_rot)), dim=-1
        )
        self.robot.write_root_pose_to_sim(corrected_pose, env_ids)
        self._update_articulation_kinematics()

        # Joint velocities along the reversed Chest-to-Pelvis chain contribute
        # to the Pelvis twist.  Measure that contribution and correct the
        # imported root velocity so the legacy Pelvis velocity is preserved.
        articulation_state = self.robot.data.root_state_w[env_ids]
        pelvis_state = self.robot.data.body_state_w[env_ids, self._legacy_root_body_id]
        corrected_lin_vel, corrected_ang_vel = math_utils.retarget_root_velocity(
            articulation_state[:, :3],
            articulation_state[:, 7:10],
            articulation_state[:, 10:13],
            pelvis_state[:, :3],
            pelvis_state[:, 7:10],
            pelvis_state[:, 10:13],
            root_lin_vel,
            root_ang_vel,
        )
        self.robot.write_root_velocity_to_sim(
            torch.cat((corrected_lin_vel, corrected_ang_vel), dim=-1), env_ids
        )
        self._update_articulation_kinematics()

    def _motion_debug_context(self, env_ids: torch.Tensor) -> str:
        """Return a bounded sample instead of dumping every affected environment."""

        sample_env_ids = env_ids[:5]
        motion_ids = self.motion_data.envid2motid[sample_env_ids].detach().cpu().tolist()
        start_frames = (
            self.motion_data.envid2startframe[sample_env_ids].detach().cpu().tolist()
        )
        paths = [
            os.path.relpath(self.motion_data.motion_paths[motion_id], PROJECT_ROOT)
            for motion_id in motion_ids
        ]
        episode_steps = self.episode_length_buf[sample_env_ids].detach().cpu().tolist()
        return (
            f"affected_envs={len(env_ids)} "
            f"sample_env_ids={sample_env_ids.detach().cpu().tolist()} "
            f"sample_episode_steps={episode_steps} "
            f"sample_motion_ids={motion_ids} "
            f"sample_start_frames={start_frames} "
            f"sample_motion_paths={paths}"
        )

    def _should_log_warning(self, name: str) -> tuple[bool, int]:
        """Log the first occurrences, then periodically, to avoid console floods."""

        occurrence = self._warning_counts.get(name, 0) + 1
        self._warning_counts[name] = occurrence
        return occurrence <= 3 or occurrence % 100 == 0, occurrence

    def _print_joint_consistency_diagnostic(
        self,
        body_pos: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> None:
        """Compare the first simulated pose with the pose written at reset."""

        if not self._joint_diagnostic_pending:
            return
        self._joint_diagnostic_pending = False

        reset_reference = self.reference_hoi[:, 0]
        frame = torch.minimum(self.episode_length_buf, self.motion_data.envid2episode_lengths - 1)
        frame = torch.clamp(frame, min=0, max=self.max_episode_length - 1)
        env_ids = torch.arange(self.num_envs, device=self.device)
        step_reference = self.reference_hoi[env_ids, frame]
        reference_joint_pos = step_reference[:, 6:162]
        reference_key_pos = step_reference[:, 328 : 328 + len(self._key_body_ids) * 3].reshape(
            self.num_envs, len(self._key_body_ids), 3
        )
        simulated_key_pos = body_pos[:, self._key_body_ids]
        articulation_root_pos = self.robot.data.root_state_w[:, :3] - self.scene.env_origins

        finite = (
            torch.isfinite(joint_pos).all(dim=-1)
            & torch.isfinite(body_pos).reshape(self.num_envs, -1).all(dim=-1)
            & torch.isfinite(articulation_root_pos).all(dim=-1)
            & torch.isfinite(reset_reference).all(dim=-1)
        )
        valid_count = int(finite.sum().item())
        if valid_count == 0:
            print(
                "[SkillMimic Lab][joint-check] no finite environments after the first physics step",
                flush=True,
            )
            return

        # Use the periodic difference for hinge coordinates so equivalent
        # +pi/-pi representations do not look like a large mismatch.
        joint_delta = joint_pos[finite] - reference_joint_pos[finite]
        joint_delta = torch.atan2(torch.sin(joint_delta), torch.cos(joint_delta))
        target_delta = joint_pos[finite] - self._initial_joint_pos[finite]
        target_delta = torch.atan2(torch.sin(target_delta), torch.cos(target_delta))
        key_delta = simulated_key_pos[finite] - reference_key_pos[finite]
        # Remove whole-character translation before judging the imported
        # skeleton's forward kinematics.  Also compare PhysX's articulation
        # root with the first link to expose a root-frame import offset.
        relative_key_delta = (
            simulated_key_pos[finite] - body_pos[finite, 0].unsqueeze(1)
        ) - (
            reference_key_pos[finite] - step_reference[finite, :3].unsqueeze(1)
        )
        root_delta = body_pos[finite, 0] - step_reference[finite, :3]
        articulation_root_delta = articulation_root_pos[finite] - step_reference[finite, :3]
        root_link_delta = body_pos[finite, 0] - articulation_root_pos[finite]

        # PhysX exposes the articulation root transform and all link
        # transforms through separate tensor APIs.  Keep this bounded detail
        # in the one-shot diagnostic: if their ordering disagrees, the link
        # nearest to the articulation root identifies the displaced index;
        # if Pelvis is nearest but the vector rotates across environments, the
        # importer has introduced a local root-frame offset.
        sim_order_body_pos = self.robot.data.body_pos_w[finite] - self.scene.env_origins[
            finite
        ].unsqueeze(1)
        root_link_dist = torch.linalg.vector_norm(
            sim_order_body_pos - articulation_root_pos[finite].unsqueeze(1), dim=-1
        ).mean(dim=0)
        nearest_body_ids = torch.topk(
            root_link_dist, k=min(5, len(self.robot.data.body_names)), largest=False
        ).indices
        nearest_bodies = ", ".join(
            f"{self.robot.data.body_names[index]}={root_link_dist[index].item():.6f}m"
            for index in nearest_body_ids.detach().cpu().tolist()
        )

        mean_joint_error = target_delta.abs().mean(dim=0)
        worst_joint_ids = torch.topk(mean_joint_error, k=min(5, self.actions.shape[1])).indices
        expected_joint_names = _mjcf_names(HUMANOID_MJCF, "joint")
        worst_joints = ", ".join(
            f"{expected_joint_names[index]}={mean_joint_error[index].item():.6f}rad"
            for index in worst_joint_ids.detach().cpu().tolist()
        )
        print(
            "[SkillMimic Lab][joint-check] "
            f"valid_envs={valid_count}/{self.num_envs} "
            f"joint_step_ref_rmse={joint_delta.square().mean().sqrt().item():.6f}rad "
            f"joint_reset_rmse={target_delta.square().mean().sqrt().item():.6f}rad "
            f"joint_max_error={joint_delta.abs().max().item():.6f}rad "
            f"key_body_rmse={key_delta.square().mean().sqrt().item():.6f}m "
            f"key_body_max_error={key_delta.abs().max().item():.6f}m "
            f"relative_key_body_rmse={relative_key_delta.square().mean().sqrt().item():.6f}m "
            f"pelvis_step_ref_rmse={root_delta.square().mean().sqrt().item():.6f}m "
            f"articulation_root_step_ref_rmse="
            f"{articulation_root_delta.square().mean().sqrt().item():.6f}m "
            f"pelvis_articulation_root_rmse="
            f"{root_link_delta.square().mean().sqrt().item():.6f}m",
            flush=True,
        )
        print(f"[SkillMimic Lab][joint-check] worst_joints={worst_joints}", flush=True)
        root_link_mean = root_link_delta.mean(dim=0)
        root_link_std = root_link_delta.std(dim=0)
        print(
            "[SkillMimic Lab][joint-check] "
            f"pelvis_articulation_root_mean={root_link_mean.detach().cpu().tolist()} "
            f"pelvis_articulation_root_std={root_link_std.detach().cpu().tolist()} "
            f"nearest_root_links={nearest_bodies}",
            flush=True,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        body_pos, body_rot, body_vel, body_ang_vel = self._local_body_state()
        robot_forces = self._robot_contact_forces()
        joint_pos, joint_vel = self._legacy_joint_state()
        humanoid_obs = compute_humanoid_observations(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            robot_forces,
            self._contact_obs_ids,
        )
        ball_state = self._local_ball_state()
        object_obs = compute_object_observations(body_pos[:, 0], body_rot[:, 0], ball_state)
        observation = torch.cat((humanoid_obs, object_obs, self.condition), dim=-1)
        if observation.shape[1] != 902:
            raise RuntimeError(f"Policy observation must be 902-D, got {observation.shape}")
        self.current_hoi = build_hoi_observations(
            body_pos[:, 0],
            body_rot[:, 0],
            joint_pos,
            joint_vel,
            body_pos[:, self._key_body_ids],
            ball_state,
            self.episode_length_buf,
        )
        # An articulation reset writes the new root/joint state directly to
        # PhysX, but Isaac Lab 1.1 can retain the pre-reset child-body cache
        # until the next physics update. If the old state contained NaNs, do
        # not expose that one stale frame to the policy.
        finite_observation = torch.isfinite(observation).all(dim=-1)
        finite_hoi = torch.isfinite(self.current_hoi).all(dim=-1)
        invalid_envs = torch.nonzero(~(finite_observation & finite_hoi), as_tuple=False).squeeze(-1)
        if len(invalid_envs) > 0:
            should_log, occurrence = self._should_log_warning("non_finite_observation")
            if should_log:
                print(
                    "[SkillMimic Lab] warning=sanitized_non_finite_observation "
                    f"occurrence={occurrence} "
                    f"{self._motion_debug_context(invalid_envs)}",
                    flush=True,
                )
            observation[invalid_envs] = 0.0
            self.current_hoi[invalid_envs] = 0.0
            self.previous_hoi[invalid_envs] = 0.0
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        self.previous_hoi.copy_(self.current_hoi)
        body_pos, body_rot, _, _ = self._local_body_state()
        ball_state = self._local_ball_state()
        joint_pos, joint_vel = self._legacy_joint_state()
        self._print_joint_consistency_diagnostic(body_pos, joint_pos)
        self.current_hoi = build_hoi_observations(
            body_pos[:, 0],
            body_rot[:, 0],
            joint_pos,
            joint_vel,
            body_pos[:, self._key_body_ids],
            ball_state,
            self.episode_length_buf,
        )
        frame = torch.minimum(self.episode_length_buf, self.motion_data.envid2episode_lengths - 1)
        frame = torch.clamp(frame, min=0, max=self.max_episode_length - 1)
        env_ids = torch.arange(self.num_envs, device=self.device)
        reference = self.reference_hoi[env_ids, frame]
        reward, reward_terms = compute_imitation_reward(
            reference,
            self.current_hoi,
            self.previous_hoi,
            self._robot_contact_forces(),
            self.ball_contact_sensor.data.net_forces_w[:, 0],
            self._undesired_contact_ids,
            len(self._key_body_ids),
            self.motion_data.reward_weights,
        )
        for name, term in reward_terms.items():
            finite = torch.isfinite(term)
            if torch.any(finite):
                term_mean = term[finite].mean()
            else:
                term_mean = torch.zeros((), device=term.device, dtype=term.dtype)
            self.extras[f"skillmimic/reward/{name}"] = term_mean
        non_finite_reward = ~torch.isfinite(reward)
        if torch.any(non_finite_reward):
            env_ids = torch.nonzero(non_finite_reward, as_tuple=False).squeeze(-1)
            should_log, occurrence = self._should_log_warning("non_finite_reward")
            if should_log:
                print(
                    "[SkillMimic Lab] warning=non_finite_reward "
                    f"occurrence={occurrence} "
                    f"{self._motion_debug_context(env_ids)}",
                    flush=True,
                )
            # DirectRLEnv computes reset_buf before rewards. Extend both reset
            # buffers here so a bad environment is reset at the end of this
            # same step, and return a finite terminal reward to the runner.
            self.reset_terminated[env_ids] = True
            self.reset_buf[env_ids] = True
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        self.extras["skillmimic/mean_reward"] = reward.mean()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        finite_state = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        state_fields = {
            "body_pos": self.robot.data.body_pos_w,
            "body_quat": self.robot.data.body_quat_w,
            "body_lin_vel": self.robot.data.body_lin_vel_w,
            "body_ang_vel": self.robot.data.body_ang_vel_w,
            "joint_pos": self.robot.data.joint_pos,
            "joint_vel": self.robot.data.joint_vel,
            "ball_root_state": self.ball.data.root_state_w,
            "robot_contact": self.robot_contact_sensor.data.net_forces_w,
            "ball_contact": self.ball_contact_sensor.data.net_forces_w,
        }
        finite_fields: dict[str, torch.Tensor] = {}
        for name, value in state_fields.items():
            finite_fields[name] = torch.isfinite(value).reshape(self.num_envs, -1).all(dim=-1)
            finite_state &= finite_fields[name]
        self._non_finite_state.copy_(~finite_state)
        if torch.any(self._non_finite_state):
            env_ids = torch.nonzero(self._non_finite_state, as_tuple=False).squeeze(-1)
            should_log, occurrence = self._should_log_warning("non_finite_sim_state")
            if should_log:
                invalid_field_counts = {
                    name: int((~field[env_ids]).sum().item())
                    for name, field in finite_fields.items()
                    if torch.any(~field[env_ids])
                }
                print(
                    "[SkillMimic Lab] warning=non_finite_sim_state "
                    f"occurrence={occurrence} "
                    f"invalid_field_counts={invalid_field_counts} "
                    f"{self._motion_debug_context(env_ids)}",
                    flush=True,
                )
        if self.cfg.early_termination:
            root_height = self.robot.data.body_pos_w[:, self._legacy_root_body_id, 2]
            terminated = (root_height < self.cfg.termination_height) & (self.episode_length_buf > 1)
        else:
            terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        terminated |= self._non_finite_state
        time_out = self.episode_length_buf >= self.motion_data.envid2episode_lengths - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        self.ball.reset(env_ids)
        super()._reset_idx(env_ids)

        motion_ids = self.motion_data.sample_motions(len(env_ids))
        deterministic = None if self.cfg.state_init < 0 else int(self.cfg.state_init)
        start_frames = self.motion_data.sample_time(motion_ids, deterministic)
        (
            reference,
            root_pos,
            root_rot,
            root_lin_vel,
            root_ang_vel,
            joint_pos,
            joint_vel,
            ball_pos,
            ball_lin_vel,
            ball_rot,
            ball_ang_vel,
        ) = self.motion_data.get_initial_state(env_ids, motion_ids, start_frames)

        # Keep sampled initial velocities within the imported articulation's
        # PhysX limits. Periodic differencing removes wrap artifacts, while
        # this guard handles genuinely fast mocap transitions.
        joint_vel = torch.clamp(
            joint_vel,
            min=-self.joint_velocity_limit.unsqueeze(0),
            max=self.joint_velocity_limit.unsqueeze(0),
        )

        self.robot.write_joint_state_to_sim(
            joint_pos[:, self._joint_ids_sim_order],
            joint_vel[:, self._joint_ids_sim_order],
            None,
            env_ids,
        )
        # On the GPU tensor backend, set_dof_positions() does not immediately
        # propagate the new joint coordinates to the child-link transforms.
        # Isaac Sim's own PhysX tensor test explicitly performs this update
        # before reading get_link_transforms().  Isaac Lab 1.1 omits both this
        # update and the lazy-buffer invalidation in its state writers.
        self._write_legacy_root_state_to_sim(
            env_ids, root_pos, root_rot, root_lin_vel, root_ang_vel
        )
        self._reset_joint_targets[env_ids] = joint_pos
        self._initial_joint_pos[env_ids] = joint_pos
        self._reset_target_substeps[env_ids] = self.cfg.decimation

        ball_pose = torch.cat((ball_pos + self.scene.env_origins[env_ids], math_utils.xyzw_to_wxyz(ball_rot)), dim=-1)
        ball_velocity = torch.cat((ball_lin_vel, ball_ang_vel), dim=-1)
        self.ball.write_root_pose_to_sim(ball_pose, env_ids)
        self.ball.write_root_velocity_to_sim(ball_velocity, env_ids)

        self.reference_hoi[env_ids] = reference
        labels = self.motion_data.motion_class[motion_ids]
        self.condition[env_ids] = F.one_hot(labels, num_classes=self.cfg.condition_size).float()
        self.current_hoi[env_ids] = 0.0
        self.previous_hoi[env_ids] = 0.0


def compute_humanoid_observations(
    body_pos: torch.Tensor,
    body_rot: torch.Tensor,
    body_vel: torch.Tensor,
    body_ang_vel: torch.Tensor,
    contact_forces: torch.Tensor,
    contact_body_ids: list[int],
) -> torch.Tensor:
    root_pos = body_pos[:, 0]
    root_rot = body_rot[:, 0]
    heading_inv = math_utils.calc_heading_quat_inv(root_rot)
    flat_heading = heading_inv.unsqueeze(1).expand(-1, body_pos.shape[1], -1).reshape(-1, 4)

    local_pos = (body_pos - root_pos.unsqueeze(1)).reshape(-1, 3)
    local_pos = math_utils.quat_rotate(flat_heading, local_pos).reshape(body_pos.shape[0], -1)[:, 3:]
    local_rot = math_utils.quat_mul(flat_heading, body_rot.reshape(-1, 4))
    local_rot = math_utils.quat_to_tan_norm(local_rot).reshape(body_pos.shape[0], -1)
    local_vel = math_utils.quat_rotate(flat_heading, body_vel.reshape(-1, 3)).reshape(body_pos.shape[0], -1)
    local_ang_vel = math_utils.quat_rotate(flat_heading, body_ang_vel.reshape(-1, 3)).reshape(body_pos.shape[0], -1)
    contact = contact_forces[:, contact_body_ids].reshape(body_pos.shape[0], -1)
    return torch.cat((root_pos[:, 2:3], local_pos, local_rot, local_vel, local_ang_vel, contact), dim=-1)


def compute_object_observations(
    root_pos: torch.Tensor, root_rot: torch.Tensor, ball_state: torch.Tensor
) -> torch.Tensor:
    heading_inv = math_utils.calc_heading_quat_inv(root_rot)
    local_pos = ball_state[:, :3] - root_pos
    local_pos[:, 2] = ball_state[:, 2]
    local_pos = math_utils.quat_rotate(heading_inv, local_pos)
    local_rot = math_utils.quat_mul(heading_inv, ball_state[:, 3:7])
    local_vel = math_utils.quat_rotate(heading_inv, ball_state[:, 7:10])
    local_ang_vel = math_utils.quat_rotate(heading_inv, ball_state[:, 10:13])
    return torch.cat((local_pos, math_utils.quat_to_tan_norm(local_rot), local_vel, local_ang_vel), dim=-1)


def build_hoi_observations(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    key_body_pos: torch.Tensor,
    ball_state: torch.Tensor,
    progress: torch.Tensor,
) -> torch.Tensor:
    joint_vel = joint_vel * (progress != 1).unsqueeze(-1)
    fake_contact = torch.zeros((root_pos.shape[0], 1), device=root_pos.device)
    return torch.cat(
        (
            root_pos,
            math_utils.quat_to_exp_map(root_rot),
            joint_pos,
            joint_vel,
            ball_state[:, :10],
            key_body_pos.reshape(root_pos.shape[0], -1),
            fake_contact,
        ),
        dim=-1,
    )


def compute_imitation_reward(
    reference: torch.Tensor,
    current: torch.Tensor,
    previous: torch.Tensor,
    robot_contact_forces: torch.Tensor,
    ball_contact_force: torch.Tensor,
    undesired_contact_ids: list[int],
    key_body_count: int,
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    root_pos = current[:, :3]
    body_rot = current[:, 3:162]
    joint_vel = current[:, 162:318]
    obj_pos = current[:, 318:321]
    obj_vel = current[:, 325:328]
    key_pos = torch.cat((root_pos, current[:, 328 : 328 + key_body_count * 3]), dim=-1)
    interaction = key_pos.view(-1, key_body_count + 1, 3) - obj_pos.unsqueeze(1)

    ref_root_pos = reference[:, :3]
    ref_body_rot = reference[:, 3:162]
    ref_joint_vel = reference[:, 162:318]
    ref_obj_pos = reference[:, 318:321]
    ref_obj_vel = reference[:, 325:328]
    ref_key_pos = torch.cat((ref_root_pos, reference[:, 328 : 328 + key_body_count * 3]), dim=-1)
    ref_interaction = ref_key_pos.view(-1, key_body_count + 1, 3) - ref_obj_pos.unsqueeze(1)
    ref_obj_contact = reference[:, -1]

    position_reward = torch.exp(-torch.mean((ref_key_pos - key_pos).square(), dim=-1) * weights["p"])
    rotation_reward = torch.exp(-torch.mean((ref_body_rot - body_rot).square(), dim=-1) * weights["r"])
    angular_velocity_reward = torch.exp(
        -torch.mean((ref_joint_vel - joint_vel).square(), dim=-1) * weights["rv"]
    )
    previous_joint_vel = previous[:, 162:318]
    smooth_error = torch.mean(
        (joint_vel - previous_joint_vel).square() / ((ref_joint_vel.square() + 1.0e-12) * 1.0e12), dim=-1
    )
    smoothness_reward = torch.exp(-0.1 * smooth_error)
    body_reward = position_reward * rotation_reward * angular_velocity_reward * smoothness_reward

    object_position_reward = torch.exp(-torch.mean((ref_obj_pos - obj_pos).square(), dim=-1) * weights["op"])
    object_velocity_reward = torch.exp(-torch.mean((ref_obj_vel - obj_vel).square(), dim=-1) * weights["opv"])
    object_reward = object_position_reward * object_velocity_reward
    interaction_reward = torch.exp(
        -torch.mean((ref_interaction - interaction).square(), dim=(1, 2)) * weights["ig"]
    )

    body_force = robot_contact_forces[:, undesired_contact_ids]
    body_contact = torch.any(torch.abs(body_force) >= 0.1, dim=(1, 2)).float()
    obj_contact = torch.any(torch.abs(ball_contact_force[:, :2]) > 0.1, dim=-1).float()
    body_contact_reward = torch.exp(-body_contact * weights["cg1"])
    object_contact_reward = torch.exp(-torch.abs(obj_contact - ref_obj_contact) * weights["cg2"])
    total_reward = body_reward * object_reward * interaction_reward * body_contact_reward * object_contact_reward
    reward_terms = {
        "position": position_reward,
        "rotation": rotation_reward,
        "angular_velocity": angular_velocity_reward,
        "smoothness": smoothness_reward,
        "object_position": object_position_reward,
        "object_velocity": object_velocity_reward,
        "interaction": interaction_reward,
        "body_contact": body_contact_reward,
        "object_contact": object_contact_reward,
        "body": body_reward,
        "object": object_reward,
        "total": total_reward,
    }
    return total_reward, reward_terms
