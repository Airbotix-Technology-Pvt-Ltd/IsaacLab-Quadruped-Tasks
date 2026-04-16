"""
License: MIT License
Copyright (c) 2024, Felipe Mohr Santos

Lite3 (Deep Robotics) robot support added by Airbotix Technology Pvt. Ltd.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from omni.isaac.lab_quadruped_tasks.cfg.quadruped_env_cfg import QuadrupedEnvCfg
from omni.isaac.lab_quadruped_tasks.cfg.quadruped_terrains_cfg import (
    ROUGH_TERRAINS_CFG,
    ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
    FULL_TERRAINS_CFG,
    FULL_TERRAINS_PLAY_CFG,
)

# ---------------------------------------------------------------------------
# Lite3 ArticulationCfg  (Deep Robotics Lite3)
# ---------------------------------------------------------------------------

LITE3_USD_PATH = (
    "/home/lite3/work/Lite3Robot/rl_training/deep_robotics_model/Lite3/Lite3_usd/Lite3.usd"
)

DEEPROBOTICS_LITE3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=LITE3_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*HipX_joint": 0.0,
            ".*HipY_joint": -0.8,
            ".*Knee_joint": 1.6,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.99,
    actuators={
        "Hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Hip[XY]_joint"],
            effort_limit=24.0,
            velocity_limit=26.2,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=5,
        ),
        "Knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Knee_joint"],
            effort_limit=36.0,
            velocity_limit=17.3,
            stiffness=30.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
            min_delay=0,
            max_delay=5,
        ),
    },
)


# ---------------------------------------------------------------------------
# Lite3 Base Environment
# ---------------------------------------------------------------------------


@configclass
class Lite3BaseEnvCfg(QuadrupedEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robot
        self.scene.robot = DEEPROBOTICS_LITE3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Height scanner sits on the robot torso (Lite3 uses "TORSO", not "base")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/TORSO"

        # Action scale
        self.actions.joint_pos.scale = 0.2

        # Override body names to match Lite3 link naming (uppercase)
        self.rewards.rew_feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*_FOOT"
        )
        self.rewards.pen_undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*_THIGH"
        )
        self.rewards.pen_feet_slide.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*_FOOT"
        )
        self.rewards.pen_feet_slide.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=".*_FOOT"
        )
        self.rewards.pen_joint_deviation.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*"]
        )

        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="TORSO"
        )
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="TORSO"
        )

        # Reward weights (tuned for Lite3 — same as Go2 baseline)
        self.rewards.rew_feet_air_time.weight = 0.75
        self.rewards.pen_joint_powers.weight = -3e-3
        self.rewards.pen_joint_deviation.weight = -0.1
        self.rewards.pen_undesired_contacts.weight = -0.25

        # Remove base_lin_vel from observations: real Lite3 hardware does not
        # expose linear velocity → SDK sends 45-dim obs, not 48.
        # This matches the official Deep Robotics rl_training config.
        self.observations.policy.base_lin_vel = None  # type: ignore


@configclass
class Lite3BaseEnvCfg_PLAY(Lite3BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None


# ---------------------------------------------------------------------------
# Lite3 Blind Flat
# ---------------------------------------------------------------------------


@configclass
class Lite3BlindFlatEnvCfg(Lite3BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.height_map = None

        self.events.change_vel_cmd = None
        self.curriculum.terrain_levels = None


@configclass
class Lite3BlindFlatEnvCfg_PLAY(Lite3BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.height_map = None

        self.events.change_vel_cmd = None
        self.curriculum.terrain_levels = None


# ---------------------------------------------------------------------------
# Lite3 Blind Rough
# ---------------------------------------------------------------------------


@configclass
class Lite3BlindRoughEnvCfg(Lite3BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.height_map = None

        self.events.change_vel_cmd = None

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG

        self.viewer.origin_type = "env"


@configclass
class Lite3BlindRoughEnvCfg_PLAY(Lite3BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.height_map = None

        self.events.change_vel_cmd = None

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_PLAY_CFG


# ---------------------------------------------------------------------------
# Lite3 Blind Stairs
# ---------------------------------------------------------------------------


@configclass
class Lite3BlindStairsEnvCfg(Lite3BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.height_map = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        self.scene.env_spacing = 8.0

        self.viewer.origin_type = "env"


@configclass
class Lite3BlindStairsEnvCfg_PLAY(Lite3BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.height_map = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG
        self.scene.env_spacing = 8.0


# ---------------------------------------------------------------------------
# Lite3 Vision (height-map)
# ---------------------------------------------------------------------------


@configclass
class Lite3VisionEnvCfg(Lite3BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FULL_TERRAINS_CFG

        self.events.change_vel_cmd = None

        self.viewer.origin_type = "env"


@configclass
class Lite3VisionEnvCfg_PLAY(Lite3BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = FULL_TERRAINS_PLAY_CFG

        self.events.change_vel_cmd = None

        self.viewer.origin_type = "env"


# ---------------------------------------------------------------------------
# Lite3 Vision Stairs
# ---------------------------------------------------------------------------


@configclass
class Lite3VisionStairsEnvCfg(Lite3BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        self.scene.env_spacing = 8.0

        self.viewer.origin_type = "env"


@configclass
class Lite3VisionStairsEnvCfg_PLAY(Lite3BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG
        self.scene.env_spacing = 8.0

        self.viewer.origin_type = "env"
