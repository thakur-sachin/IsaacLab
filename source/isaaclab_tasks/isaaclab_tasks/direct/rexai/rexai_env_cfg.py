# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the RexAI bipedal locomotion environment."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots import REXAI_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for environment randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RexAIEnvCfg(DirectRLEnvCfg):
    """Configuration for the RexAI bipedal locomotion environment."""

    # Environment settings
    episode_length_s = 2000.0
    decimation = 4
    action_scale = 0.5
    action_space = 10  # 8 leg joints + 2 auxiliary (head, tail)
    observation_space = 45  # Will be computed in environment
    state_space = 0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # Robot
    robot: ArticulationCfg = REXAI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Contact sensor for feet
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*feet.*",
        history_length=3,
        update_period=0.0,  # Update every step
        track_air_time=True,
    )

    # Events
    events: EventCfg = EventCfg()

    # Reward scales - Forward locomotion focused
    lin_vel_reward_scale = 1.5  # Reward forward velocity
    yaw_rate_reward_scale = 0.3  # Small reward for turning ability
    z_vel_reward_scale = -2.0  # Penalize vertical movement
    ang_vel_reward_scale = -0.05  # Penalize unwanted rotations
    joint_torque_reward_scale = -1e-5  # Penalize energy consumption
    joint_accel_reward_scale = -2.5e-7  # Penalize rapid joint acceleration
    action_rate_reward_scale = -0.01  # Encourage smooth actions
    feet_air_time_reward_scale = 0.25  # Reward proper stepping
    undesired_contact_reward_scale = -1.0  # Penalize body/knee contacts
    flat_orientation_reward_scale = -1.0  # Penalize tilting
    dof_pos_limits_reward_scale = -1.0  # Penalize joint limit violations
    feet_stumble_reward_scale = -0.5  # Penalize feet stumbling

    # Termination settings
    termination_height = -0.1  # Terminate if base goes below 0.2m (robot is 0.3m tall)

    # Command settings (target velocities)
    commands_x_range = (0.5, 1.5)  # Target forward velocity range (m/s)
    commands_y_range = (0.0, 0.0)  # No lateral velocity for now
    commands_yaw_range = (-0.5, 0.5)  # Small turning velocity (rad/s)
