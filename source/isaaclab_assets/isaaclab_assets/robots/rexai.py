# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the RexAI Bipedal Robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

REXAI_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{{ISAACLAB_ASSETS_DATA_DIR}}/Robots/RexAI/Assets/Trex/Trex.urdf",
        usd_dir=f"{{ISAACLAB_ASSETS_DATA_DIR}}/Robots/RexAI/Assets/Trex",
        usd_file_name="Trex.usd",
        fix_base=False,
        merge_fixed_joints=True,
        force_usd_conversion=False,
        make_instanceable=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),  # Spawn at 0.35m height (slightly above robot height of 0.3m)
        joint_pos={
            "head_rev": 0.0,
            "tail_rev": 0.0,
            "left_hip_rev": 0.0,
            "left_femur_rev": 0.0,
            "left_tibia_rev": 0.0,
            "left_ankle_rev": 0.0,
            "right_hip_rev": 0.0,
            "right_femur_rev": 0.0,
            "right_tibia_rev": 0.0,
            "right_ankle_rev": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_rev",
                "left_femur_rev",
                "left_tibia_rev",
                "left_ankle_rev",
                "right_hip_rev",
                "right_femur_rev",
                "right_tibia_rev",
                "right_ankle_rev",
            ],
            stiffness={
                "left_hip_rev": 40.0,
                "left_femur_rev": 40.0,
                "left_tibia_rev": 40.0,
                "left_ankle_rev": 20.0,
                "right_hip_rev": 40.0,
                "right_femur_rev": 40.0,
                "right_tibia_rev": 40.0,
                "right_ankle_rev": 20.0,
            },
            damping={
                "left_hip_rev": 5.0,
                "left_femur_rev": 5.0,
                "left_tibia_rev": 5.0,
                "left_ankle_rev": 2.0,
                "right_hip_rev": 5.0,
                "right_femur_rev": 5.0,
                "right_tibia_rev": 5.0,
                "right_ankle_rev": 2.0,
            },
        ),
        "auxiliary": ImplicitActuatorCfg(
            joint_names_expr=["head_rev", "tail_rev"],
            stiffness={"head_rev": 10.0, "tail_rev": 10.0},
            damping={"head_rev": 1.0, "tail_rev": 1.0},
        ),
    },
)
"""Configuration for the RexAI bipedal robot with position-controlled actuators."""
