# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RexAI bipedal locomotion environment."""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .rexai_env_cfg import RexAIEnvCfg


class RexAIEnv(DirectRLEnv):
    """Environment for training RexAI bipedal robot to walk."""

    cfg: RexAIEnvCfg

    def __init__(self, cfg: RexAIEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Buffers for actions and commands
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # Velocity commands (x_vel, y_vel, yaw_rate)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Buffers for computing rewards
        self._joint_pos_prev = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Episode reward sums for logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "dof_pos_limits",
                "feet_stumble",
            ]
        }

        # Get contact sensor body indices
        # self._feet_ids, _ = self._contact_sensor.find_bodies(".*feet.*")
        # # Undesired contacts: base, thighs, knees
        # self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies("base_link|.*femur.*|.*tibia.*")

    def _setup_scene(self):
        """Setup the scene with robot, terrain, and sensors."""
        # Add robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Add contact sensor
        # self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Setup terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics step."""
        self._actions = actions.clone()
        # Scale actions and add to default joint positions for position control
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        """Apply processed actions to the robot."""
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        """Compute observations for the policy.

        Observations (45D):
        - Base linear velocity in base frame (3)
        - Base angular velocity in base frame (3)
        - Projected gravity in base frame (3)
        - Velocity commands (3)
        - Joint positions relative to default (10)
        - Joint velocities (10)
        - Previous actions (10)
        - Feet contact states (2)
        - Feet air time (2)
        """
        self._previous_actions = self._actions.clone()

        # Get feet contact information
        # feet_contact = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids, 2].max(dim=1)[0] > 1.0
        # feet_air_time = self._contact_sensor.data.current_air_time[:, self._feet_ids]

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,                                    # 3
                self._robot.data.root_ang_vel_b,                                    # 3
                self._robot.data.projected_gravity_b,                               # 3
                self._commands,                                                      # 3
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,   # 10
                self._robot.data.joint_vel,                                         # 10
                self._previous_actions,                                              # 10
                # feet_contact.float(),                                                # 2
                # feet_air_time,                                                       # 2
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for bipedal locomotion."""
        # Linear velocity tracking reward
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        lin_vel_reward = torch.exp(-lin_vel_error / 0.25) * self.cfg.lin_vel_reward_scale

        # Angular velocity tracking reward (yaw rate)
        ang_vel_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        ang_vel_reward = torch.exp(-ang_vel_error / 0.25) * self.cfg.yaw_rate_reward_scale

        # Penalize vertical velocity
        z_vel_penalty = torch.square(self._robot.data.root_lin_vel_b[:, 2]) * self.cfg.z_vel_reward_scale

        # Penalize roll and pitch rates
        ang_vel_xy_penalty = (
            torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1) * self.cfg.ang_vel_reward_scale
        )

        # Joint torque penalty (energy)
        joint_torque_penalty = (
            torch.sum(torch.square(self._robot.data.applied_torque), dim=1) * self.cfg.joint_torque_reward_scale
        )

        # Joint acceleration penalty (smoothness)
        joint_accel_penalty = (
            torch.sum(
                torch.square(self._robot.data.joint_acc),
                dim=1,
            )
            * self.cfg.joint_accel_reward_scale
        )

        # Action rate penalty (smooth actions)
        action_rate_penalty = (
            torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
            * self.cfg.action_rate_reward_scale
        )

        # Feet air time reward (encourage dynamic walking)
        # first_contact = self._contact_sensor.data.current_air_time[:, self._feet_ids] > 0.0
        # feet_air_time_reward = (
        #     torch.sum(
        #         self._contact_sensor.data.current_air_time[:, self._feet_ids] * first_contact.float(),
        #         dim=1,
        #     )
        #     * self.cfg.feet_air_time_reward_scale
        # )

        # Undesired contact penalty (body, thighs, knees shouldn't touch ground)
        # undesired_contact_penalty = (
        #     torch.sum(
        #         torch.any(
        #             torch.norm(
        #                 self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_body_ids, :],
        #                 dim=-1,
        #             )
        #             > 1.0,
        #             dim=1,
        #         ).float(),
        #         dim=1,
        #     )
        #     * self.cfg.undesired_contact_reward_scale
        # )

        # Flat orientation penalty (penalize tilting)
        flat_orientation_penalty = (
            torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
            * self.cfg.flat_orientation_reward_scale
        )

        # Joint position limits penalty
        joint_pos = self._robot.data.joint_pos
        joint_limits = self._robot.data.soft_joint_pos_limits
        dof_at_limit = torch.logical_or(
            joint_pos < joint_limits[:, :, 0] + 0.1, joint_pos > joint_limits[:, :, 1] - 0.1
        )
        dof_pos_limits_penalty = torch.sum(dof_at_limit.float(), dim=1) * self.cfg.dof_pos_limits_reward_scale

        # Feet stumble penalty (feet velocity while in contact should be low)
        # feet_vel = torch.norm(self._robot.data.body_lin_vel_w[:, self._feet_ids, :2], dim=-1)
        # feet_contact = (
        #     torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids, :], dim=-1) > 1.0
        # ).float()
        # feet_stumble_penalty = (
        #     torch.sum(feet_vel * feet_contact, dim=1) * self.cfg.feet_stumble_reward_scale
        # )

        # Total reward
        total_reward = (
            lin_vel_reward
            + ang_vel_reward
            + z_vel_penalty
            + ang_vel_xy_penalty
            + joint_torque_penalty
            + joint_accel_penalty
            + action_rate_penalty
            # + feet_air_time_reward
            # + undesired_contact_penalty
            + flat_orientation_penalty
            + dof_pos_limits_penalty
            # + feet_stumble_penalty
        )

        # Log episode sums
        self._episode_sums["track_lin_vel_xy_exp"] += lin_vel_reward
        self._episode_sums["track_ang_vel_z_exp"] += ang_vel_reward
        self._episode_sums["lin_vel_z_l2"] += z_vel_penalty
        self._episode_sums["ang_vel_xy_l2"] += ang_vel_xy_penalty
        self._episode_sums["dof_torques_l2"] += joint_torque_penalty
        self._episode_sums["dof_acc_l2"] += joint_accel_penalty
        self._episode_sums["action_rate_l2"] += action_rate_penalty
        # self._episode_sums["feet_air_time"] += feet_air_time_reward
        # self._episode_sums["undesired_contacts"] += undesired_contact_penalty
        self._episode_sums["flat_orientation_l2"] += flat_orientation_penalty
        self._episode_sums["dof_pos_limits"] += dof_pos_limits_penalty
        # self._episode_sums["feet_stumble"] += feet_stumble_penalty

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination and timeout conditions."""
        # Timeout condition
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Termination conditions
        # 1. Base height too low (robot fell)
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        fallen = base_height < self.cfg.termination_height

        # 2. Robot tilted too much (upside down or severe tilt)
        tilted = torch.abs(self._robot.data.projected_gravity_b[:, 2]) < 0.3  # ~72 degrees from upright

        # Combine termination conditions
        terminated = fallen | tilted

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Reset parent class (handles managers)
        super()._reset_idx(env_ids)

        # Robot state reset is handled by event manager

        # Sample new velocity commands for reset environments
        self._commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (
            self.cfg.commands_x_range[1] - self.cfg.commands_x_range[0]
        ) + self.cfg.commands_x_range[0]
        self._commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (
            self.cfg.commands_y_range[1] - self.cfg.commands_y_range[0]
        ) + self.cfg.commands_y_range[0]
        self._commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (
            self.cfg.commands_yaw_range[1] - self.cfg.commands_yaw_range[0]
        ) + self.cfg.commands_yaw_range[0]

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Log episode sums
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
