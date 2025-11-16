# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets import REXAI_CFG
##
# Configuration
##

@configclass
class RexaiEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition
    action_space = 10
    observation_space = 42
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = REXAI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=10, env_spacing=4.0, replicate_physics=True)

    joint_gears: list = [
        1.00,  # head_rev
        1.00,  # left_hip_rev
        1.00,  # left_femur_rev
        1.00,  # left_tibia_rev
        1.00,  # left_ankle_rev
        1.00,  # right_hip_rev
        1.00,  # right_femur_rev
        1.00,  # right_tibia_rev
        1.00,  # right_ankle_rev
        1.00,  # tail_rev
    ]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.8

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate_inverse, sample_uniform, yaw_quat



class RexaiEnv(DirectRLEnv):
    cfg: RexaiEnvCfg

    def __init__(self, cfg: RexaiEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Get joint indices for all actuated joints
        self._joint_indices, _ = self.robot.find_joints(".*")
        
        # Store dimensions
        self._num_joints = len(self._joint_indices)
        
        # Cache for observations
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        
        print(f"\n=== RexAI Environment Initialized ===")
        print(f"Number of environments: {self.num_envs}")
        print(f"Number of joints: {self._num_joints}")
        print(f"Joint indices: {self._joint_indices}")
        print(f"Action space: {self.cfg.action_space}")
        print(f"Observation space: {self.cfg.observation_space}")
        print(f"=====================================\n")

    def _setup_scene(self):
        """Setup the scene with robot and ground plane."""
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # Add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions before physics step."""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply actions as joint torques."""
        # Apply joint gears scaling
        joint_gears = torch.tensor(self.cfg.joint_gears, device=self.device, dtype=torch.float32)
        scaled_actions = self.actions * joint_gears * self.cfg.action_scale
        
        # Set joint efforts
        self.robot.set_joint_effort_target(scaled_actions, joint_ids=self._joint_indices)

    def _get_observations(self) -> dict:
        """
        Compute observations: 42-dimensional vector
        - Base linear velocity in base frame (3)
        - Base angular velocity in base frame (3)
        - Projected gravity vector (3)
        - Joint positions (10)
        - Joint velocities (10)
        - Previous actions (10)
        - Commands: velocity x, velocity y, angular velocity (3)
        Total: 3 + 3 + 3 + 10 + 10 + 10 + 3 = 42
        """
        # Get robot state
        root_quat = self.robot.data.root_quat_w  # (num_envs, 4) - world frame quaternion
        root_vel = self.robot.data.root_vel_w    # (num_envs, 6) - linear and angular velocity in world frame
        
        # Transform velocities to base frame
        self.base_lin_vel[:] = quat_rotate_inverse(root_quat, root_vel[:, 0:3])
        self.base_ang_vel[:] = quat_rotate_inverse(root_quat, root_vel[:, 3:6])
        
        # Project gravity vector into base frame
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self.projected_gravity[:] = quat_rotate_inverse(root_quat, gravity_vec)
        
        # Get joint states
        joint_pos = self.robot.data.joint_pos[:, self._joint_indices]
        joint_vel = self.robot.data.joint_vel[:, self._joint_indices] * self.cfg.dof_vel_scale
        
        # Commands (for now, simple forward walking command)
        # You can make this dynamic later
        commands = torch.zeros(self.num_envs, 3, device=self.device)
        commands[:, 0] = 1.0  # Forward velocity command
        
        # Concatenate all observations
        obs = torch.cat(
            (
                self.base_lin_vel,           # 3
                self.base_ang_vel * self.cfg.angular_velocity_scale,  # 3
                self.projected_gravity,       # 3
                joint_pos,                    # 10
                joint_vel,                    # 10
                self.actions,                 # 10 (previous actions)
                commands,                     # 3
            ),
            dim=-1,
        )
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for the bipedal locomotion task."""
        total_reward = compute_rewards(
            # Reward scales from config
            self.cfg.alive_reward_scale,
            self.cfg.heading_weight,
            self.cfg.up_weight,
            self.cfg.energy_cost_scale,
            self.cfg.actions_cost_scale,
            self.cfg.death_cost,
            # State information
            self.robot.data.root_pos_w[:, 2],     # Base height
            self.robot.data.root_quat_w,          # Base orientation
            self.robot.data.root_vel_w[:, 0:3],   # Linear velocity
            self.robot.data.root_vel_w[:, 3:6],   # Angular velocity
            self.robot.data.joint_vel[:, self._joint_indices],  # Joint velocities
            self.actions,                          # Actions (torques)
            self.cfg.termination_height,          # Termination height threshold
            self.reset_terminated,                # Termination flags
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        # Time out condition
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Get base height
        base_height = self.robot.data.root_pos_w[:, 2]
        
        # Check if robot fell (base too low)
        fallen = base_height < self.cfg.termination_height
        
        # Check if robot is too tilted
        root_quat = self.robot.data.root_quat_w
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        projected_gravity = quat_rotate_inverse(root_quat, gravity_vec)
        # If z-component of projected gravity is positive, robot is upside down
        upside_down = projected_gravity[:, 2] > 0.5
        
        # Combine termination conditions
        terminated = fallen | upside_down
        
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Get default states
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        
        # Randomize initial joint positions slightly for better exploration
        joint_pos += sample_uniform(
            -0.1,  # -0.1 radians
            0.1,   # +0.1 radians
            joint_pos.shape,
            self.device,
        )
        
        # Set initial joint velocities to zero
        joint_vel[:] = 0.0

        # Reset root state
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        
        # Add random yaw rotation
        random_yaw = sample_uniform(-torch.pi, torch.pi, (len(env_ids), 1), self.device)
        yaw_quat_random = yaw_quat(random_yaw.squeeze(-1))
        default_root_state[:, 3:7] = yaw_quat_random
        
        # Add environment origins
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Ensure robot spawns at correct height (adjust if needed)
        default_root_state[:, 2] = 0.65  # Adjust this based on your robot's standing height

        # Write states to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset action buffer for these environments
        self.actions[env_ids] = 0.0


@torch.jit.script
def compute_rewards(
    # Reward scales
    alive_reward_scale: float,
    heading_weight: float,
    up_weight: float,
    energy_cost_scale: float,
    actions_cost_scale: float,
    death_cost: float,
    # State
    base_height: torch.Tensor,
    base_quat: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
    termination_height: float,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    """
    Compute rewards for bipedal locomotion.
    
    Rewards:
    - Alive bonus: reward for staying upright
    - Forward velocity: reward for moving forward
    - Upright posture: reward for maintaining upright orientation
    - Energy penalty: penalize large joint velocities
    - Action penalty: penalize large control efforts
    - Death penalty: large negative reward for falling
    """
    # Alive reward
    alive = (base_height > termination_height).float()
    rew_alive = alive_reward_scale * alive
    
    # Death penalty
    rew_death = death_cost * reset_terminated.float()
    
    # Forward velocity reward (heading in x-direction)
    rew_heading = heading_weight * base_lin_vel[:, 0]
    
    # Upright reward (penalize tilting)
    gravity_vec = torch.zeros_like(base_lin_vel)
    gravity_vec[:, 2] = -1.0
    projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)
    # Reward being upright (gravity pointing down in base frame)
    rew_up = up_weight * projected_gravity[:, 2]
    
    # Energy cost (penalize high joint velocities)
    rew_energy = -energy_cost_scale * torch.sum(torch.square(joint_vel), dim=-1)
    
    # Action cost (penalize large control efforts)
    rew_actions = -actions_cost_scale * torch.sum(torch.square(actions), dim=-1)
    
    # Total reward
    total_reward = rew_alive + rew_heading + rew_up + rew_energy + rew_actions + rew_death
    
    return total_reward