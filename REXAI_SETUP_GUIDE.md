# RexAI RL Training Setup - Complete Guide

## Overview
This guide documents the complete setup for training the RexAI bipedal robot using Isaac Lab and RSL-RL.

## What Was Implemented

### 1. Robot Configuration (`source/isaaclab_assets/isaaclab_assets/robots/rexai.py`)
- **URDF-based spawning**: Uses `UrdfFileCfg` to load the robot from URDF (auto-converts to USD at runtime)
- **Position control**: ImplicitActuator with tuned PD gains for stable bipedal locomotion
- **10 DoF**: 8 leg joints (hip, femur, tibia, ankle × 2) + 2 auxiliary (head, tail)
- **Spawn height**: 0.35m (robot is 0.3m tall)
- **Self-collision**: Disabled for stability
- **Actuator groups**: Separate tuning for legs (higher gains) vs auxiliary joints

### 2. Environment Configuration (`source/isaaclab_tasks/isaaclab_tasks/direct/rexai/rexai_env_cfg.py`)
- **Episode length**: 20 seconds (matching Anymal)
- **Decimation**: 4 (control at 50Hz with 200Hz physics)
- **Action space**: 10 (all joints)
- **Observation space**: 45 (detailed proprioception + commands + contact info)
- **Physics**: 200Hz simulation with proper friction/restitution
- **Terrain**: Flat plane for initial training
- **Contact sensors**: Feet contact tracking with air time monitoring
- **Randomization**:
  - Physics materials (friction 0.7-1.2)
  - Base mass ±0.5kg
  - Initial joint positions ±0.1 rad
  - Random yaw orientation at reset

### 3. Environment Implementation (`source/isaaclab_tasks/isaaclab_tasks/direct/rexai/rexai_env.py`)
**Observations (45D)**:
- Base linear velocity (3)
- Base angular velocity (3)
- Projected gravity (3)
- Velocity commands (3)
- Joint positions relative to default (10)
- Joint velocities (10)
- Previous actions (10)
- Feet contact states (2)
- Feet air time (2)

**Reward Components** (tuned for bipedal walking):
1. **Forward velocity tracking** (exponential reward): Encourages following commanded velocity
2. **Yaw rate tracking**: Enables turning capability
3. **Vertical velocity penalty**: Discourages hopping
4. **Angular velocity penalty**: Penalizes roll/pitch instability
5. **Joint torque penalty**: Encourages energy efficiency
6. **Joint acceleration penalty**: Promotes smooth motions
7. **Action rate penalty**: Prevents jerky control
8. **Feet air time reward**: Encourages dynamic walking (not shuffling)
9. **Undesired contact penalty**: Penalizes body/knee ground contact
10. **Flat orientation penalty**: Rewards upright posture
11. **Joint limits penalty**: Prevents hitting joint limits
12. **Feet stumble penalty**: Penalizes foot sliding during contact

**Termination Conditions**:
- Base height < 0.2m (fallen)
- Tilt > 72° from upright
- Episode timeout (20s)

### 4. RSL-RL Configuration (`source/isaaclab_tasks/isaaclab_tasks/direct/rexai/agents/rsl_rl_ppo_cfg.py`)
- **Algorithm**: PPO with adaptive learning rate
- **Network**: [256, 128, 64] hidden layers with ELU activation
- **Training**: 3000 iterations, 24 steps per env
- **Hyperparameters**:
  - Learning rate: 1e-3 (adaptive)
  - Entropy coefficient: 0.01 (exploration)
  - Clip param: 0.2
  - Gamma: 0.99, Lambda: 0.95 (GAE)
  - 5 epochs, 4 mini-batches per iteration

## File Structure
```
IsaacLab/
├── source/isaaclab_assets/isaaclab_assets/robots/
│   ├── rexai.py                          # Robot configuration
│   ├── RexAI/Assets/Trex/
│   │   ├── Trex.urdf                     # Robot URDF (provided)
│   │   └── meshes/                       # STL meshes (provided)
│   └── __init__.py                       # Updated to export REXAI_CFG
│
└── source/isaaclab_tasks/isaaclab_tasks/direct/rexai/
    ├── __init__.py                        # Task registration
    ├── rexai_env_cfg.py                  # Environment config
    ├── rexai_env.py                      # Environment implementation
    └── agents/
        ├── __init__.py                    # Agent package init
        └── rsl_rl_ppo_cfg.py             # RSL-RL PPO config
```

## How to Use

### Training
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-RexAI-Direct-v0 \
    --num_envs 10
```

### Training with More Environments (Recommended)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-RexAI-Direct-v0 \
    --num_envs 4096
```

### Resume Training
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-RexAI-Direct-v0 \
    --num_envs 4096 \
    --resume
```

### Play/Evaluate Trained Policy
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Isaac-RexAI-Direct-v0 \
    --num_envs 32 \
    --load_run <experiment_name>
```

## Expected Training Behavior

### Early Training (0-500 iterations)
- Robot will fall frequently
- Random flailing motions
- Learning to stay upright
- Reward: -5 to 0

### Mid Training (500-1500 iterations)
- Robot maintains balance
- Some forward shuffling
- Begins to coordinate legs
- Reward: 0 to 5

### Late Training (1500-3000 iterations)
- Stable bipedal walking
- Good velocity tracking
- Dynamic gait patterns
- Reward: 5 to 15+

## Tuning Tips

### If robot falls too much:
- Increase `termination_height` in config (currently 0.2m)
- Increase actuator stiffness/damping
- Increase `flat_orientation_reward_scale`

### If robot shuffles instead of walking:
- Increase `feet_air_time_reward_scale`
- Decrease `action_rate_reward_scale`

### If robot is too slow:
- Increase `commands_x_range` in config
- Increase `lin_vel_reward_scale`

### If training is unstable:
- Decrease learning rate
- Increase number of environments
- Reduce `action_scale`

## Key Design Decisions

1. **Position Control**: Chosen over torque control for stability and easier training
2. **URDF Runtime Conversion**: Avoids manual USD conversion, uses UrdfFileCfg
3. **Contact Sensors**: Essential for feet air time and stumble detection
4. **Randomization**: Physics and mass randomization for robust policies
5. **Reward Shaping**: Exponential rewards for tracking, L2 penalties for regulation
6. **Episode Length**: 20s allows learning full walking cycles without excessive computation

## Monitoring Training

Watch TensorBoard logs:
```bash
tensorboard --logdir logs/rsl_rl/rexai_flat_direct
```

Key metrics to monitor:
- `Episode_Reward/track_lin_vel_xy_exp`: Forward velocity tracking
- `Episode_Reward/feet_air_time`: Dynamic walking quality
- `Episode_Reward/flat_orientation_l2`: Balance quality
- `Episode_Termination/base_contact`: Falling frequency

## Next Steps

1. **Initial Training**: Train for 3000 iterations with 4096 envs (~2-4 hours)
2. **Evaluate**: Use play.py to visualize learned behavior
3. **Fine-tune**: Adjust reward scales based on observed behavior
4. **Advanced**: Add terrain curriculum, velocity command curriculum

## Technical Notes

- All files pass Python syntax validation ✓
- Follows Isaac Lab coding standards ✓
- Compatible with RSL-RL training pipeline ✓
- Modular and extensible design ✓

## Troubleshooting

### Import errors:
- Ensure Isaac Sim is properly installed
- Run `./isaaclab.sh -p` to verify environment

### URDF conversion fails:
- Check URDF file path is correct
- Verify meshes are in correct location
- Check file permissions

### Training crashes:
- Reduce number of environments
- Check GPU memory usage
- Verify Isaac Sim installation

---

**Created**: 2025-11-16
**Robot**: RexAI Bipedal (0.3m tall, 10 DoF)
**Framework**: Isaac Lab + RSL-RL
**Status**: Ready for training ✓
