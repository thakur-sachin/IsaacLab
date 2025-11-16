#!/usr/bin/env python3
"""Test script to verify RexAI environment registration."""

import gymnasium as gym

# Import isaaclab tasks to trigger registration
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.rexai import agents  # noqa: F401

def main():
    """Test if RexAI environment is properly registered."""
    print("=" * 80)
    print("Testing RexAI Environment Registration")
    print("=" * 80)

    # Check if the environment is registered
    try:
        env_id = "Isaac-RexAI-Direct-v0"
        print(f"\n1. Checking if '{env_id}' is registered...")

        if env_id in gym.envs.registry:
            print(f"   ✓ Environment '{env_id}' is registered!")

            # Get the entry point
            spec = gym.spec(env_id)
            print(f"   Entry point: {spec.entry_point}")
            print(f"   Kwargs: {spec.kwargs}")

            # Try to get the environment configuration
            print(f"\n2. Attempting to load environment configuration...")
            env_cfg_entry = spec.kwargs.get("env_cfg_entry_point")
            if env_cfg_entry:
                print(f"   Environment config: {env_cfg_entry}")

            rsl_rl_cfg_entry = spec.kwargs.get("rsl_rl_cfg_entry_point")
            if rsl_rl_cfg_entry:
                print(f"   RSL-RL config: {rsl_rl_cfg_entry}")

            print(f"\n✓ All checks passed! Environment is ready for training.")
            print(f"\nYou can now train with:")
            print(f"  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task={env_id} --num_envs 10")

        else:
            print(f"   ✗ Environment '{env_id}' is NOT registered!")
            print(f"\n   Available Isaac Lab Direct environments:")
            isaac_envs = [env for env in gym.envs.registry.keys() if "Isaac" in env and "Direct" in env]
            for env in sorted(isaac_envs)[:10]:
                print(f"     - {env}")
            return False

    except Exception as e:
        print(f"   ✗ Error during registration check: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("=" * 80)
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
