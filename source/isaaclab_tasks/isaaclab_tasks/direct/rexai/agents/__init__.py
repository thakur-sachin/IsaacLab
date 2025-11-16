# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing agent configurations for RexAI environment."""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_AGENTS_METADATA = toml.load(
    os.path.join(os.path.dirname(__file__), "../../../../isaaclab_rl/config.toml")
)

##
# Register RSL-RL configurations
##

from .rsl_rl_ppo_cfg import *  # noqa: F401, F403
