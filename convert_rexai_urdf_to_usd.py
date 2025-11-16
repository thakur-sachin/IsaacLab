#!/usr/bin/env python3
"""
Script to convert RexAI URDF to USD using IsaacLab utilities.
This script uses the urdf importer from Isaac Sim through IsaacLab.
"""

import argparse
from pathlib import Path

# Import Isaac Lab utilities
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main():
    # Define paths
    urdf_path = Path("source/isaaclab_assets/isaaclab_assets/robots/RexAI/Assets/Trex/Trex.urdf")
    usd_path = Path("source/isaaclab_assets/isaaclab_assets/robots/RexAI/Assets/Trex/Trex.usd")

    # Create converter configuration
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=str(urdf_path.absolute()),
        usd_dir=str(usd_path.parent.absolute()),
        usd_file_name=usd_path.name,
        fix_base=False,  # Bipedal robot should not have fixed base
        merge_fixed_joints=True,
        force_usd_conversion=True,
        make_instanceable=False,
    )

    # Create converter and convert
    urdf_converter = UrdfConverter(urdf_converter_cfg)

    # Convert URDF to USD
    print(f"Converting URDF from: {urdf_path}")
    print(f"Output USD to: {usd_path}")
    usd_path_result = urdf_converter.convert()
    print(f"✓ Conversion successful! USD file created at: {usd_path_result}")


if __name__ == "__main__":
    main()
