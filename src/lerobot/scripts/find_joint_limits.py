#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.scripts.server.find_joint_limits \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue
```
"""

import time
from dataclasses import dataclass

import draccus
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,  # ensure SO101 is registered for CLI choices
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    gamepad,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,  # ensure SO101 leader is registered for CLI choices
)
import lerobot.teleoperators.keyboard  # noqa: F401  # ensure keyboard teleop is registered
from lerobot.utils.robot_utils import busy_wait


@dataclass
class FindJointLimitsConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second. By default, no limit.
    teleop_time_s: float = 30
    # Display all cameras on screen
    display_data: bool = False
    # Optional kinematics overrides (so we can get EE pose even with joint-space robots)
    urdf_path: str | None = None
    target_frame_name: str = "gripper"
    # Skip robot calibration prompts (use cached/current motor settings as-is)
    skip_calibration: bool = False


@draccus.wrap()
def find_joint_and_ee_bounds(cfg: FindJointLimitsConfig):
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect(calibrate=not cfg.skip_calibration)

    start_episode_t = time.perf_counter()
    robot_type = getattr(robot.config, "robot_type", "so101")
    if "so100" in robot_type or "so101" in robot_type:
        # Note to be compatible with the rest of the codebase,
        # we are using the new calibration method for so101 and so100
        robot_type = "so_new_calibration"
    # Initialize kinematics only if URDF info is available (prefer top-level flags, then robot fields)
    kinematics = None
    urdf_path = cfg.urdf_path if cfg.urdf_path else getattr(cfg.robot, "urdf_path", None)
    target_frame_name = (
        cfg.target_frame_name if cfg.target_frame_name else getattr(cfg.robot, "target_frame_name", "gripper")
    )
    if urdf_path:
        try:
            kinematics = RobotKinematics(urdf_path, target_frame_name)
        except Exception as e:
            print(f"Warning: Kinematics disabled (URDF/solver issue): {e}")

    # Initialize min/max values
    observation = robot.get_observation()
    joint_positions = np.array([observation[f"{key}.pos"] for key in robot.bus.motors])
    ee_pos = kinematics.forward_kinematics(joint_positions)[:3, 3] if kinematics else None

    max_pos = joint_positions.copy()
    min_pos = joint_positions.copy()
    max_ee = ee_pos.copy() if ee_pos is not None else None
    min_ee = ee_pos.copy() if ee_pos is not None else None

    last_print_t = 0.0
    while True:
        action = teleop.get_action()
        robot.send_action(action)

        observation = robot.get_observation()
        joint_positions = np.array([observation[f"{key}.pos"] for key in robot.bus.motors])
        ee_pos = kinematics.forward_kinematics(joint_positions)[:3, 3] if kinematics else None

        # Skip initial warmup period
        if (time.perf_counter() - start_episode_t) < 5:
            continue

        # Update min/max values
        if ee_pos is not None:
            max_ee = np.maximum(max_ee, ee_pos) if max_ee is not None else ee_pos.copy()
            min_ee = np.minimum(min_ee, ee_pos) if min_ee is not None else ee_pos.copy()
        max_pos = np.maximum(max_pos, joint_positions)
        min_pos = np.minimum(min_pos, joint_positions)

        # Print live EE position at ~5 Hz when available
        now_t = time.perf_counter()
        if ee_pos is not None and (now_t - last_print_t) > 0.2:
            print(f"joints: {np.round(joint_positions, 4).tolist()} EE pos {np.round(ee_pos, 4).tolist()}")
            last_print_t = now_t

        if time.perf_counter() - start_episode_t > cfg.teleop_time_s:
            if max_ee is not None and min_ee is not None:
                print(f"Max ee position {np.round(max_ee, 4).tolist()}")
                print(f"Min ee position {np.round(min_ee, 4).tolist()}")
            else:
                print("EE position bounds not available (no URDF provided).")
            print(f"Max joint pos position {np.round(max_pos, 4).tolist()}")
            print(f"Min joint pos position {np.round(min_pos, 4).tolist()}")
            break

        busy_wait(0.01)


if __name__ == "__main__":
    find_joint_and_ee_bounds()
