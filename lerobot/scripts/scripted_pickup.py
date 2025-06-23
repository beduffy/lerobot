"""
A scripted policy to pick up a cube in the simulation using inverse kinematics.
"""

import time
import mujoco
import mujoco.viewer
import numpy as np
import draccus
from dataclasses import dataclass

from lerobot.common.model.kinematics import RobotKinematics


@dataclass
class ScriptConfig:
    mjcf_path: str = "lerobot/standalone_scene.xml"
    robot_type: str = "so_new_calibration"
    max_iter: int = 10000


@draccus.wrap()
def scripted_pickup(cfg: ScriptConfig):
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path(cfg.mjcf_path)
    data = mujoco.MjData(model)

    # Initialize the kinematics model.
    kinematics = RobotKinematics(robot_type=cfg.robot_type)

    # Get the joint names for the robot.
    mujoco_joint_names = [model.joint(i).name for i in range(model.njnt)]
    # Map to indices for joints "1" through "6"
    joint_ids = [mujoco_joint_names.index(str(i)) for i in range(1, 7)]
    gripper_joint_id = mujoco_joint_names.index("6")


    # Launch the viewer.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Get the initial position of the cube.
        cube_body_id = model.body("cube").id
        cube_pos = data.body(cube_body_id).xpos
        print(f"Cube initial position: {cube_pos}")

        # Define a sequence of target poses for the end-effector.
        # 1. Move above the cube
        target_pos_above = cube_pos + np.array([0, 0, 0.1])
        # 2. Move to the cube
        target_pos_at = cube_pos.copy()
        target_pos_at[2] += 0.045 # a bit above to grab
        # 3. Lift the cube
        target_pos_lift = target_pos_at + np.array([0, 0, 0.1])

        # Define a simple state machine for the pickup sequence.
        # Each pose is a 4x4 transformation matrix.
        # For simplicity, we only care about position, so rotation is identity.
        target_poses = []
        for pos in [target_pos_above, target_pos_at, target_pos_at, target_pos_lift, target_pos_lift]:
            pose = np.eye(4)
            pose[:3, 3] = pos
            target_poses.append(pose)

        # Gripper states: 0 for closed, 1 for open
        # We need to find what joint value corresponds to open/closed
        # from the XML, joint "6" has range -0.17 to 1.74 rad
        # Let's say open is 0.0 and closed is 1.0
        gripper_states = [0.0, 0.0, 1.0, 1.0, 0.0]  # open, open, close, close, open

        current_target_idx = 0
        current_joint_pos_deg = np.zeros(6)

        # Simulation loop
        for i in range(cfg.max_iter):
            if current_target_idx >= len(target_poses):
                print("Sequence finished.")
                break

            target_ee_pose = target_poses[current_target_idx]

            # Use IK to get the target joint angles.
            # `ik` returns angles in degrees.
            q_target_deg = kinematics.ik(
                current_joint_pos=current_joint_pos_deg,
                desired_ee_pose=target_ee_pose,
                position_only=True
            )

            # Interpolate to smoothly move to the target joint positions
            # For simplicity, let's just step towards the target
            # A more advanced implementation would use a proper motion planner
            step_size_deg = 0.5  # degrees per simulation step
            error_deg = q_target_deg - current_joint_pos_deg
            
            if np.linalg.norm(error_deg) > step_size_deg:
                # Move towards the target by one step
                update_deg = error_deg / np.linalg.norm(error_deg) * step_size_deg
                current_joint_pos_deg += update_deg
            else:
                # We are close enough, snap to target
                current_joint_pos_deg = q_target_deg

            q_rad = np.deg2rad(current_joint_pos_deg)

            for j in range(6):
                 data.qpos[joint_ids[j]] = q_rad[j]
            
            # Set gripper state
            data.qpos[gripper_joint_id] = gripper_states[current_target_idx]

            mujoco.mj_step(model, data)
            viewer.sync()

            # Check if we have reached the target pose
            current_ee_pose = kinematics.forward_kinematics(current_joint_pos_deg)
            pos_error = np.linalg.norm(current_ee_pose[:3, 3] - target_ee_pose[:3, 3])
            
            if pos_error < 0.01:
                print(f"Reached target {current_target_idx}")
                current_target_idx += 1
                if current_target_idx < len(target_poses):
                    # Pause at each waypoint before moving to the next
                    time.sleep(1)
                

            time.sleep(1 / 60.0)

        print("Closing viewer.")
        viewer.close()

if __name__ == "__main__":
    scripted_pickup() 