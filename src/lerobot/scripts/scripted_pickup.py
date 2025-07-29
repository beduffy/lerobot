"""
A scripted policy to pick up a cube in the simulation by following a joint-space trajectory.
"""

import time
import mujoco
import mujoco.viewer
import numpy as np
import draccus
from dataclasses import dataclass

@dataclass
class ScriptConfig:
    mjcf_path: str = "lerobot/standalone_scene.xml"
    max_iter: int = 10000

@draccus.wrap()
def scripted_pickup(cfg: ScriptConfig):
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path(cfg.mjcf_path)
    data = mujoco.MjData(model)

    # Get the joint names for the robot.
    mujoco_joint_names = [model.joint(i).name for i in range(model.njnt)]
    joint_ids = [mujoco_joint_names.index(str(i)) for i in range(1, 7)]
    gripper_joint_id = mujoco_joint_names.index("6")

    # Define key poses in joint space (in degrees).
    # These will need to be manually tuned.
    # For now, let's define a simple sequence to test the concept.
    home_pos_deg = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
    pre_grasp_pos_deg = np.array([0, 45, -45, 0, 45, 0], dtype=np.float64) # Example pose
    grasp_pos_deg = np.array([0, 60, -60, 0, 60, 0], dtype=np.float64) # Example pose
    lift_pos_deg = np.array([0, 45, -45, 0, 45, 0], dtype=np.float64) # Example pose

    trajectory = [home_pos_deg, pre_grasp_pos_deg, grasp_pos_deg, grasp_pos_deg, lift_pos_deg]
    gripper_states = [0.0, 0.0, 1.0, 1.0, 0.0]

    current_target_idx = 0
    current_joint_pos_deg = np.zeros(6, dtype=np.float64)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and current_target_idx < len(trajectory):
            target_joint_pos_deg = trajectory[current_target_idx]

            # Simple interpolation towards the target joint positions.
            error_deg = target_joint_pos_deg - current_joint_pos_deg
            if np.linalg.norm(error_deg) > 0.5:
                current_joint_pos_deg += error_deg / np.linalg.norm(error_deg) * 0.5
            else:
                current_joint_pos_deg = target_joint_pos_deg
                print(f"Reached waypoint {current_target_idx}")
                current_target_idx += 1
                time.sleep(1)

            # Set joint positions
            start_joint_idx = joint_ids[0]
            end_joint_idx = joint_ids[-1] + 1
            data.qpos[start_joint_idx:end_joint_idx] = np.deg2rad(current_joint_pos_deg)
            if current_target_idx < len(gripper_states):
                 data.qpos[gripper_joint_id] = gripper_states[current_target_idx]

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1 / 240.0)

    print("Script finished.")

if __name__ == "__main__":
    scripted_pickup() 