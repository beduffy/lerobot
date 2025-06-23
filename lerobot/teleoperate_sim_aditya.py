
"""
Copied from here: https://github.com/adityakamath/lerobot/blob/mujoco-teleop/lerobot/teleoperate_sim.py
Simulated teleoperation: reads teleoperator hardware input and uses it to control a Mujoco simulation.
No physical robot is connected or commanded. Use this to control a robot in simulation.

Example usage:

mjpython -m lerobot.teleoperate_sim \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodemXXXX \
  --teleop.id=my_leader \
  --mjcf_path=path/to/your_robot.xml \
  --display_data=true

python -m lerobot.teleoperate_sim_aditya \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --mjcf_path=/home/ben/all_projects/SO-ARM100/Simulation/SO101/so101_new_calib.xml \
    --display_data=true
"""

import time
import logging
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import rerun as rr
import mujoco
import mujoco.viewer
from lerobot.common.robots import RobotConfig
from lerobot.common.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.common.utils.utils import init_logging
from lerobot.common.utils.visualization_utils import _init_rerun
import numpy as np

from .common.teleoperators import koch_leader, so100_leader, so101_leader  # noqa: F401

print('Mujoco version:', mujoco.__version__)

@dataclass
class TeleoperateSimConfig:
    teleop: TeleoperatorConfig
    mjcf_path: str
    fps: int = 10
    display_data: bool = False

@draccus.wrap()
def teleoperate_sim(cfg: TeleoperateSimConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation_sim")

    # Load Mujoco model
    model = mujoco.MjModel.from_xml_path(cfg.mjcf_path)
    data = mujoco.MjData(model)

    # Map Mujoco joint names ("1", "2", ..., "6") to indices
    mujoco_joint_names = [model.joint(i).name for i in range(model.njnt)]
    print("Mujoco joint names:", mujoco_joint_names)
    mujoco_indices = [mujoco_joint_names.index(str(i)) for i in range(1, 7)]

    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop.connect()

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                action = teleop.get_action()
                # Create a dummy action with random values since no leader arm is connected.
                # The values are in degrees, and will be converted to radians.
                # A range of [-90, 90] degrees seems reasonable for random movements.
                # action_values = np.random.uniform(-90, 90, 6)
                # action = {f"joint_{i+1}": v for i, v in enumerate(action_values)}

                # Map the first 6 teleop joint values (in order) to Mujoco joints "1"-"6"
                joint_values = list(action.values())[:6]
                # Convert from degrees to radians before sending to Mujoco
                joint_values = np.deg2rad(joint_values)
                for idx, val in zip(mujoco_indices, joint_values):
                    data.qpos[idx] = val
                mujoco.mj_step(model, data)
                viewer.sync()
                if cfg.display_data:
                    for i, val in enumerate(joint_values, 1):
                        rr.log(f"action_{i}", rr.Scalar(val))
                    print("Simulated Joint States (action):")
                    for i, v in enumerate(joint_values, 1):
                        print(f"  {i}: {v}")
                    print("-" * 40)
                time.sleep(1.0 / cfg.fps)
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        # teleop.disconnect()

if __name__ == "__main__":
    teleoperate_sim()