
- let's start extremely simple and just make ANYTHING WORK. I will try
    1. find joint limits now
    2. prove actor + learner can be run locally and do something
    3. prove at least reward classifier can begin training and collect initial data to train this
    4. prove intervention works. when i ran in simulation it wasn't so easy.
    5. prove my so101 leader arm can follow the follower to make interventions easier
    6. prove gamepad or keypad and FK/IK works.


1) Find safe workspace limits

python -m lerobot.scripts.find_joint_limits \
  --robot.type=so100_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0

python -m lerobot.scripts.find_joint_limits \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue

should work
python -m lerobot.find_joint_limits \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm

with keyboard control, curious if FK/IK works
python -m lerobot.scripts.find_joint_limits \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=keyboard

so100
python -m lerobot.scripts.find_joint_limits \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=keyboard

(robosuite) ✘-1 ~/all_projects/lerobot/lerobot [main|✚ 3…7⚑ 3] 
17:49 $ python -m lerobot.scripts.find_joint_limits     --robot.type=so101_follower     --robot.port=/dev/ttyACM0     --robot.id=my_awesome_follower_arm     --teleop.type=keyboard
usage: find_joint_limits.py [-h] [--config_path str] [--teleop str]
                            [--teleop.type {gamepad,koch_leader,so100_leader}]
                            [--teleop.use_gripper str]
                            [--teleop.gripper_open_pos str] [--teleop.id str]
                            [--teleop.calibration_dir str] [--teleop.port str]
                            [--robot str]
                            [--robot.type {koch_follower,so100_follower,so100_follower_end_effector}]
                            [--robot.id str] [--robot.calibration_dir str]
                            [--robot.port str]
                            [--robot.disable_torque_on_disconnect str]
                            [--robot.max_relative_target str]
                            [--robot.cameras str] [--robot.use_degrees str]
                            [--robot.urdf_path str]
                            [--robot.target_frame_name str]
                            [--robot.end_effector_bounds str]
                            [--robot.max_gripper_pos str]
                            [--robot.end_effector_step_sizes str]
                            [--teleop_time_s str] [--display_data str]
find_joint_limits.py: error: argument --robot.type: invalid choice: 'so101_follower' (choose from 'koch_follower', 'so100_follower', 'so100_follower_end_effector')

