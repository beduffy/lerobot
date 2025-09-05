
- let's start extremely simple and just make ANYTHING WORK. I will try
    1. find joint limits now
    2. prove actor + learner can be run locally and do something
    3. prove at least reward classifier can begin training and collect initial data to train this
    4. prove intervention works. when i ran in simulation it wasn't so easy.
    5. prove my so101 leader arm can follow the follower to make interventions easier
    6. prove gamepad or keypad and FK/IK works.


# 1) Find safe workspace limits

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
python -m lerobot.scripts.find_joint_limits \
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

seemed to run, just need to connect
python -m lerobot.scripts.find_joint_limits \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm \
  --teleop_time_s 60

same
python -m lerobot.scripts.find_joint_limits \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so101 \
  --teleop.type=gamepad \
  --teleop.id=pad1 \
  --teleop_time_s 60

worked
python -m lerobot.scripts.find_joint_limits \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm \
  --urdf_path /home/ben/all_projects/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --target_frame_name gripper \
  --teleop_time_s 60

PYTHONPATH=/home/ben/all_projects/lerobot/src python -m lerobot.scripts.find_joint_limits \
  --robot.type=so100_follower_end_effector \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so101 \
  --robot.urdf_path /home/ben/all_projects/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --robot.target_frame_name gripper \
  --robot.end_effector_bounds '{"min":[-0.2,-0.2,0.05],"max":[0.2,0.2,0.25]}' \
  --robot.end_effector_step_sizes '{"x":0.02,"y":0.02,"z":0.02}' \
  --teleop.type=keyboard_ee \
  --teleop_time_s 60

PYTHONPATH=/home/ben/all_projects/lerobot/src python -m lerobot.scripts.find_joint_limits \
  --robot.type=so100_follower_end_effector \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so101 \
  --robot.urdf_path /home/ben/all_projects/SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --robot.target_frame_name gripper \
  --robot.end_effector_bounds '{"min":[-0.2,-0.2,0.05],"max":[0.2,0.2,0.25]}' \
  --robot.end_effector_step_sizes '{"x":0.02,"y":0.02,"z":0.02}' \
  --teleop.type=keyboard_ee \
  --skip_calibration true \
  --teleop_time_s 60

PYTHONPATH=/home/ben/all_projects/lerobot/src python -m lerobot.scripts.find_joint_limits   --robot.type=so101_follower   --robot.port=/dev/ttyACM0   --robot.id=my_awesome_follower_arm   --teleop.type=so101_leader   --teleop.port=/dev/ttyACM1   --teleop.id=my_awesome_leader_arm   --urdf_path /home/ben/all_projects/SO-ARM100/Simulation/SO101/so101_new_calib.urdf   --target_frame_name gripper   --skip_calibration true   --teleop_time_s 60


