import pybullet as p, pybullet_data as pd, time, math

urdf = "/home/ben/all_projects/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.resetDebugVisualizerCamera(1.0, 45, -30, [0,0,0])
rb = p.loadURDF(urdf, useFixedBase=True)

name_to_idx = {p.getJointInfo(rb,i)[1].decode(): i for i in range(p.getNumJoints(rb))}
# SO101 joint names in URDF are "1","2","3","4","5","6"
def set_deg(j1,j2,j3,j4,j5,j6):
    deg = [j1,j2,j3,j4,j5,j6]
    for i,name in enumerate(["1","2","3","4","5","6"]):
        p.resetJointState(rb, name_to_idx[name], math.radians(deg[i]))
    p.stepSimulation()

# Example: neutral-ish
set_deg(0,0,0,0,0,0)
time.sleep(0.5)
# Try a few poses
# set_deg(10,20,-30,15,0,0); time.sleep(0.5)
# set_deg(-20,10,20,-10,0,0); time.sleep(0.5)

# joints: [-5.5881, -99.8286, 24.9206, 99.8254, 0.8342, 0.1384] EE pos [0.0206, -0.0233, 0.2633]
# set_deg(-5.5881, -99.8286, 24.9206, 99.8254, 0.8342, 0.1384)  # looks good
# joints: [-5.2755, 38.5604, 34.453, -11.4797, 0.7821, 0.1384] EE pos [0.0386, -0.2182, 0.0384]
set_deg(-5.2755, 38.5604, 34.453, -11.4797, 0.7821, 0.1384)  # looks good

# Print current EE (use 'gripper' link)
gripper_idx = [i for i in range(p.getNumJoints(rb)) if p.getJointInfo(rb,i)[12].decode()=="gripper"][0]
print("EE world pos:", p.getLinkState(rb, gripper_idx)[4])
input("Press Enter to exit")