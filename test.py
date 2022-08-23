import numpy as np
import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)  # p.DIRECT for no visualization
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)

# Load Assets
p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])  # [x, y, z], + orientation(default upright orientation)
targid = p.loadURDF("humanoid/humanoid.urdf", [0, 0, 0], [0, 0, 0, 10])
obj_of_focus = targid

# joint_id = 5
# jlower = p.getJointInfo(targid, joint_id)[8]
# jupper = p.getJointInfo(targid, joint_id)[9]
# print(jlower)
# print(jupper)

for step in range(900):
    #joint_five_target = np.random.uniform(jlower, jupper)
    #p.setJointMotorControlArray(targid, [5], p.POSITION_CONTROL, targetPositions = [joint_five_target])

    focus_position, __ = p.getBasePositionAndOrientation(targid)
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=focus_position)
    p.stepSimulation()
    time.sleep(.01)
