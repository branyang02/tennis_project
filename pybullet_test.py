"""
Created on Sun Mar 15 19:51:40 2020
@author: linux-asd
"""

import pybullet as p
import numpy as np
import time
import pybullet_data
from pybullet_debuger import pybulletDebug  
from kinematic_model import robotKinematics
from gaitPlanner import trotGait

"""Connect to rhe physics client, you can choose GUI or DIRECT mode"""
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
"""set Gravity"""
p.setGravity(0,0,-9.8)

"""import the robot's URDF file, if it is fixed no ground plane is needed""" 
cubeStartPos = [0,0,0.2]
FixedBase = False #if fixed no plane is imported
if (FixedBase == False):
    p.loadURDF("plane.urdf")
boxId = p.loadURDF("4leggedRobot.urdf",cubeStartPos, useFixedBase=FixedBase)

jointIds = []
paramIds = [] 
time.sleep(0.5)