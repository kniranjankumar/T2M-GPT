import math
import numpy as np
import scipy.optimize
import pybullet as p
import pybullet_data
import sys
import time
import matplotlib.pyplot as plt
import pybullet_data
import argparse
import sys
import options.option_transformer as option_trans
import clip
import torch
import numpy as np
import models.vqvae as vqvae
import models.t2m_trans as trans
import warnings
from utils.motion_process import recover_from_ric

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)

plane_id = p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
humanoid_id = p.loadMJCF("/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/mjcf/amp_humanoid_pybullet.xml")[0]
humanoid_eff_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
humanoid_bodies = [p.getJointInfo(humanoid_id, i)[12].decode() for i in range(p.getNumJoints(humanoid_id))]
# humanoid_eff = [humanoid_bodies.index(name) for name in humanoid_eff_names]

humanoid_joints = [i for i in range(p.getNumJoints(humanoid_id)) if p.getJointInfo(humanoid_id, i)[2] != p.JOINT_FIXED]
motion_file_path = "/home/dan/Projects/dynamic_motion_imitation/T2M-GPT/motion.npy"
llm_xyz = np.load(motion_file_path)
kinematic_chain = [[0, 2, 5, 8, 11], # right leg
                   [0, 1, 4, 7, 10], # left leg
                   [0, 3, 6, 9, 12, 15], # spine
                   [9, 14, 17, 19, 21], # right arm
                   [9, 13, 16, 18, 20]] # left arm
humanoid_eff = [kinematic_chain[3][-1], kinematic_chain[4][-1], kinematic_chain[0][-1], kinematic_chain[1][-1]]
llm_xyz = llm_xyz.reshape(-1, 22, 3)
for i in range(llm_xyz.shape[0]):
    for j in range(llm_xyz.shape[1]):
        llm_xyz[i,j,:] = p.multiplyTransforms([0,0,0], [0,0.707,0,0.707], llm_xyz[i,j,:], [0,0,0,1])[0] 

right_leg_joint_angles = []
# for a,b in range(kinematic_chain[0][0:-1], kinematic_chain[0][1:]):
#     vector = llm_xyz[0,b,:] - llm_xyz[0,a,:]
#     right_leg_joint_angles.append(math.atan2(vector[2], vector[0]))
eff_pos = [llm_xyz[0,i,:] for i in humanoid_eff]
# print(humanoid_eff)
# assert False
colors = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]]
p.createConstraint(humanoid_id, -1, plane_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,0],[0,0,llm_xyz[0,0,1]],[0,0,0,1],[0,0,0,1])

for pos,color in zip(eff_pos, colors):
    print("creating marker at", pos)
    marker = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=color)
    ori = [0,0,0,1]
    offset = np.array([0,0,0.0])
    
    marker_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=marker, basePosition=np.array([pos[0],pos[2],pos[1]])+offset, baseOrientation=ori)
    # markers.append(marker_id)
    
while True:
    p.stepSimulation()
    time.sleep(0.1)
    