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
import os
import options.option_transformer as option_trans
import clip
import torch
import numpy as np
# import numpy as np
from scipy.spatial.transform import Rotation as R
import models.vqvae as vqvae
import models.t2m_trans as trans
import warnings
from utils.motion_process import recover_from_ric
from IPython.display import HTML
import base64
import visualization.plot_3d_global as plot_3d
warnings.filterwarnings('ignore')
args = option_trans.get_args_parser()
print(args.instruction)
# parser = argparse.ArgumentParser()
# parser.add_argument("motion_file", help="path to the motion file", type=str)#, default="amp_humanoid.hop.npy")
# args = parser.parse_args()

args.dataname = 't2m'
args.resume_pth = 'pretrained/VQVAE/net_last.pth'
args.resume_trans = 'pretrained/VQTransformer_corruption05/net_best_fid.pth'
args.down_t = 2
args.depth = 3
args.block_size = 51
kinematic_chain = [[0, 2, 5, 8, 11], # right leg
                   [0, 1, 4, 7, 10], # left leg
                   [0, 3, 6, 9, 12, 15], # spine
                   [9, 14, 17, 19, 21], # right arm
                   [9, 13, 16, 18, 20]] # left arm

class DebugDuckie:
    def __init__(self, position, body_id, link_id):
        self.position = position
        self.body_id = body_id
        self.link_id = link_id
        self.duck_id = p.loadURDF("duck_vhacd.urdf",self.position,[0,0,0,1], useFixedBase=True, globalScaling=5.0)
        self.base_rotation = p.getQuaternionFromEuler([np.pi/2,0,np.pi])
        orientation = p.getLinkState(self.body_id, self.link_id)[5]
        _, duckie_orientation = p.multiplyTransforms([0,0,0], orientation, [0,0,0], self.base_rotation)
        p.resetBasePositionAndOrientation(self.duck_id, self.position, duckie_orientation)


    
    def update(self, orientation=None):
        if orientation is None:
            orientation = p.getLinkState(self.body_id, self.link_id)[5]
        _, duckie_orientation = p.multiplyTransforms([0,0,0], orientation, [0,0,0], self.base_rotation)
        p.resetBasePositionAndOrientation(self.duck_id, self.position, duckie_orientation)

def retarget_humanoid_to_digit_ik(marker_positions, marker_orientations, digit_eff_idxs, digit_id, initPose):
    # curr_orientation = [p.getLinkState(digit_id, digit_eff_idx)[5] for digit_eff_idx in digit_eff_idxs]
    # curr_orientation = np.array(curr_orientation)
    # curr_orientation[2,:] = marker_orientations[2]
    # curr_orientation[3,:] = marker_orientations[3]
    joint_pose = p.calculateInverseKinematics2(digit_id, digit_eff_idxs,
                                                    marker_positions,
                                                    # marker_orientations,
                                                    # lowerLimits=[-1]*22,
                                                    # upperLimits=[1]*22,
                                                    restPoses=initPose,
                                                    # solver=p.IK_DLS,
                                                    # maxNumIterations=1000,
                                                    # residualThreshold=1e-8,
                                                    jointDamping=[0.01]*22)
    # joint_pose = [0]*p.getNumJoints(digit_id)
    joint_pose = np.array(joint_pose)
    # joint_pose*=0
    joint_pose_left_leg = p.calculateInverseKinematics(digit_id, digit_eff_idxs[3],
                                                        marker_positions[3],
                                                        marker_orientations[3],
                                                        restPoses=initPose)
    joint_pose_right_leg = p.calculateInverseKinematics(digit_id, digit_eff_idxs[2],
                                                        marker_positions[2],
                                                        marker_orientations[2],
                                                        restPoses=initPose)
    joint_pose_left_arm = p.calculateInverseKinematics(digit_id, digit_eff_idxs[1],
                                                        marker_positions[1],
                                                        marker_orientations[1],
                                                        restPoses=initPose)
    joint_pose_right_arm = p.calculateInverseKinematics(digit_id, digit_eff_idxs[0],
                                                        marker_positions[0],
                                                        marker_orientations[0],
                                                        restPoses=initPose)
    
    left_leg_joints = ['hip_abduction_left', 'hip_rotation_left', 'hip_flexion_left', 'knee_joint_left', 'shin_to_tarsus_left', 'toe_pitch_joint_left', 'toe_roll_joint_left']
    right_leg_joints = ['hip_abduction_right', 'hip_rotation_right', 'hip_flexion_right', 'knee_joint_right', 'shin_to_tarsus_right', 'toe_pitch_joint_right', 'toe_roll_joint_right']
    left_arm_joints = ['shoulder_roll_joint_left',  'shoulder_pitch_joint_left', 'shoulder_yaw_joint_left', 'elbow_joint_left']
    right_arm_joints = ['shoulder_roll_joint_right', 'shoulder_pitch_joint_right', 'shoulder_yaw_joint_right', 'elbow_joint_right']
    # right_arm_joints = ['hip_abduction_right', 'hip_rotation_right', 'hip_flexion_right', 'knee_joint_right', 'knee_to_shin_right', 'shin_to_tarsus_right', 'toe_pitch_joint_right', 'toe_roll_joint_right']
    joints = [p.getJointInfo(digit_id, i)[1].decode() for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[3] > -1]
    # print(joints)
    left_leg_idxs = [joints.index(name) for name in left_leg_joints]
    right_leg_idxs = [joints.index(name) for name in right_leg_joints]
    left_arm_idxs = [joints.index(name) for name in left_arm_joints]
    right_arm_idxs = [joints.index(name) for name in right_arm_joints]
    # left_leg_idxs = [i for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[1].decode() in left_leg_joints]
    # right_leg_idxs = [i for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[1].decode() in right_leg_joints]
    # left_arm_idxs = [i for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[1].decode() in left_arm_joints]
    # right_arm_idxs = [i for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[1].decode() in right_arm_joints]
    # print(left_leg_idxs, right_leg_idxs,left_arm_idxs, right_arm_idxs, len(joint_pose))
    # print(len(joint_pose_left_leg), len(joint_pose_right_leg))
    
    # print(np.array(joint_pose_leg)-np.array(joint_pose))
    joint_pose_left_leg = np.array(joint_pose_left_leg)
    joint_pose_right_leg = np.array(joint_pose_right_leg)
    joint_pose_left_arm = np.array(joint_pose_left_arm)
    joint_pose_right_arm = np.array(joint_pose_right_arm)
    
    joint_pose[right_leg_idxs] = joint_pose_right_leg[right_leg_idxs]
    joint_pose[left_leg_idxs] = joint_pose_left_leg[left_leg_idxs]
    # joint_pose[right_arm_idxs] = joint_pose_right_arm[right_arm_idxs]
    # joint_pose[left_arm_idxs] = joint_pose_left_arm[left_arm_idxs]
    # print(right_arm_idxs, left_arm_idxs, right_leg_idxs, left_leg_idxs)
    
    return  np.array(joint_pose)

def get_error(marker_id, digit_eff_idxs, digit_id):
    eff_pos = [p.getLinkState(digit_id, idx)[4] for idx in digit_eff_idxs]
    eff_pos = np.array(eff_pos)
    marker_positions = np.array([p.getBasePositionAndOrientation(marker)[0] for marker in marker_id])
    return np.linalg.norm(eff_pos-marker_positions, axis=1)

def get_quat_error(quat1,quat2):
    error_q = p.multiplyTransforms([0,0,0], quat1, [0,0,0], p.invertTransform([0,0,0], quat2)[1])[1]
    angles = p.getEulerFromQuaternion(error_q)
    # print(angles)
    
    return angles

def create_markers_from_id(agent_id, body_idxs, radius=0.05,colors=[[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]]):
    markers = []
    for color, body_idx in zip(colors,body_idxs):
        pos, ori = p.getLinkState(agent_id, body_idx)[4:6]
        offset = np.array([0,0,0.2])
        marker = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        marker_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=marker, basePosition=np.array(pos)+offset, baseOrientation=ori)
        markers.append(marker_id)
    return markers   

def create_markers_from_eff_pos(eff_pos, radius=0.05, colors=[[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]]):
    markers = []
    for pos,color in zip(eff_pos, colors):
        marker = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        ori = [0,0,0,1]
        offset = np.array([0,-1,0.0])
        
        marker_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=marker, basePosition=np.array([pos[0],pos[2],pos[1]])+offset, baseOrientation=ori)
        markers.append(marker_id)
    return markers   

def drawVector(vector):
    p.addUserDebugLine([0,0,0], vector*10, [1,0,0], 5)

def get_quaterion_from_torso_points(torso_points):
    torso_points = list(torso_points)
    ori_vec = np.array(get_orientation_vector(*torso_points))
    relative_quaternion = p.getQuaternionFromAxisAngle(ori_vec, np.pi)
    # drawVector(ori_vec)
    # ori_vec_ref = np.array([0,0,1])
    # rotation_matrix = (R.align_vectors(ori_vec_ref.reshape(1, -1), ori_vec.reshape(1, -1))[0]).as_matrix()
    # relative_quaternion = R.from_matrix(rotation_matrix).as_quat()
    return relative_quaternion

def get_orientation_vector(point1, point2, point3):
    # Calculate the vectors between the points
    v1 = point2 - point1
    v2 = point3 - point1
    
    # Calculate the cross product of the vectors
    cross_product = np.cross(v1, v2)
    # Normalize the cross product
    norm_cross_product = cross_product / np.linalg.norm(cross_product)
    return norm_cross_product
    
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)

################################################################################
########################### Load digit and humanoid ############################
################################################################################


plane_id = p.loadURDF("plane.urdf",[0,0,0],[0,0,0,1])
digit_id = p.loadURDF("/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/urdf/DigitRobot/DigitRobot/urdf/digit_model.urdf",[0,-1,1.1],[0,0,0,1])#, useFixedBase=True)
# digit_id = p.loadURDF("../assets/urdf/digit_description-main/urdf/digit_float.urdf",[0,-1,1.1],[0,0,0,1])
humanoid_id = p.loadMJCF("/home/dan/Projects/dynamic_motion_imitation/IsaacGymEnvs/assets/mjcf/amp_humanoid_pybullet.xml")[0]

humanoid_eff_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
humanoid_bodies = [p.getJointInfo(humanoid_id, i)[12].decode() for i in range(p.getNumJoints(humanoid_id))]
humanoid_eff = [humanoid_bodies.index(name) for name in humanoid_eff_names]
humanoid_joints = [i for i in range(p.getNumJoints(humanoid_id)) if p.getJointInfo(humanoid_id, i)[2] != p.JOINT_FIXED]

digit_eff_names = ["right_hand", "left_hand", "right_toe_roll", "left_toe_roll"]
digit_bodies = [p.getJointInfo(digit_id, i)[12].decode() for i in range(p.getNumJoints(digit_id))]
digit_eff = [digit_bodies.index(name) for name in digit_eff_names]

for joint in range(p.getNumJoints(digit_id)):
    p.setJointMotorControl2(digit_id, joint, p.POSITION_CONTROL, targetVelocity=1, force=10)
for joint in range(p.getNumJoints(humanoid_id)):
        p.setJointMotorControl2(humanoid_id, joint, p.POSITION_CONTROL, targetVelocity=1, force=10)
p.setJointMotorControlArray(digit_id, [i for i in range(p.getNumJoints(digit_id))], p.POSITION_CONTROL, targetPositions=[0]*p.getNumJoints(digit_id))


################################################################################
#################################### Generate motion from prompt ################################
################################################################################
# motion_file = "amp_humanoid_run.npy"

# motion_file_path = "/home/dan/Projects/dynamic_motion_imitation/T2M-GPT/motion.npy"
load_motion = True
scaling_factor = 0.9
instruction = '_'.join(args.instruction)
save_gif=False
# print(instruction)
motion_file_path = "/home/dan/Projects/dynamic_motion_imitation/T2M-GPT/motions/"+instruction+".npy"

if load_motion and os.path.isfile(motion_file_path):
    print("loading file")
    xyz = np.load(motion_file_path).reshape(-1,22,3)
    # display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))
else:
    clip_text = [instruction]
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False, download_root='./')  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p_ in clip_model.parameters():
        p_.requires_grad = False

    net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate)


    trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code,
                                    embed_dim=1024,
                                    clip_dim=args.clip_dim,
                                    block_size=args.block_size,
                                    num_layers=9,
                                    n_head=16,
                                    drop_out_rate=args.drop_out_rate,
                                    fc_rate=args.ff_rate)


    print ('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.cuda()

    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
    trans_encoder.eval()
    trans_encoder.cuda()

    mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
    std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()

    text = clip.tokenize(clip_text, truncate=True).cuda()
    feat_clip_text = clip_model.encode_text(text).float()
    index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
    pred_pose = net.forward_decoder(index_motion)

    pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22)
    xyz = pred_xyz.detach().cpu().numpy().reshape(-1, 22, 3)
    np.save(motion_file_path, xyz)
if save_gif:
    pose_vis = plot_3d.draw_to_batch(xyz.reshape(1,-1,22,3),instruction, [instruction+'.gif'])
    print("saved gif")
    b64 = base64.b64encode(open(instruction+'.gif','rb').read()).decode('ascii')

kinematic_chain = [[0, 2, 5, 8, 11], # right leg
                   [0, 1, 4, 7, 10], # left leg
                   [0, 3, 6, 9, 12, 15], # spine
                   [9, 14, 17, 19, 21], # right arm
                   [9, 13, 16, 18, 20]] # left arm
humanoid_eff = [kinematic_chain[4][-1], kinematic_chain[3][-1], kinematic_chain[1][-1], kinematic_chain[0][-1]]
print(humanoid_eff)
llm_xyz = np.zeros_like(xyz)
# for j in range(llm_xyz.shape[1]):
#     # print(j)
#     # if j == 11 or j == 10:
#     #     llm_xyz[:,j,:] =np.array([[[1,scaling_factor,1]]])*(xyz[:,j,:]- xyz[:,0,:])
#     #     llm_xyz[:,j,:] =np.array([[[1,scaling_factor,1]]])*(xyz[:,j,:]- xyz[:,0,:])
#     # else:
#     llm_xyz[:,j,:] = (xyz[:,j,:]- xyz[:,0,:])
#     llm_xyz[:,j,1] += xyz[:,0,1]
llm_xyz = xyz
torso_points = np.stack([llm_xyz[:,i,:] for i in [9,14,13]], axis=1)
llm_xyz[:,:,1] +=0.09
# llm_xyz[:,:,1] += 0.15
# llm_xyz[:,10,1] -=0.11
# llm_xyz[:,11,1] -=0.11
height_offset = 0.2
# llm_xyz[:,:,1] -=  height_offset
# assert False
# print(llm_xyz[:,21,:])

for i in range(llm_xyz.shape[0]):
    for j in range(llm_xyz.shape[1]):
        llm_xyz[i,j,:] = p.multiplyTransforms([0,0,0], [0,0.707,0,0.707], llm_xyz[i,j,:], [0,0,0,1])[0]
# ori_vec = -np.array(get_orientation_vector(*torso_points))
# print(ori_vec)
# ori_vec_ref = np.array([1,0,0])

# # Calculate the rotation matrix that rotates ori_vec_ref to ori_vec
# rotation_matrix = (R.align_vectors(ori_vec_ref.reshape(1, -1), ori_vec.reshape(1, -1))[0]).as_matrix()
# # print(rotation_matrix.as_matrix())
# # Convert the rotation matrix to a quaternion
# relative_quaternion = R.from_matrix(rotation_matrix).as_quat()

# print(relative_quaternion)
# assert False

        # p.multiplyTransforms([0,0,0], [0.707,0,0,0.707], llm_xyz[i,j,:], [0,0,0,1])[0] 
# print(llm_xyz[:,11,0])
# assert False
############################################################################################################
###################################### Mount humanoid and digit to rack ####################################
############################################################################################################
quat = get_quaterion_from_torso_points(torso_points[0])
# p.createConstraint(digit_id, -1, plane_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,llm_xyz[0,0,1]]+np.array([0,-1,height_offset]),[0,0,0,1],[0,0,0,1])
# p.createConstraint(humanoid_id, -1, plane_id, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,llm_xyz[0,0,1]]+np.array([0,0,0.15]),[0,0,0,1],[0,0,0,1])

############################################################################################################
###################################### Create Rod constraints for digit ####################################
############################################################################################################

global_com_knee_l = np.array(p.getLinkState(digit_id, 4)[4:6])
global_com_knee_r = np.array(p.getLinkState(digit_id, 18)[4:6])
global_com_tarsus_l = np.array(p.getLinkState(digit_id, 5)[4:6])
global_com_tarsus_r = np.array(p.getLinkState(digit_id, 19)[4:6])
global_com_toe_l = np.array(p.getLinkState(digit_id, 6)[4:6])
global_com_toe_r = np.array(p.getLinkState(digit_id, 20)[4:6])

rknee_offset = np.array([-0.02, 0.1, 0.0])
lknee_offset = np.array([-0.02, -0.1, 0.0])
rtarsus_offset2 = np.array([-0.1,0.01,0])
ltarsus_offset2 = np.array([-0.1,-0.01,0])
rtoe_offset = np.array([-0.049,0.01,0.0])
rtarsus_offset = np.array([0.11,0.085,0])
ltoe_offset = np.array([-0.049,-0.01,0.0])
ltarsus_offset = np.array([0.11,-0.085,0])

local_com_knee_l = np.array(p.getLinkState(digit_id, 4)[2:4])
local_com_knee_r = np.array(p.getLinkState(digit_id, 18)[2:4])
local_com_tarsus_l = np.array(p.getLinkState(digit_id, 5)[2:4])
local_com_tarsus_r = np.array(p.getLinkState(digit_id, 19)[2:4])
local_com_toe_l = np.array(p.getLinkState(digit_id, 6)[2:4])
local_com_toe_r = np.array(p.getLinkState(digit_id, 20)[2:4])

com2offset_knee_l = p.multiplyTransforms(*p.invertTransform(*local_com_knee_l),lknee_offset,[0,0,0,1])
com2offset_tarsus2_l = p.multiplyTransforms(*p.invertTransform(*local_com_tarsus_l),ltarsus_offset2,[0,0,0,1])
com2offset_knee_r = p.multiplyTransforms(*p.invertTransform(*local_com_knee_r),rknee_offset,[0,0,0,1])
com2offset_tarsus2_r = p.multiplyTransforms(*p.invertTransform(*local_com_tarsus_r),rtarsus_offset2,[0,0,0,1])

com2offset_toe_l = p.multiplyTransforms(*p.invertTransform(*local_com_toe_l),ltoe_offset,[0,0,0,1])
com2offset_tarsus_l = p.multiplyTransforms(*p.invertTransform(*local_com_tarsus_l),ltarsus_offset,[0,0,0,1])
com2offset_toe_r = p.multiplyTransforms(*p.invertTransform(*local_com_toe_r),rtoe_offset,[0,0,0,1])
com2offset_tarsus_r = p.multiplyTransforms(*p.invertTransform(*local_com_tarsus_r),rtarsus_offset,[0,0,0,1])

p.addUserDebugLine(p.multiplyTransforms(global_com_knee_r[0], global_com_knee_r[1],rknee_offset,[0,0,0,1])[0] ,
                p.multiplyTransforms(global_com_tarsus_r[0], global_com_tarsus_r[1],rtarsus_offset2,[0,0,0,1])[0], [1,0,0])

p.addUserDebugLine(p.multiplyTransforms(global_com_toe_r[0], global_com_toe_r[1],rtoe_offset,[0,0,0,1])[0] ,
                p.multiplyTransforms(global_com_tarsus_r[0], global_com_tarsus_r[1],rtarsus_offset,[0,0,0,1])[0], [1,0,0])

c1 = p.createConstraint(digit_id, 4, digit_id, 5, p.JOINT_POINT2POINT, [0,0,0], com2offset_knee_l,com2offset_tarsus2_l)
c2 = p.createConstraint(digit_id, 5, digit_id, 6, p.JOINT_POINT2POINT, [0,0,0], com2offset_tarsus_l, com2offset_toe_l)
c3 = p.createConstraint(digit_id, 18, digit_id, 19, p.JOINT_POINT2POINT, [0,0,0], com2offset_knee_r,com2offset_tarsus2_r)
c4 = p.createConstraint(digit_id, 19, digit_id, 20, p.JOINT_POINT2POINT, [0,0,0], com2offset_tarsus_r, com2offset_toe_r)
p.changeConstraint(c1, maxForce=1000)
p.changeConstraint(c2, maxForce=1000)
p.changeConstraint(c3, maxForce=1000)
p.changeConstraint(c4, maxForce=1000)

retargetted_poses = []
body_state = []
error = []
# humanoid_dduckie = DebugDuckie([1,0,1], humanoid_id, humanoid_eff[2])
digit_dduckie = DebugDuckie([1,-1,1], digit_id, digit_eff[3])
digit_left_foot_base_rot = [-0.331,-0.367,-0.654, -0.573]
digit_right_foot_base_rot = [0.331,-0.367,0.654, -0.573]
digit_left_foot_base_rot_inv = p.invertTransform([0,0,0],digit_left_foot_base_rot)[1]
digit_right_foot_base_rot_inv = p.invertTransform([0,0,0],digit_right_foot_base_rot)[1]
digit_dduckie.update(p.multiplyTransforms([0,0,0], p.getLinkState(digit_id, digit_eff[2])[5],[0,0,0], digit_right_foot_base_rot_inv)[1])
print("running command", instruction)
p.addUserDebugText(instruction, [0,-1,2], [1,0,0])

for idx in range(llm_xyz.shape[0]):

    # p.setJointMotorControlArray(humanoid_id, humanoid_joints, p.POSITION_CONTROL, targetPositions=dof_pos[idx].cpu().numpy())    
    p.stepSimulation()
    # humanoid_dduckie.update()
    # digit_dduckie.update(p.multiplyTransforms([0,0,0], p.getLinkState(digit_id, digit_eff[3])[5],*p.invertTransform([0,0,0],digit_left_foot_base_rot))[1])
    digit_dduckie.update(p.multiplyTransforms([0,0,0], p.getLinkState(digit_id, digit_eff[2])[5],[0,0,0], digit_right_foot_base_rot_inv)[1])
    
    # digit_dduckie.update(p.multiplyTransforms(*p.invertTransform([0,0,0],digit_left_foot_base_rot),[0,0,0], p.getLinkState(digit_id, digit_eff[3])[5])[1])
    
    # digit_dduckie.update(p.multiplyTransforms([0,0,0],p.invertTransform([0,0,0],[0.183,-0.613,-0.206,-0.740])[1],[0,0,0], p.getLinkState(digit_id, digit_eff[2])[5])[1])
    pos_effs_humanoid = [llm_xyz[idx,i,:] for i in humanoid_eff]

    if idx == 0:
        for i in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
        # markers_humanoid = create_markers_from_id(humanoid_id, body_idxs=humanoid_eff, radius=0.05, colors=[[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]])
        markers_digit = create_markers_from_id(digit_id, body_idxs=digit_eff, radius=0.05, colors=[[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]])
        markers_humanoid = create_markers_from_eff_pos(pos_effs_humanoid, radius=0.05, colors=[[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]])
    ################### update marker position from end-effector position and orientation ###################
    for i in range(len(markers_humanoid)):
        # offset = np.array([0,0,0.0])
        offset = np.array([0,-1,0.0])
        pos_eff_humanoid = pos_effs_humanoid[i]
        # pos_eff_humanoid,rot_eff_humanoid = p.getLinkState(humanoid_id, humanoid_eff[i])[4:6]
        rot_eff_humanoid = [0,0,0,1]
        pos_eff_digit,rot_eff_digit = p.getLinkState(digit_id, digit_eff[i])[4:6]
        
        foot_factor = np.array([1,1,1])
        # if i >1:
        #     foot_factor[-1]=1
        # if i == 3:
        #    rot_eff_humanoid = p.multiplyTransforms([0,0,0], rot_eff_humanoid, [0,0,0], digit_left_foot_base_rot)[1]
        # elif i == 2:
        #     rot_eff_humanoid = p.multiplyTransforms([0,0,0], rot_eff_humanoid, [0,0,0], digit_right_foot_base_rot)[1]
        # else:
        rot_eff_humanoid = rot_eff_digit
        
        p.resetBasePositionAndOrientation(markers_humanoid[i], np.array([pos_eff_humanoid[0],pos_eff_humanoid[2],pos_eff_humanoid[1] ])+offset, rot_eff_humanoid)
        p.resetBasePositionAndOrientation(markers_digit[i], pos_eff_digit, rot_eff_digit)
        left_foot_rot_error = np.linalg.norm(get_quat_error(p.getBasePositionAndOrientation(markers_humanoid[3])[1], p.getLinkState(digit_id, digit_eff[3])[5]))
        right_foot_rot_error = np.linalg.norm(get_quat_error(p.getBasePositionAndOrientation(markers_humanoid[2])[1], p.getLinkState(digit_id, digit_eff[2])[5]))
    # while get_error(markers_humanoid, digit_eff, digit_id).sum()>0.0001 or left_foot_rot_error>0.1:# or right_foot_rot_error>0.1:
    # for k in range(10):
    #     digit_dofs = retarget_humanoid_to_digit_ik([p.getBasePositionAndOrientation(marker)[0] for marker in markers_humanoid], 
    #                                                 [p.getBasePositionAndOrientation(marker)[1] for marker in markers_humanoid],
    #                                             digit_eff,
    #                                             digit_id,
    #                                             p.getJointStates(digit_id,[i for i in range(p.getNumJoints(digit_id)) if p.getJointInfo(digit_id, i)[3]>-1])[0])

    #     for i in range(p.getNumJoints(digit_id)):
    #         jointInfo = p.getJointInfo(digit_id, i)
    #         qIndex = jointInfo[3]
    #         if qIndex > -1:
    #                 p.resetJointState(digit_id,i,digit_dofs[qIndex-7])
        left_foot_rot_error = np.linalg.norm(get_quat_error(p.getBasePositionAndOrientation(markers_humanoid[3])[1], p.getLinkState(digit_id, digit_eff[3])[5]))
        right_foot_rot_error = np.linalg.norm(get_quat_error(p.getBasePositionAndOrientation(markers_humanoid[2])[1], p.getLinkState(digit_id, digit_eff[2])[5]))
    # retargetted_poses.append(np.concatenate((root_pos[idx].cpu().numpy(),root_rot[idx].cpu().numpy(), digit_dofs)))
    joint_info = {p.getJointInfo(digit_id, digit_body_idx)[12].decode():p.getLinkState(digit_id, digit_body_idx)[5] for digit_body_idx in range(30)}
    llm_pos = llm_xyz[idx,0,:]
    # body_state.append((joint_info,np.array([llm_pos[0],llm_pos[2],llm_pos[1]]),[0,0,0,1],digit_dofs))
    q = get_quaterion_from_torso_points(torso_points[idx])
    print(q)
    p.resetBasePositionAndOrientation(digit_id, [llm_pos[0],llm_pos[2],llm_pos[1]]+np.array([0,-1,height_offset]), q)
    # error.append(get_error(markers_digit, digit_eff, digit_id))
    # quaternion_error = get_quat_error(p.getBasePositionAndOrientation(markers_humanoid[2])[1], p.getLinkState(digit_id, digit_eff[2])[5])
    # print("q error:", quaternion_error)
    time.sleep(0.1)

np.save("digit_motion/"+instruction+".npy", body_state)
# print("saved digit state at", "digit_state_"+motion_file.split("_")[-1])
# error = np.array(error)**2
# # np.save("error_digit_v2.npy", error)
# fig, axes = plt.subplots(nrows=2, ncols=2)
# axes[0,0].plot(error[:,0])
# axes[0,1].plot(error[:,1])
# axes[1,0].plot(error[:,2])
# axes[1,1].plot(error[:,3])
# axes[0,0].set_title("squared error right hand")
# axes[0,1].set_title("squared error left hand")
# axes[1,0].set_title("squared error right foot")
# axes[1,1].set_title("squared error left foot")
# plt.show()
