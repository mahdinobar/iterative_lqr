#!/usr/bin/env python
# coding: utf-8
# #### iLQR for kinematic example
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from ocp import *
from costs import *
from ocp_utils import *
import pybullet as p
import pybullet_data

np.set_printoptions(precision=4, suppress=True)
# #### Setup pybullet with the urdf
# configure pybullet and load plane.urdf and quadcopter.urdf
physicsClient = p.connect(p.DIRECT)  # pybullet only for computations no visualisation, faster
# physicsClient = p.connect(p.GUI)  # pybullet with visualisation
# physicsClient = p.connect(p.GUI, options="--width=1920 --height=1080 --mp4=\"/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/test.mp4\" --mp4fps=10")  # pybullet with visualisation and recording
# p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=30, cameraPitch=-90, cameraTargetPosition=[0,0.,0])
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.resetSimulation()
# p.loadURDF('plane.urdf')
robot_urdf = "../data/urdf/frankaemika_new/panda_arm.urdf"

# parameters ################################################################################
# Construct the robot system
demo_name='warm_start_3'
warm_start=False
if warm_start is True:
    warm_start_demo_name='warm_start_3'
n_iter = 40
T = 50 # number of data points
dt = 0.5
dof = 7

robot1_base_pose=[0, 0, 0]
robot2_base_pose=[0.6198, -0.7636, 0]

ViaPnts1=np.array([[+0.6526506, -0.19499213, +0.20],
                   [+0.6526506, -0.19499213, +0.05541289],
                   [+0.6526506, -0.19499213, +0.20]])
ViaPnts2=np.array([[+0.51262467, -0.01968636, +0.20],
                   [+0.51262467, -0.01968636, +0.05541289],
                   [+0.51262467, -0.01968636, +0.20]])
# todo check make code robust
# specify at which time step to pass viapoints
nbViaPnts=np.shape(ViaPnts1)[0]
idx=np.array([20, 25, 30],dtype=int)

p_target_1 = np.array([+0.25818314,
                       +0.2210157,
                       +0.20])
p_target_2 = np.array([+0.80018013,
                       -0.50010305,
                       +0.20])

# idx= np.linspace(1,1.*T,nbViaPnts+2, dtype='int')[1:-1]

# Set precisions
Q_q1 = 1e-3
Q_q2 = 1e-3

QT_s1 = 1e0
QT_s2 = 1e0

W = np.zeros((6, 6))
WT_p1 = 1e4
WT_p2 = 1e4

Wvia_p1 = 1e4
Wvia_p2 = 1e4

R_dq1 = 1e0
R_dq2 = 1e0
R_dq2_j2 = 1e0

R_ds1 = 1e0
R_ds2 = 1e0

S_dq1 = 1e-1
S_dq2 = 1e-1
S_dq2_j2 = 1e-1

S_ds1 = 1e-1
S_ds2 = 1e-1

qobs = 0
obs_thresh = 2
model_Q_obs_s = 2

s1_ref = 10
s2_ref = 10
# ############################################################################################

# make the target inside positive xy plane
theta=0#-np.pi/2 #this needs to be matched with jacobian modification
r = R.from_matrix([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
baseOrientation=r.as_quat()
robot1_id = p.loadURDF(robot_urdf, basePosition=robot1_base_pose, useFixedBase=1)
robot2_id = p.loadURDF(robot_urdf, basePosition=robot2_base_pose, baseOrientation=baseOrientation, useFixedBase=1)
joint_limits = get_joint_limits(robot1_id, 7)

# Define the end-effector
link_id = 10
link_name = 'panda_grasptarget_hand'
# Create a ball to show the target
_, _, ballId1 = create_primitives(radius=0.03, rgbaColor=[1, 0, 0, 1])
_, _, ballId2 = create_primitives(radius=0.03, rgbaColor=[0, 0, 1, 1])

_, _, cylinderId1 = create_primitives(shapeType=p.GEOM_CYLINDER, length=0.023, radius=0.01, rgbaColor=[1, 0, 0, 1])
_, _, cylinderId2 = create_primitives(shapeType=p.GEOM_CYLINDER, length=0.023, radius=0.01, rgbaColor=[0, 0, 1, 1])

_, _, boxId1 = create_primitives(shapeType=p.GEOM_BOX, length=0.020, radius=0.01, rgbaColor=[1, 0, 0, 1], halfExtents = [0.1, 0.1, 0.05])
_, _, boxId2 = create_primitives(shapeType=p.GEOM_BOX, length=0.020, radius=0.01, rgbaColor=[0, 0, 1, 1], halfExtents = [0.1, 0.1, 0.05])

# Finding the joint (and link) index
for i in range(p.getNumJoints(robot1_id)):
    print(i, p.getJointInfo(robot1_id, i)[1])
    print(i, p.getJointInfo(robot2_id, i)[1])

sys = URDFRobot_spacetime_dual(dof=dof, robot1_id=robot1_id, robot2_id=robot2_id, dt=dt)

if warm_start is False:
    # Set the initial state
    # comment for warm start
    # q0_1 = np.array([0., 0., 0., 0., 0., 0., 0.])
    # q0_2 = np.array([0., 0., 0., 0., 0., 0., 0.])
    q0_1=np.mean(joint_limits,0)
    q0_2=np.mean(joint_limits,0)
    # fix first joint of robot2
    q0_1[0]=0.5
    q0_2[0]=1
    x0 = np.concatenate([q0_1, q0_2, np.zeros(2)])
    sys.set_init_state(x0)
    ### Set initial control output
    # set initial control output to be all zeros
    # add epsilon offset to avoid barrier
    # # comment for warm start
    us = np.hstack((np.zeros((T + 1, sys.Du-2)),1e-3*np.ones((T + 1, 2))))
    _ = sys.compute_matrices(x=None, u=us[0])
    xs = sys.rollout(us[:-1])

else:
    # uncomment to warm start traj
    us=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/us0_tailor.npy".format(warm_start_demo_name))
    xs=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/xs0_tailor.npy".format(warm_start_demo_name))
    x0=xs[0,:]
    sys.set_init_state(x0)

# #### Try forward kinematics
pos1_0, quat1_0, pos2_0, quat2_0 = sys.compute_ee(x0, link_id)
# # Put the ball at the end-effector
p.resetBasePositionAndOrientation(ballId1, pos1_0, quat1_0)
p.resetBasePositionAndOrientation(ballId2, pos2_0, quat2_0)

p.resetBasePositionAndOrientation(cylinderId1, ViaPnts1[1], [0,0,0,1])
p.resetBasePositionAndOrientation(cylinderId2, ViaPnts2[1], [0,0,0,1])

p.resetBasePositionAndOrientation(boxId1, p_target_1, [0,0,0,1])
p.resetBasePositionAndOrientation(boxId2, p_target_2, [0,0,0,1])

# # #### Plot initial trajectory
# # interpolate the virtual time for visualization of both
# xs_interp=np.zeros(xs.shape)
# tt=np.linspace(0,np.max(xs[:,14:]), T)
# for i in range(dof):
#     xs_interp[:,i] = np.interp(T, xs[:,14],xs[:,i])
#     xs_interp[:,dof+i] = np.interp(T, xs[:, 15], xs[:, dof+i])
# sys.vis_traj(xs_interp)

Q = np.diag(np.concatenate((Q_q1*np.ones(7),Q_q2*np.ones(7),[0, 0])))
Qf = np.diag(np.concatenate((np.zeros(14),[QT_s1, QT_s2])))
WT = np.diag(np.concatenate((WT_p1*np.ones(3),WT_p2*np.ones(3))))
Wvia = np.diag(np.concatenate((WT_p1*np.ones(3),WT_p2*np.ones(3))))
R = np.diag(np.concatenate((R_dq1*np.array([1,1,1,1,1,1,1]),R_dq2**np.array([1]),R_dq2_j2**np.array([1]),R_dq2**np.array([1,1,1,1,1]),[R_ds1,R_ds2])))
S = np.diag(np.concatenate((S_dq1*np.array([1,1,1,1,1,1,1]),S_dq2**np.array([1]),S_dq2_j2**np.array([1]),S_dq2**np.array([1,1,1,1,1]),[S_ds1,S_ds2])))
x_ref= np.concatenate((np.mean(joint_limits,0),np.mean(joint_limits,0),[s1_ref,s2_ref]))
ds1_ref=0
ds2_ref=0
u_ref = np.concatenate((np.zeros(sys.Dx-2),[ds1_ref,ds2_ref]))

# todo for batch?
mu = 1e-6  # regularization coefficient
# #### Set end effector target
# W and WT: cost coefficients for the end-effector reaching task

# ### iLQR using cost model
# #### Define the cost
# The costs consist of: a) state regularization (Q), b) control regularization (R), and c) End-effector reaching task (W)
# Running cost is for the time 0 <= t < T, while terminal cost is for the time t = T

id=0
costs = []

for i in range(T):
    # BarrierCost = CostModelBarrier(sys, K=K, x_ref=x_ref)
    # todo check make code robust
    if any(i == c for c in idx) and nbViaPnts>0:
        runningEECost = CostModelQuadraticTranslation_dual(sys, W=Wvia, ee_id=link_id, p_target_1=ViaPnts1[id],
                                                           p_target_2=ViaPnts2[id])
        id += 1
    else:
        runningEECost = CostModelQuadraticTranslation_dual(sys, W=W, ee_id=link_id, p_target_1=p_target_1,
                                                           p_target_2=p_target_2)
    runningStateCost = CostModelQuadratic(sys, Q=Q, x_ref=x_ref)
    runningControlCost = CostModelQuadratic(sys, R=R, u_ref=u_ref)
    smoothCost = CostModelQuadratic(sys, S=S)
    # obstAvoidCost = CostModelObstacle_exp4(sys, ee_id=link_id, qobs=qobs, Qobs=Qobs, th=obs_thresh)
    obstAvoidCost = CostModelObstacle_ellipsoids_exp4(sys, ee_id=link_id, qobs=qobs, th=obs_thresh, model_Q_obs_s=model_Q_obs_s)
    runningCost = CostModelSum(sys, [runningStateCost, runningControlCost, smoothCost, runningEECost, obstAvoidCost])
    costs += [runningCost]
terminalStateCost = CostModelQuadratic(sys, Q=Qf, x_ref=x_ref)
terminalControlCost = CostModelQuadratic(sys, R=R)
terminalSmoothCost = CostModelQuadratic(sys, S=S)
terminalEECost = CostModelQuadraticTranslation_dual(sys, W=WT, ee_id=link_id, p_target_1=p_target_1, p_target_2=p_target_2)
obstAvoidCost = CostModelObstacle_ellipsoids_exp4(sys, ee_id=link_id, qobs=qobs, th=obs_thresh, model_Q_obs_s=model_Q_obs_s)
terminalCost = CostModelSum(sys, [terminalStateCost, terminalControlCost, terminalEECost, obstAvoidCost])
costs += [terminalCost]

# #### Construct ILQR
ilqr_cost = ILQR(sys, mu)
ilqr_cost.set_init_state(x0)
ilqr_cost.set_timestep(T)
ilqr_cost.set_cost(costs)
ilqr_cost.set_state(xs, us)  # set initial trajectory

# #### Solve and Plot
ilqr_cost.solve(n_iter, method='batch', threshold_alpha=1e-5)
xs_batch, us_batch = ilqr_cost.xs, ilqr_cost.us
clear_output()

# #### Play traj
# interpolate the virtual time for visualization of both
nbVis=50
xs_interp=np.zeros((nbVis, ilqr_cost.xs.shape[1]))
tt=np.linspace(0,np.max(ilqr_cost.xs[:,14:]), nbVis)
for i in range(dof):
    xs_interp[:,i] = np.interp(tt, ilqr_cost.xs[:,14],ilqr_cost.xs[:,i])
    xs_interp[:,dof+i] = np.interp(tt, ilqr_cost.xs[:, 15], ilqr_cost.xs[:, dof+i])
    xs_interp[:, -1]=tt
    xs_interp[:, -2] = tt

np.save("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/xs_interp.npy".format(demo_name),xs_interp)
np.save("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/xs.npy".format(demo_name),ilqr_cost.xs)
np.save("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/us.npy".format(demo_name),ilqr_cost.us)


# xs=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/xs.npy".format(demo_name))
# us=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/us.npy".format(demo_name))
# #### Play traj
# interpolate the virtual time for visualization of both
nbVis=3000
xs_interp=np.zeros((nbVis, xs.shape[1]))
tt=np.linspace(0,np.max(xs[:,14:]), nbVis)
for i in range(dof):
    xs_interp[:,i] = np.interp(tt, xs[:,14],xs[:,i])
    xs_interp[:,dof+i] = np.interp(tt, xs[:, 15], xs[:, dof+i])
    xs_interp[:, -1]=tt
    xs_interp[:, -2] = tt
np.save("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/xs_interp_more_steps.npy".format(demo_name),xs_interp)

xs_interp=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/xs_interp.npy".format(demo_name))
# # unocmment to record only final traj
p.disconnect()
physicsClient = p.connect(p.GUI, options="--width=1920 --height=1080 --mp4=\"/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/test.mp4\" --mp4fps=10".format(demo_name))  # pybullet with visualisation and recording
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=120, cameraPitch=-30, cameraTargetPosition=(ViaPnts1[1]+ViaPnts2[1])/2)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
robot_urdf = "../data/urdf/frankaemika_new/panda_arm.urdf"
robot1_id = p.loadURDF(robot_urdf, basePosition=robot1_base_pose, useFixedBase=1)
robot2_id = p.loadURDF(robot_urdf, basePosition=robot2_base_pose, baseOrientation=baseOrientation, useFixedBase=1)
p.loadURDF('plane.urdf')
# Create a ball to show the target
# _, _, ballId1 = create_primitives(radius=0.05, rgbaColor=[1, 0, 0, 1])
# _, _, ballId2 = create_primitives(radius=0.05, rgbaColor=[0, 0, 1, 1])
# p.resetBasePositionAndOrientation(ballId1, p_target_1, (0, 0, 0, 1))
# p.resetBasePositionAndOrientation(ballId2, p_target_2, (0, 0, 0, 1))

_, _, cylinderId1 = create_primitives(shapeType=p.GEOM_CYLINDER, length=0.023, radius=0.01, rgbaColor=[1, 0, 0, 1])
_, _, cylinderId2 = create_primitives(shapeType=p.GEOM_CYLINDER, length=0.023, radius=0.01, rgbaColor=[0, 0, 1, 1])

box_length=0.10
_, _, boxId1 = create_primitives(shapeType=p.GEOM_BOX, length=0.020, radius=0.01, rgbaColor=[1, 0, 0, 1], halfExtents = [box_length, box_length, box_length/4])
_, _, boxId2 = create_primitives(shapeType=p.GEOM_BOX, length=0.020, radius=0.01, rgbaColor=[0, 0, 1, 1], halfExtents = [box_length, box_length, box_length/4])

p.resetBasePositionAndOrientation(cylinderId1, ViaPnts1[1], [0,0,0,1])
p.resetBasePositionAndOrientation(cylinderId2, ViaPnts2[1], [0,0,0,1])

box_tol = +0.5*np.array([box_length,
                       box_length,
                       0])
p.resetBasePositionAndOrientation(boxId1, np.array([p_target_1[0],p_target_1[1],0])+box_tol, [0,0,0,1])
p.resetBasePositionAndOrientation(boxId2, np.array([p_target_2[0],p_target_2[1],0])+box_tol, [0,0,0,1])

# _, _, ballId1_middle = create_primitives(radius=0.05, rgbaColor=[1, 0, 0, 1])
# p.resetBasePositionAndOrientation(ballId1_middle, ViaPnts1[0], (0, 0, 0, 1))

sys.vis_traj(xs_interp, vis_dt=0.1)

# # #### Compute Error
pos1, _, pos2, _ = sys.compute_ee(ilqr_cost.xs[-1], link_id)

print('pos1-p_target_1={}, pos2-p_target_2={}'.format(pos1-p_target_1, pos2-p_target_2))

# test joint limits
joint_limit_threshold=5/100*(np.max(joint_limits,axis=0)-np.min(joint_limits,axis=0))
if np.any(np.max(xs[:,:7],axis=0)>joint_limits[1,:]-joint_limit_threshold) or np.any(np.min(xs[:,:7],axis=0)<joint_limits[0,:]+joint_limit_threshold) or np.any(np.max(xs[:,7:14],axis=0)>joint_limits[1,:]-joint_limit_threshold) or np.any(np.min(xs[:,7:14],axis=0)<joint_limits[0,:]+joint_limit_threshold):
    print('---ERROR---joint limits are NOT satisfied!')
else:
    print('joint limits are satisfied!')

if warm_start is False:
    # # uncomment to save warm start traj
    np.save("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/xs0_tailor.npy".format(demo_name),ilqr_cost.xs)
    np.save("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/NIST_demos/{}/us0_tailor.npy".format(demo_name),ilqr_cost.us)
