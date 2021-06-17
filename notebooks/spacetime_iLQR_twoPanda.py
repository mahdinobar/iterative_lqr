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

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
np.set_printoptions(precision=4, suppress=True)

# #### Setup pybullet with the urdf
# configure pybullet and load plane.urdf and quadcopter.urdf
physicsClient = p.connect(p.DIRECT)  # pybullet only for computations no visualisation, faster
# physicsClient = p.connect(p.GUI)  # pybullet with visualisation
# physicsClient = p.connect(p.GUI, options="--width=1920 --height=1080 --mp4=\"/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/test.mp4\" --mp4fps=10")  # pybullet with visualisation and recording
# p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=30, cameraPitch=-30, cameraTargetPosition=[0,0.5,0])
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.loadURDF('plane.urdf')

robot_urdf = "../data/urdf/frankaemika_new/panda_arm.urdf"
robot1_base_pose=[0, 0, 0]
robot2_base_pose=[0, 0.7, 0]
robot1_id = p.loadURDF(robot_urdf, basePosition=robot1_base_pose, useFixedBase=1)
robot2_id = p.loadURDF(robot_urdf, basePosition=robot2_base_pose, useFixedBase=1)
p_target_1 = np.array([.6, .1, .5])
p_target_2 = np.array([.6, .2, .5])
ViaPnts1=np.array([[.3, .5, .5]])
ViaPnts2=np.array([])
joint_limits = get_joint_limits(robot1_id, 7)

# Define the end-effector
link_id = 10
link_name = 'panda_grasptarget_hand'

# Create a ball to show the target
_, _, ballId1 = create_primitives(radius=0.05, rgbaColor=[1, 0, 0, 1])
_, _, ballId2 = create_primitives(radius=0.05, rgbaColor=[0, 0, 1, 1])

# Finding the joint (and link) index
for i in range(p.getNumJoints(robot1_id)):
    print(i, p.getJointInfo(robot1_id, i)[1])
    print(i, p.getJointInfo(robot2_id, i)[1])
# getLinkState

# Construct the robot system
n_iter = 20
T = 20 # number of data points
dt = 0.5
dof = 7
sys = URDFRobot_spacetime_dual(dof=dof, robot1_id=robot1_id, robot2_id=robot2_id, dt=dt)

# Set the initial state
# comment for warm start
q0_1 = np.array([0., 0., 0., 0., 0., 0., 0.])
q0_2 = np.array([0., 0., 0., 0., 0., 0., 0.])
x0 = np.concatenate([q0_1, q0_1, np.zeros(2)])

# # uncomment to warm start traj
# us=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/us0.npy")
# xs=np.load("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/xs0.npy")
# x0=xs[0,:]

sys.set_init_state(x0)

### Set initial control output
# set initial control output to be all zeros
# add epsilon offset to avoid barrier
# # comment for warm start
us = np.hstack((np.zeros((T + 1, sys.Du-2)),1e-3*np.ones((T + 1, 2))))
_ = sys.compute_matrices(x=None, u=us[0])
xs = sys.rollout(us[:-1])

# #### Try forward kinematics
pos1_0, quat1_0, pos2_0, quat2_0 = sys.compute_ee(x0, link_id)
# Put the ball at the end-effector
p.resetBasePositionAndOrientation(ballId1, pos1_0, quat1_0)
p.resetBasePositionAndOrientation(ballId2, pos2_0, quat2_0)

# #### Plot initial trajectory
# interpolate the virtual time for visualization of both
xs_interp=np.zeros(xs.shape)
tt=np.linspace(0,np.max(xs[:,14:]), T)
for i in range(dof):
    xs_interp[:,i] = np.interp(T, xs[:,14],xs[:,i])
    xs_interp[:,dof+i] = np.interp(T, xs[:, 15], xs[:, dof+i])
sys.vis_traj(xs_interp)
# #### Set the regularization cost coefficients Q and R
Q_q1=1e-3
Q_q2=1e-3
Q = np.diag(np.concatenate((Q_q1*np.ones(7),Q_q2*np.ones(7),[0, 0])))
QT_s1=1e0
QT_s2=1e0
Qf = np.diag(np.concatenate((np.zeros(14),[QT_s1, QT_s2])))

W = np.zeros((6,6))
WT_p1=1e4
WT_p2=1e4
WT = np.diag(np.concatenate((WT_p1*np.ones(3),WT_p2*np.ones(3))))

Wvia_p1=1e4
Wvia_p2=0
Wvia = np.diag(np.concatenate((WT_p1*np.ones(3),WT_p2*np.ones(3))))

Rfactor_dq1=1e-1
Rfactor_dq2=1e-1
Rfactor_dq2_j6=1e-1

Rfactor_ds1=1e0
Rfactor_ds2=1e0
R = np.diag(np.concatenate((Rfactor_dq1*np.array([1,1,1,1,1,1,1]),Rfactor_dq2**np.array([1,1,1,1,1]),Rfactor_dq2_j6**np.array([1]),Rfactor_dq2**np.array([1]),[Rfactor_ds1,Rfactor_ds2])))

qobs=0e3
obs_thresh=2.
model_Q_obs_s=1e1 # 100 is at the order corrosponding hyper-ellipsoid size 0.1 m
# model_Q_obs_x=1e0
# Qobs=np.diag(np.concatenate((model_Q_obs_x*np.ones(3),[model_Q_obs_s])))

s1_ref=10
s2_ref=10
x_ref= np.concatenate((np.zeros(sys.Dx-2),[s1_ref,s2_ref]))
ds1_ref=s1_ref/T
ds2_ref=s2_ref/T
u_ref= np.concatenate((np.zeros(sys.Dx-2),[ds1_ref,ds2_ref]))

# todo for batch?
mu = 1e-6  # regularization coefficient
# #### Set end effector target
# W and WT: cost coefficients for the end-effector reaching task

# p.resetBasePositionAndOrientation(ballId1, p_target_1, (0, 0, 0, 1))
# p.resetBasePositionAndOrientation(ballId2, p_target_2, (0, 0, 0, 1))

# ### iLQR using cost model
# #### Define the cost
# The costs consist of: a) state regularization (Q), b) control regularization (R), and c) End-effector reaching task (W)
# Running cost is for the time 0 <= t < T, while terminal cost is for the time t = T

# todo check make code robust
nbViaPnts=np.shape(ViaPnts1)[0]
idx= np.linspace(1,T,nbViaPnts+2, dtype='int')[1:-1]
id=0
costs = []
for i in range(T):
    # BarrierCost = CostModelBarrier(sys, K=K, x_ref=x_ref)
    # todo check make code robust
    if any(i == c for c in idx) and nbViaPnts>0:
        runningEECost = CostModelQuadraticTranslation_dual(sys, W=Wvia, ee_id=link_id, p_target_1=ViaPnts1[id],
                                                           p_target_2=p_target_2)
        id += 1
    else:
        runningEECost = CostModelQuadraticTranslation_dual(sys, W=W, ee_id=link_id, p_target_1=p_target_1,
                                                           p_target_2=p_target_2)
    runningStateCost = CostModelQuadratic(sys, Q=Q, x_ref=x_ref)
    runningControlCost = CostModelQuadratic(sys, R=R, u_ref=u_ref)
    # obstAvoidCost = CostModelObstacle_exp4(sys, ee_id=link_id, qobs=qobs, Qobs=Qobs, th=obs_thresh)
    obstAvoidCost = CostModelObstacle_ellipsoids_exp4(sys, ee_id=link_id, qobs=qobs, th=obs_thresh, model_Q_obs_s=model_Q_obs_s)
    runningCost = CostModelSum(sys, [runningStateCost, runningControlCost, runningEECost, obstAvoidCost])
    costs += [runningCost]
terminalStateCost = CostModelQuadratic(sys, Q=Qf, x_ref=x_ref)
terminalControlCost = CostModelQuadratic(sys, R=R)
terminalEECost = CostModelQuadraticTranslation_dual(sys, W=WT, ee_id=link_id, p_target_1=p_target_1, p_target_2=p_target_2)
# obstAvoidCost = CostModelObstacle_exp4(sys, ee_id=link_id, qobs=qobs, Qobs=Qobs, th=obs_thresh)
obstAvoidCost = CostModelObstacle_ellipsoids_exp4(sys, ee_id=link_id, qobs=qobs, th=obs_thresh, model_Q_obs_s=model_Q_obs_s)
terminalCost = CostModelSum(sys, [terminalStateCost, terminalControlCost, terminalEECost, obstAvoidCost])
costs += [terminalCost]

# #### Construct ILQR
ilqr_cost = ILQR(sys, mu)
ilqr_cost.set_init_state(x0)
ilqr_cost.set_timestep(T)
ilqr_cost.set_cost(costs)
ilqr_cost.set_state(xs, us)  # set initial trajectory

# todo for batch?
ilqr_cost.mu = 1e-5

# #### Solve and Plot
ilqr_cost.solve(n_iter, method='batch', threshold_alpha=1e-5)
xs_batch, us_batch = ilqr_cost.xs, ilqr_cost.us
# clear_output()

# #### Play traj
# interpolate the virtual time for visualization of both
nbVis=50
xs_interp=np.zeros((nbVis, ilqr_cost.xs.shape[1]))
tt=np.linspace(0,np.max(ilqr_cost.xs[:,14:]), nbVis)
for i in range(dof):
    xs_interp[:,i] = np.interp(tt, ilqr_cost.xs[:,14],ilqr_cost.xs[:,i])
    xs_interp[:,dof+i] = np.interp(tt, ilqr_cost.xs[:, 15], ilqr_cost.xs[:, dof+i])


# # unocmment to record only final traj
p.disconnect()
physicsClient = p.connect(p.GUI, options="--width=1920 --height=1080 --mp4=\"/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/test.mp4\" --mp4fps=10")  # pybullet with visualisation and recording
p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=30, cameraPitch=-30, cameraTargetPosition=[0,0.5,0])
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.loadURDF('plane.urdf')
robot_urdf = "../data/urdf/frankaemika_new/panda_arm.urdf"
robot1_id = p.loadURDF(robot_urdf, basePosition=robot1_base_pose, useFixedBase=1)
robot2_id = p.loadURDF(robot_urdf, basePosition=robot2_base_pose, useFixedBase=1)
# Create a ball to show the target
_, _, ballId1 = create_primitives(radius=0.05, rgbaColor=[1, 0, 0, 1])
_, _, ballId2 = create_primitives(radius=0.05, rgbaColor=[0, 0, 1, 1])
p.resetBasePositionAndOrientation(ballId1, p_target_1, (0, 0, 0, 1))
p.resetBasePositionAndOrientation(ballId2, p_target_2, (0, 0, 0, 1))

_, _, ballId1_middle = create_primitives(radius=0.05, rgbaColor=[1, 0, 0, 1])
p.resetBasePositionAndOrientation(ballId1_middle, ViaPnts1[0], (0, 0, 0, 1))

sys.vis_traj(xs_interp, vis_dt=0.1)

# for i in range(21):
#     print(i)
#     sys.compute_ee(ilqr_cost.xs[i,:], link_id)

# # #### Compute Error
pos1, _, pos2, _ = sys.compute_ee(ilqr_cost.xs[-1], link_id)

print('pos1-p_target_1={}, pos2-p_target_2={}'.format(pos1-p_target_1, pos2-p_target_2))

# # uncomment to save warm start traj
np.save("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/xs0_tailor.npy",ilqr_cost.xs)
np.save("/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/us0_tailor.npy",ilqr_cost.us)
