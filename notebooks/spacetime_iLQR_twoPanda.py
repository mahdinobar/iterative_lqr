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
# physicsClient = p.connect(p.DIRECT)  # pybullet only for computations no visualisation, faster
physicsClient = p.connect(p.GUI, options="--width=1920 --height=1080 --mp4=\"/home/mahdi/RLI/codes/iterative_lqr/notebooks/tmp/test.mp4\" --mp4fps=10")  # pybullet with visualisation
p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=30, cameraPitch=-45, cameraTargetPosition=[0,0.6,0])
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.loadURDF('plane.urdf')

robot_urdf = "../data/urdf/frankaemika_new/panda_arm.urdf"
robot1_id = p.loadURDF(robot_urdf, basePosition=[0., 0.0, 0.], useFixedBase=1)
robot2_id = p.loadURDF(robot_urdf, basePosition=[0., 0.7, 0.], useFixedBase=1)
p_target_1 = np.array([.7, 0.7, .5])
p_target_2 = np.array([.7, 0.0, .5])
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

# Construct the robot system
n_iter = 5
dt = 0.5
T = 10 # number of data points
dof = 7
sys = URDFRobot_spacetime_dual(dof=dof, robot1_id=robot1_id, robot2_id=robot2_id, dt=dt)

# Set the initial state
# q0 = np.random.rand(7)
q0_1 = np.array([0., 0., 0., 0., 0., 0., 0.])
q0_2 = np.array([0., 0., 0., 0., 0., 0., 0.])
# q0 = np.array([0.4201, 0.4719, 0.9226, 0.8089, 0.3113, 0.7598, 0.364 ])
x0 = np.concatenate([q0_1, q0_1, np.zeros(2)])
sys.set_init_state(x0)

# #### Try forward kinematics
pos1_0, quat1_0, pos2_0, quat2_0 = sys.compute_ee(x0, link_id)

# Put the ball at the end-effector
p.resetBasePositionAndOrientation(ballId1, pos1_0, quat1_0)
p.resetBasePositionAndOrientation(ballId2, pos2_0, quat2_0)

# #### Set initial control output
# set initial control output to be all zeros
us = np.zeros((T + 1, sys.Du))
_ = sys.compute_matrices(x=None, u=us[0])
xs = sys.rollout(us[:-1])

# #### Plot initial trajectory
sys.vis_traj(xs)
# #### Set the regularization cost coefficients Q and R
Q_q1=1e-3
Q_q2=1e-3
Q = np.diag(np.concatenate((Q_q1*np.ones(7),Q_q2*np.ones(7),[0, 0])))
QT_s1=1e0
QT_s2=1e0
Qf = np.diag(np.concatenate((np.zeros(14),[QT_s1, QT_s2])))

W = np.zeros((6,6))
WT_p1=1e2
WT_p2=1e2
WT = np.diag(np.concatenate((WT_p1*np.ones(3),WT_p2*np.ones(3))))

Rfactor_dq1=1e0
Rfactor_dq2=1e0
Rfactor_ds1=1e0
Rfactor_ds2=1e0
R = np.diag(np.concatenate((Rfactor_dq1*np.ones(7),Rfactor_dq2*np.ones(7),[Rfactor_ds1,Rfactor_ds2])))

qobs=0
obs_thresh=10
model_Q_obs_x=1e0
model_Q_obs_s=1e0
Qobs=np.diag(np.concatenate((model_Q_obs_x*np.ones(3),[model_Q_obs_s])))

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

p.resetBasePositionAndOrientation(ballId1, p_target_1, (0, 0, 0, 1))
p.resetBasePositionAndOrientation(ballId2, p_target_2, (0, 0, 0, 1))

# ### iLQR using cost model
# #### Define the cost
# The costs consist of: a) state regularization (Q), b) control regularization (R), and c) End-effector reaching task (W)
# Running cost is for the time 0 <= t < T, while terminal cost is for the time t = T
costs = []
for i in range(T):
    runningStateCost = CostModelQuadratic(sys, Q=Q, x_ref=x_ref)
    runningControlCost = CostModelQuadratic(sys, R=R, u_ref=u_ref)
    runningEECost = CostModelQuadraticTranslation_dual(sys, W=W, ee_id=link_id, p_target_1=p_target_1, p_target_2=p_target_2)
    obstAvoidCost = CostModelObstacle_exp4(sys, ee_id=link_id, qobs=qobs, Qobs=Qobs, th=obs_thresh)
    runningCost = CostModelSum(sys, [runningStateCost, runningControlCost, runningEECost, obstAvoidCost])
    costs += [runningCost]
terminalStateCost = CostModelQuadratic(sys, Q=Qf)
terminalControlCost = CostModelQuadratic(sys, R=R)
terminalEECost = CostModelQuadraticTranslation_dual(sys, W=WT, ee_id=link_id, p_target_1=p_target_1, p_target_2=p_target_2)
obstAvoidCost = CostModelObstacle_exp4(sys, ee_id=link_id, qobs=qobs, Qobs=Qobs, th=obs_thresh)
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
sys.vis_traj(ilqr_cost.xs, vis_dt=0.1)

# # #### Compute Error
pos1, _, pos2, _ = sys.compute_ee(ilqr_cost.xs[-1], link_id)
print('pos1-p_target_1={}, pos2-p_target_2={}'.format(pos1-p_target_1, pos2-p_target_2))
