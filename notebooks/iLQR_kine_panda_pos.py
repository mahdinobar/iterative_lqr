#!/usr/bin/env python
# coding: utf-8

# #### iLQR for kinematic example

# In[1]:


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

# In[2]:


# configure pybullet and load plane.urdf and quadcopter.urdf
#physicsClient = p.connect(p.DIRECT)  # pybullet only for computations no visualisation, faster
physicsClient = p.connect(p.GUI)  # pybullet with visualisation
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

p.setAdditionalSearchPath(pybullet_data.getDataPath())


# In[3]:


p.resetSimulation()

robot_urdf = "../data/urdf/frankaemika_new/panda_arm.urdf"
robot_id = p.loadURDF(robot_urdf, useFixedBase=1)
joint_limits = get_joint_limits(robot_id, 7)

p.loadURDF('plane.urdf')

#Define the end-effector
link_id = 10
link_name = 'panda_grasptarget_hand'

#Create a ball to show the target
_,_,ballId = create_primitives(radius=0.05)


# #### Finding the joint (and link) index 

# In[4]:


for i in range(p.getNumJoints(robot_id)):
    print(i, p.getJointInfo(robot_id, i)[1])


# ### Construct the robot system

# In[5]:


dt = 0.05
T = 100
dof = 7
sys = URDFRobot(dof=dof, robot_id=robot_id, dt = dt)


# #### Set the initial state

# In[6]:


q0 = np.random.rand(7)
q0 = np.array([0.,0.,0.,0.,0.,0.,0.])
#q0 = np.array([0.4201, 0.4719, 0.9226, 0.8089, 0.3113, 0.7598, 0.364 ])
x0 = np.concatenate([q0, np.zeros(7)])
sys.set_init_state(x0)


# #### Try forward kinematics

# In[7]:


pos0, quat0 = sys.compute_ee(x0, link_id)

#Put the ball at the end-effector
p.resetBasePositionAndOrientation(ballId, pos0, quat0)
print("pos0 = ", pos0)


# #### Set initial control output

# In[8]:


#set initial control output to be all zeros
us = np.zeros((T+1,sys.Du))
_ = sys.compute_matrices(x0, us[0])
xs = sys.rollout(us[:-1])


# #### Plot initial trajectory
sys.vis_traj(xs)
# #### Set the regularization cost coefficients Q and R 

# In[9]:


Q = np.eye(sys.Dx)*.1
Q[0:sys.dof,0:sys.dof] *= 0.01  #only put cost regularization on the velocity, not on the joint angles
Qf = np.eye(sys.Dx)*1
Qf[0:sys.dof,0:sys.dof] *= 0.01 #only put cost regularization on the velocity, not on the joint angles
R = np.eye(sys.Du)*1e-6
mu = 1e-6          #regularization coefficient


# #### Set end effector target 

# In[10]:


#W and WT: cost coefficients for the end-effector reaching task
p_target = np.array([0.5, -.6, 0.3])
W = np.eye(3)*1
WT = np.eye(3)*100


# In[11]:


p.resetBasePositionAndOrientation(ballId, p_target, (0,0,0,1))


# ### iLQR using cost model

# #### Define the cost

# In[12]:


#The costs consist of: a) state regularization (Q), b) control regularization (R), and c) End-effector reaching task (W)
#Running cost is for the time 0 <= t < T, while terminal cost is for the time t = T
costs = []

for i in range(T):
    runningStateCost = CostModelQuadratic(sys, Q)
    runningControlCost = CostModelQuadratic(sys, None, R)
    runningEECost = CostModelQuadraticTranslation(sys,W, link_id,p_target)
    runningCost = CostModelSum(sys, [runningStateCost, runningControlCost, runningEECost])
    costs += [runningCost]
    
terminalStateCost = CostModelQuadratic(sys,Qf)
terminalControlCost = CostModelQuadratic(sys, None,R)
terminalEECost = CostModelQuadraticTranslation(sys,WT, link_id,p_target)
terminalCost = CostModelSum(sys, [terminalStateCost, terminalControlCost, terminalEECost])

costs += [terminalCost]


# #### Construct ILQR

# In[13]:


ilqr_cost = ILQR(sys, mu)
ilqr_cost.set_init_state(x0)
ilqr_cost.set_timestep(T)
ilqr_cost.set_cost(costs)
ilqr_cost.set_state(xs, us) #set initial trajectory


# In[14]:


ilqr_cost.mu = 1e-6


# #### Solve and Plot

# In[15]:


n_iter = 30
ilqr_cost.solve(n_iter, method='recursive')
xs_batch, us_batch = ilqr_cost.xs, ilqr_cost.us

#clear_output()


# #### Play traj

# In[16]:


sys.vis_traj(ilqr_cost.xs)


# #### Compute Error

# In[17]:


pos, quat = sys.compute_ee(ilqr_cost.xs[-1], link_id)


# In[18]:
print("pos0 = ", pos0)
print("pos = ", pos)
print("p_target = ", p_target)
print("pos-p_target = ", pos-p_target)
