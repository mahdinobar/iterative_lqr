import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pybullet as p
import time
from scipy.spatial.transform import Rotation as R

class LinearSystem():
    def __init__(self,A,B):
        self.A = A
        self.B = B
        self.Dx = A.shape[0]
        self.Du = B.shape[1]
        
    def reset_AB(self, A,B):
        self.A = A
        self.B = B
        
    def set_init_state(self,x0):
        self.x0 = x0
    
    def compute_matrices(self,x,u):
        #compute the derivatives of the dynamics
        return self.A,self.B
    
    def compute_ee(self,x, ee_id=1):
        #The end-effector for a point mass system is simply its position
        #The ee_id is added as dummy variable, just for uniformity of notation with other systems
        return x[:int(self.Dx/2)], None 
    
    def compute_Jacobian(self,x, ee_id=1):
        #The end-effector Jacobian for a point mass system is simply an identity matrix
        #The ee_id is added as dummy variable, just for uniformity of notation with other systems
        return np.eye(int(self.Dx/2)) 
    
    
    def step(self, x, u):
        return self.A.dot(x) + self.B.dot(u)
    
    def rollout(self,us):
        x_cur = self.x0
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)

        
class TwoLinkRobot():
    def __init__(self, dt = 0.01):
        self.dt = dt
        self.Dx = 4
        self.Du = 2
        self.dof = 2
        self.l1 = 1.5
        self.l2 = 1
        self.p_ref = np.zeros(2)
        
    def set_init_state(self,x0):
        self.x0 = x0
        
    def set_pref(self, p_ref):
        self.p_ref = p_ref
    
    def compute_matrices(self,x,u):
        #compute the derivatives of the dynamics
        A = np.eye(self.Dx)
        B = np.zeros((self.Dx,self.Du))
        
        A[0,2] = self.dt
        A[1,3] = self.dt
        
        B[2:,:] = np.eye(self.Du)
        
        self.A, self.B = A,B
        return A,B
    
    def compute_ee(self,x, ee_id=0):
        self.p1 = np.array([self.l1*np.cos(x[0]), self.l1*np.sin(x[0])])
        self.p2 = np.array([self.p1[0] + self.l2*np.cos(x[0] + x[1]), self.p1[1] + self.l2*np.sin(x[0] + x[1])])
        return self.p2, self.p1

    
    def compute_Jacobian(self, x, ee_id=0):
        J = np.zeros((2, 2))
        s1 = np.sin(x[0])
        c1 = np.cos(x[0])
        s12 = np.sin(x[0] + x[1])
        c12 = np.cos(x[0] + x[1])
        
        J[0,0] = -self.l1*s1 - self.l2*s12
        J[0,1] = - self.l2*s12
        J[1,0] =  self.l1*c1 + self.l2*c12
        J[1,1] =  self.l2*c12
        
        self.J = J
        return self.J
        
    def step(self, x, u):
        x_next = self.A.dot(x) + self.B.dot(u)
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    
    def plot(self, x, color='k'):
        self.compute_ee(x)
        
        line1 = plt.plot(np.array([0, self.p1[0]]),np.array([0, self.p1[1]]) , marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        line2 = plt.plot(np.array([self.p1[0], self.p2[0]]),np.array([self.p1[1], self.p2[1]]) , marker='o', color=color, lw=10, mfc='w', solid_capstyle='round')
        xlim = [-1.5*(self.l1+self.l2), 1.5*(self.l1+self.l2)]
        #plt.axes().set_aspect('equal')
        plt.axis(xlim+xlim)
        return line1,line2

    def plot_traj(self, xs, dt = 0.1):
        for x in xs:
            clear_output(wait=True)
            self.plot(x)
            plt.plot(self.p_ref[0], self.p_ref[1], '*')
            plt.show()
            time.sleep(self.dt)
            
class URDFRobot():
    def __init__(self, dof, robot_id, joint_indices = None, dt = 0.01):
        self.dt = dt
        self.Dx = dof*2
        self.Du = dof
        self.dof = dof
        self.robot_id = robot_id
        if joint_indices is None:
            self.joint_indices = np.arange(dof)
        else:
            self.joint_indices = joint_indices
        
    def set_init_state(self,x0):
        self.x0 = x0
        self.set_q(x0)

    def compute_matrices(self,x,u):
        #compute the derivatives of the dynamics
        A = np.eye(self.Dx)
        B = np.zeros((self.Dx,self.Du))
        
        A[:self.dof, self.dof:] = np.eye(self.dof)*self.dt
        
        #B[self.dof:,:] = np.eye(self.Du)
        B[:self.dof,:] = np.eye(self.Du) * self.dt * self.dt /2
        B[-self.dof:,:] = np.eye(self.Du) * self.dt    

        self.A, self.B = A,B
        return A,B
    
    def compute_ee(self,x, ee_id):
        self.set_q(x)
        ee_data = p.getLinkState(self.robot_id, ee_id)
        pos = np.array(ee_data[0])
        quat = np.array(ee_data[1])
        return pos, quat
    
    def compute_Jacobian(self, x, ee_id):
        zeros = [0.]*self.dof
        Jl, Ja = p.calculateJacobian(self.robot_id, ee_id, [0.,0.,0.], x[:self.dof].tolist(),zeros,zeros)
        Jl, Ja = np.array(Jl), np.array(Ja)
        self.J = np.concatenate([Jl, Ja], axis=0)
        return self.J
        
    def step(self, x, u):
        x_next = self.A.dot(x) + self.B.dot(u)
        return x_next
    
    def rollout(self,us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)
    
    def set_q(self, x):
        q = x[:self.dof]
        for i in range(self.dof):
            p.resetJointState(self.robot_id, self.joint_indices[i], q[i])
        return 

    def vis_traj(self, xs, dt = 0.1):
        for x in xs:
            clear_output(wait=True)
            self.set_q(x)
            time.sleep(self.dt)


class URDFRobot_spacetime_dual():
    '''
    two same Panda arms spacetime framework with single integrator control
    '''
    def __init__(self, dof, robot1_id, robot2_id, joint_indices=None, dt=0.01):
        self.dt = dt
        self.Dx = dof * 2 + 2  # x=(q11,q12,...,q17,q21,q22,...,q27,s1,s2)
        self.Du = dof * 2 + 2  # u=(dq11,dq12,...,dq17,dq21,dq22,...,dq27,ds1,ds2)
        self.dof = dof
        self.robot1_id = robot1_id
        self.robot2_id = robot2_id
        # todo
        if joint_indices is None:
            self.joint_indices = np.arange(dof)
        else:
            self.joint_indices = joint_indices

    def set_init_state(self, x0):
        self.x0 = x0
        self.set_q(x0)

    def compute_matrices(self, x, u):
        # linearize system
        A = np.eye(self.Dx)
        B = np.diag(np.concatenate((u[14]*np.ones(self.dof),u[15]*np.ones(self.dof),[1,1])))
        B[:self.dof, 14] = u[:self.dof]
        B[self.dof:self.dof*2, 15] = u[self.dof:self.dof*2]
        self.A, self.B = A, B
        return A, B

    def compute_ee(self, x, ee_id):
        self.set_q(x)
        ee1_data = p.getLinkState(self.robot1_id, ee_id)
        ee2_data = p.getLinkState(self.robot2_id, ee_id)
        pos1 = np.array(ee1_data[0])
        quat1 = np.array(ee1_data[1])
        pos2 = np.array(ee2_data[0])
        quat2 = np.array(ee2_data[1])
        # print("++++++",np.linalg.norm(pos1 - pos2))
        return pos1, quat1, pos2, quat2

    def compute_elipsoids(self, x, ee_id=None):
        self.set_q(x)
        centers1 = np.zeros((p.getNumJoints(self.robot1_id),3))
        sizes1 = 1*np.array([[.2,.2,.1],
                          [.1,.2,.2],
                          [.1,.2,.2],
                          [.2,.1,.2],
                          [.2,.2,.2],
                          [.1,.2,.3],
                          [.1,.1,.2],
                          [.1,.1,.1],
                          [.1,.2,.1],
                          [.05,.05,.1],
                          [.05,.05,.1]])
        rotations1 = np.zeros((p.getNumJoints(self.robot1_id), 3, 3))
        centers2 = np.zeros((p.getNumJoints(self.robot1_id),3))
        sizes2 = sizes1
        rotations2 = np.zeros((p.getNumJoints(self.robot1_id), 3, 3))
        # scipy: quat: (x, y, z, w)
        # pybullet: quat: [x,y,z,w]
        for i in range(p.getNumJoints(self.robot1_id)):
            centers1[i, :]=p.getLinkState(self.robot1_id, i)[0]
            centers2[i, :]=p.getLinkState(self.robot2_id, i)[0]
            r = R.from_quat(p.getLinkState(self.robot1_id, i)[1])
            rotations1[i, :, :] = r.as_matrix()
            r = R.from_quat(p.getLinkState(self.robot2_id, i)[1])
            rotations2[i, :, :] = r.as_matrix()
        return centers1, sizes1, rotations1, centers2, sizes2, rotations2

    def compute_ellipsoid_Jacobian(self, x, link1, link2):
        zeros = [0.] * self.dof
        # todo give nonzero joint velocities?
        Jl1, Ja1 = p.calculateJacobian(self.robot1_id, link1, [0., 0., 0.], x[:self.dof].tolist(), zeros, zeros)
        Jl1, Ja1 = np.array(Jl1), np.array(Ja1)
        self.J1 = np.concatenate([Jl1, Ja1], axis=0)
        # todo give nonzero joint velocities?
        Jl2, Ja2 = p.calculateJacobian(self.robot2_id, link2, [0., 0., 0.], x[self.dof:self.dof*2].tolist(), zeros, zeros)
        Jl2, Ja2 = np.array(Jl2), np.array(Ja2)
        self.J2 = np.concatenate([Jl2, Ja2], axis=0)
        return self.J1, self.J2

    def compute_Jacobian(self, x, ee_id):
        zeros = [0.] * self.dof
        # todo give nonzero joint velocities?
        Jl1, Ja1 = p.calculateJacobian(self.robot1_id, ee_id, [0., 0., 0.], x[:self.dof].tolist(), zeros, zeros)
        Jl1, Ja1 = np.array(Jl1), np.array(Ja1)
        self.J1 = np.concatenate([Jl1, Ja1], axis=0)
        # todo give nonzero joint velocities?
        Jl2, Ja2 = p.calculateJacobian(self.robot2_id, ee_id, [0., 0., 0.], x[self.dof:self.dof*2].tolist(), zeros, zeros)
        Jl2, Ja2 = np.array(Jl2), np.array(Ja2)

        ### Important
        # if baseOrientation is modified for robot2_id then the jacobian should be rotated back to be with respect ot World coordinate system
        Jl2[[0, 1]] = Jl2[[1, 0]]
        Jl2[[1]] = -Jl2[[1]]

        self.J2 = np.concatenate([Jl2, Ja2], axis=0)
        return self.J1, self.J2

    def step(self, x, u):
        # todo replace with linearized system ?
        B = np.diag(np.concatenate((u[14]*np.ones(self.dof),u[15]*np.ones(self.dof),[1,1])))
        x_next = x + B.dot(u)
        return x_next

    def rollout(self, us):
        x_cur = self.x0.copy()
        xs = [x_cur]
        T = len(us)
        for i in range(T):
            x_cur = self.step(x_cur, us[i])
            xs += [x_cur]
        return np.array(xs)

    def set_q(self, x):
        q1 = x[:self.dof]
        q2 = x[self.dof:self.dof*2]
        for i in range(self.dof):
            p.getJointState(self.robot1_id, 10)
            p.resetJointState(self.robot1_id, self.joint_indices[i], q1[i])
            p.resetJointState(self.robot2_id, self.joint_indices[i], q2[i])
        return

    def vis_traj(self, xs, vis_dt=0.1):
        for x in xs:
            clear_output(wait=True)
            self.set_q(x)
            time.sleep(vis_dt)