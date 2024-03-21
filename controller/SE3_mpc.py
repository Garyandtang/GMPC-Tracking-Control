import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from liecasadi import SO3, SO3Tangent, SE3Tangent, SE3
from utils.enum_class import TrajType, ControllerType
from controller.ref_traj_generator import TrajGenerator
from controller.se3_traj_generator import SE3TrajGenerator
import manifpy as manif
import time

class SE3MPC:
    def __init__(self, ref_traj_config):
        self.controllerType = ControllerType.SE3MPC
        self.nPos = 3  # [x, y, z]
        self.nQuat = 4  # [qx, qy, qz, qw]
        self.nTwist = 6  # [vx, vy, vz, wx, wy, wz]
        self.nControl = None # todo: not implemented yet
        self.nTraj = None
        self.dt = None
        self.nPred = None
        self.solve_time = 0.0
        self.setup_solver()
        self.set_ref_traj(ref_traj_config)
        self.set_control_bound(-40, 40, -120, 100)

    def set_ref_traj(self, traj_config):
        traj_generator = SE3TrajGenerator(traj_config)
        self.ref_state, self.ref_control, self.dt = traj_generator.get_traj()
        self.nTraj = self.ref_state.shape[1]

    def setup_solver(self, Q=100, R=1, nPred=10):
        self.Q = Q * np.diag(np.ones(self.nTwist))
        self.R = R * np.diag(np.ones(self.nTwist))
        self.nPred = nPred

    def set_control_bound(self, v_min, v_max, w_min, w_max):
        self.twist_min = np.array([v_min, v_min, v_min, w_min, w_min, w_min])
        self.twist_max = np.array([v_max, v_max, v_max, w_max, w_max, w_max])
    def solve(self, curr_state, t):
        start_time =time.time()
        if self.ref_state is None:
            raise ValueError('Reference trajectory is not set up yet!')
        dt = self.dt
        k = round(t / dt)
        R = self.R
        Q = self.Q


        # setup casadi solver
        opti = cs.Opti()
        pos = opti.variable(self.nPos, self.nPred)
        quat = opti.variable(self.nQuat, self.nPred)
        twist = opti.variable(self.nTwist, self.nPred - 1)

        # initial condition
        curr_pos = curr_state[:3]  # [x, y, z]
        curr_quat = curr_state[3:]  # [qx, qy, qz, qw]
        opti.subject_to(pos[:, 0] == curr_pos)
        opti.subject_to(quat[:, 0] == curr_quat)

        # dynamics constraints
        for i in range(self.nPred - 1):
            curr_SE3 = SE3(pos[:, i], quat[:, i])
            next_SE3 = SE3(pos[:, i + 1], quat[:, i + 1])
            curr_se3 = SE3Tangent(twist[:, i]*dt)
            forward_SE3 = curr_SE3 * curr_se3.exp()
            opti.subject_to(forward_SE3.pos == next_SE3.pos)
            opti.subject_to(forward_SE3.xyzw == next_SE3.xyzw)

        # cost function
        cost = 0
        for i in range(self.nPred - 1):
            index = min(k + i, self.nTraj - 1)
            curr_SE3 = SE3(pos[:, i], quat[:, i])
            ref_SE3 = SE3(self.ref_state[:3, index], self.ref_state[3:, index])
            SE3_diff = curr_SE3 - ref_SE3  # Log(SE3_ref^-1 * SE3), vector space
            cost += cs.mtimes([SE3_diff.vector().T, Q, SE3_diff.vector()])
            twist_d = np.zeros(6)
            cost += cs.mtimes([(twist[:, i] - twist_d).T, R, (twist[:, i] - twist_d)])

        last_SE3 = SE3(pos[:, -1], quat[:, -1])
        last_ref_SE3 = SE3(self.ref_state[:3, -1], self.ref_state[3:, -1])
        last_SE3_diff = last_SE3 - last_ref_SE3
        cost += cs.mtimes([last_SE3_diff.vector().T, Q, last_SE3_diff.vector()])

        opti.minimize(cost)

        # control bound
        for i in range(self.nTwist):
            opti.subject_to(twist[i, :] >= self.twist_min[i])
            opti.subject_to(twist[i, :] <= self.twist_max[i])

        # solve
        opti.solver("ipopt")
        sol = opti.solve()
        self.solve_time = time.time() - start_time

        return sol.value(twist[:, 0])



def test_SE3MPC():
    config = {'type': TrajType.POSE_REGULATION,
              'param': {'end_pos': np.array([1, 1, 0]),
                        'end_euler': np.array([0, 0, 0]),
                        'dt': 0.02,
                        'nTraj': 60}}
    controller = SE3MPC(config)
    ref_state = controller.ref_state
    ref_control = controller.ref_control
    dt = controller.dt
    nTraj = controller.nTraj

    init_state = np.array([0, 0, 0, 1, 0, 0, 0])
    store_state = np.zeros((7, nTraj))
    store_control = np.zeros((6, nTraj - 1))
    store_state[:, 0] = init_state
    t = 0
    for i in range(nTraj - 1):
        curr_state = store_state[:, i]
        curr_control = controller.solve(curr_state, t)
        store_control[:, i] = curr_control

        curr_SE3 = manif.SE3(curr_state[:3], curr_state[3:])
        curr_se3 = manif.SE3Tangent(curr_control * dt)
        next_SE3 = curr_SE3 + curr_se3
        next_state = np.zeros(7)
        next_state[:3] = next_SE3.translation()
        next_state[3:] = next_SE3.quat()
        store_state[:, i + 1] = next_state
        t += dt

    fig = plt.figure()
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(store_state[0,:], store_state[1, :], store_state[2, :])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot(ref_state[0, :], ref_state[1, :], ref_state[2, :], 'or')
    plt.show()

    # plot control
    fig = plt.figure()
    ax = plt.figure().add_subplot()
    ax.plot(store_control[0, :], label='v_x')
    ax.plot(store_control[1, :], label='v_y')
    ax.plot(store_control[2, :], label='v_z')
    ax.plot(store_control[3, :], label='w_x')
    ax.plot(store_control[4, :], label='w_y')
    ax.plot(store_control[5, :], label='w_z')
    ax.legend()
    plt.show()

    # plot x y
    fig = plt.figure()
    ax = plt.figure().add_subplot()
    ax.plot(store_state[0, :], store_state[1, :])
    ax.plot(ref_state[0, :], ref_state[1, :], 'or')
    plt.show()




if __name__ == '__main__':
    test_SE3MPC()






