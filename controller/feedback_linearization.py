import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import casadi as ca
import math
from utils.enum_class import WMRType, TrajType, ControllerType
from planner.ref_traj_generator import TrajGenerator

class FBLinearizationController:
    def __init__(self, Kp=np.array([2, 2, 2])):
        self.controllerType = ControllerType.FEEDBACK_LINEARIZATION
        k1, k2, k3 = Kp
        self.K = np.array([[-k1, 0, 0],
                           [0, -k2, -k3]])

        self.set_control_bound()
        self.solve_time = 0.0

    def feedback_control(self, curr_state, ref_state, ref_vel_cmd):
        """
        :param curr_state: [x, y, theta]
        :param ref_state: [x_d, y_d, theta_d]
        :return: vel_cmd:[v, w]
        """
        start_time = time.time()
        v_d, w_d = ref_vel_cmd
        state_diff = ref_state - curr_state
        state_diff[2] = np.arctan2(np.sin(state_diff[2]), np.cos(state_diff[2]))
        frame_rot = np.array([[np.cos(curr_state[2]), np.sin(curr_state[2]), 0],
                              [-np.sin(curr_state[2]), np.cos(curr_state[2]), 0],
                              [0, 0, 1]])
        error = frame_rot @ state_diff
        u = self.K @ error
        v = v_d * np.cos(error[2]) - u[0]
        w = w_d - u[1]
        vel_cmd = np.array([v, w])
        self.solve_time = time.time() - start_time
        return vel_cmd

    def get_solve_time(self):
        return self.solve_time

    def set_control_bound(self, v_min=-4, v_max=4, w_min=-4, w_max=4):
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max

    def saturate_control(self, vel_cmd):
        v, w = vel_cmd
        if v < self.v_min:
            v = self.v_min
        elif v > self.v_max:
            v = self.v_max
        if w < self.w_min:
            w = self.w_min
        elif w > self.w_max:
            w = self.w_max
        return np.array([v, w])


def test_fb_linearization_controller():
    # set up init state and reference trajectory
    init_state = np.array([-0.2, -0.2, np.pi / 6])
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.5,
                             'angular_vel': 0.5,
                             'nTraj': 600,
                             'dt': 0.02}}
    traj_generator = TrajGenerator(traj_config)
    ref_SE2, ref_twist, dt = traj_generator.get_traj()
    fb_linearization_controller = FBLinearizationController()

    # plot reference trajectory
    plt.figure()
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'b', label='reference')
    plt.plot(init_state[0], init_state[1], 'r*', label='init state')
    plt.legend()
    plt.show()

    # container for recording SE2 state and twist
    nState = 3
    nVelCmd = 2
    store_state = np.zeros((nState, ref_SE2.shape[1]))
    store_vel_cmd = np.zeros((nVelCmd, ref_twist.shape[1]))

    # simulate
    t = 0
    store_state[:, 0] = init_state
    for i in range(ref_SE2.shape[1] - 1):
        curr_state = store_state[:, i]
        curr_SE2 = SE2(curr_state[0], curr_state[1], curr_state[2])
        ref_SE2_coeff = ref_SE2[:, i]
        X_ref = SE2(ref_SE2_coeff)
        ref_state = np.array([X_ref.x(), X_ref.y(), X_ref.angle()])
        ref_vel_cmd = np.array([ref_twist[0, i], ref_twist[2, i]])
        vel_cmd = fb_linearization_controller.feedback_control(curr_state, ref_state, ref_vel_cmd)
        twist_cmd = np.array([vel_cmd[0], 0, vel_cmd[1]])
        next_SE2 = curr_SE2 + SE2Tangent(twist_cmd) * dt
        next_state = np.array([next_SE2.x(), next_SE2.y(), next_SE2.angle()])
        store_state[:, i + 1] = next_state
        store_vel_cmd[:, i] = vel_cmd
        t += dt

    # plot
    plt.figure()
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'b', label='reference')
    plt.plot(store_state[0, :], store_state[1, :], 'r', label='actual')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory Tracking')
    plt.grid(True)
    plt.show()

    # plot distance difference
    plt.figure()
    distance_store = np.linalg.norm(store_state[0:2, :] - ref_SE2[0:2, :], axis=0)
    plt.plot(distance_store)
    plt.xlabel('time')
    plt.ylabel('distance')
    plt.title('Distance Difference')
    plt.grid(True)
    plt.show()

    # plot orientation difference
    plt.figure()
    orientation_store = np.zeros(ref_SE2.shape[1])
    for i in range(ref_SE2.shape[1]):
        theta = store_state[2, i]
        theta_d = SE2(ref_SE2[:, i]).angle()
        orientataion_diff = SO2(theta_d).between(SO2(theta))
        orientation_store[i] = scipy.linalg.norm(orientataion_diff.log().coeffs())

    plt.figure()
    plt.plot(orientation_store[0:])
    plt.title('orientation difference')
    plt.xlabel('time')
    plt.ylabel('orientation difference')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    test_fb_linearization_controller()
