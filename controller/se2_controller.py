import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import casadi as ca
import math
from utils.enum_class import WMRType, TrajType
from ref_traj_generator import TrajGenerator
class SE2Controller:
    def __init__(self, Kp = np.array([1, 1, 1])):
        self.Kp = Kp

    def feedback_control(self, curr_state, curr_ref_state):
        """
        :param curr_state: [x, y, theta]
        :param ref_state: [x_d, y_d, theta_d]
        :return: control:[v, w]
        """
        X = SE2(curr_state[0], curr_state[1], curr_state[2])
        X_d = SE2(curr_ref_state[0], curr_ref_state[1], curr_ref_state[2])
        X_err = X.between(X_d)
        se2_err = X_err.log()
        twist = self.Kp * se2_err.coeffs()
        control = self.local_twsit_to_vel_cmd(twist)
        return control

    def feedback_feedforward_control(self, curr_state, curr_ref_state, curr_ref_control):
        control_fb = self.feedback_control(curr_state, curr_ref_state)
        control_ff = curr_ref_control
        vel_cmd = control_fb + control_ff
        return vel_cmd

    def local_twsit_to_vel_cmd(self, local_twist, type=WMRType.UNICYCLE):
        if type == WMRType.UNICYCLE:
            return np.array([local_twist[0], local_twist[2]])
            # return np.array([local_twist[0]**2 + local_twist[1]**2, local_twist[2]])

    def vel_cmd_to_local_twist(self, vel_cmd, type=WMRType.UNICYCLE):
        if type == WMRType.UNICYCLE:
            return np.array([vel_cmd[0], 0, vel_cmd[1]])



def test_se2_controller():
    # set up init state and reference trajectory
    init_state = np.array([-0.2, -0.2, np.pi/6])
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.5,
                             'angular_vel': 0.5,
                             'nTraj': 6000,
                             'dt': 0.02}}
    traj_generator = TrajGenerator(traj_config)
    ref_SE2, ref_twist, dt = traj_generator.get_traj()
    se2_controller = SE2Controller()

    # container for recording SE2 state and twist
    nSE2 = 4
    nTwist = 3
    store_SE2 = np.zeros((nSE2, ref_SE2.shape[1]))
    store_twist = np.zeros((nTwist, ref_SE2.shape[1]))
    store_SE2[:, 0] = SE2(init_state[0], init_state[1], init_state[2]).coeffs()

    t = 0
    for i in range(ref_SE2.shape[1]-1):
        curr_SE2 = store_SE2[:, i]
        curr_ref_SE2 = ref_SE2[:, i]
        curr_ref_twist = ref_twist[:, i]
        curr_twist = se2_controller.feedback_feedforward_control(curr_SE2, curr_ref_SE2, curr_ref_twist)
        curr_vel_cmd = se2_controller.local_twsit_to_vel_cmd(curr_twist)
        curr_twist = se2_controller.vel_cmd_to_local_twist(curr_vel_cmd)
        store_twist[:, i] = curr_twist
        # next SE2 state
        next_SE2 = SE2(curr_SE2) + SE2Tangent(curr_twist) * dt
        store_SE2[:, i + 1] = next_SE2.coeffs()
        t += dt

    # plot
    plt.figure()
    plt.plot(store_SE2[0, :], store_SE2[1, :], 'b')
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'r')
    plt.show()

    # plot distance error
    plt.figure()
    plt.plot(np.linalg.norm(store_SE2[0:2, :] - ref_SE2[0:2, :], axis=0))
    plt.title('distance error')
    plt.show()

    # plot angle error
    plt.figure()
    orientation_store = np.zeros(ref_SE2.shape[1])
    for i in range(ref_SE2.shape[1]):
        X_d = SE2(ref_SE2[:, i])
        X = SE2(store_SE2[:, i])
        X_d_inv_X = SO2(X_d.angle()).between(SO2(X.angle()))
        orientation_store[i] = scipy.linalg.norm(X_d_inv_X.log().coeffs())


    plt.figure()
    plt.plot(orientation_store[0:])
    plt.title('orientation difference')
    plt.show()

def one_step(curr_state, curr_control, dt):
    curr_SE2 = SE2(curr_state[0], curr_state[1], curr_state[2])
    curr_twist = SE2Tangent(curr_control[0], 0, curr_control[1])
    next_SE2 = curr_SE2 + curr_twist * dt
    next_state = np.array([next_SE2.x(), next_SE2.y(), next_SE2.angle()])
    return next_state

def test_pose_regulation():
    init_state = np.array([1.1, 1.2, 0])
    config = {'type': TrajType.POSE_REGULATION,
              'param': {'end_state': np.array([0, 0, 0]),
                        'dt': 0.05,
                        'nTraj': 1700}}
    traj_generator = TrajGenerator(config)
    ref_state, ref_control, dt = traj_generator.get_traj()
    se2_controller = SE2Controller()
    # container for recording SE2 state and twist
    nState = 3
    nControl = 2
    store_state = np.zeros((nState, ref_state.shape[1]))
    store_control = np.zeros((nControl, ref_state.shape[1]))
    store_state[:, 0] = init_state

    t = 0
    for i in range(ref_state.shape[1]-1):
        curr_state = store_state[:, i]
        curr_ref_state = ref_state[:, i]
        curr_ref_control = ref_control[:, i]
        curr_control = se2_controller.feedback_feedforward_control(curr_state, curr_ref_state, curr_ref_control)
        store_control[:, i] = curr_control
        # next state
        next_state = one_step(curr_state, curr_control, dt)
        store_state[:, i + 1] = next_state
        t += dt

    # plot
    plt.figure()
    plt.plot(store_state[0, :], store_state[1, :], 'b')
    # plot finial state
    plt.plot(store_state[0, 0], store_state[1, 0], 'ro')
    plt.plot(ref_state[0, -1], ref_state[1, -1], 'bo')
    plt.show()

    # # plot (x, y, theta) pose figure with arrow
    # plt.figure()
    # plt.plot(store_SE2[0, :], store_SE2[1, :], 'b')
    # plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'r')
    # plt.quiver(store_SE2[0, :], store_SE2[1, :], np.cos(store_SE2[2, :]), np.sin(store_SE2[2, :]), color='b')
    # plt.quiver(ref_SE2[0, :], ref_SE2[1, :], np.cos(ref_SE2[2, :]), np.sin(ref_SE2[2, :]), color='r')
    # plt.show()




    # # plot distance error
    # plt.figure()
    # plt.plot(np.linalg.norm(store_SE2[0:2, :] - ref_SE2[0:2, :], axis=0))
    # plt.title('distance error')
    # plt.show()
    #
    # # plot angle error
    # plt.figure()
    # orientation_store = np.zeros(ref_SE2.shape[1])
    # for i in range(ref_SE2.shape[1]):
    #     X_d = SE2(ref_SE2[:, i])
    #     X = SE2(store_SE2[:, i])
    #     X_d_inv_X = SO2(X_d.angle()).between(SO2(X.angle()))
    #     orientation_store[i] = scipy.linalg.norm(X_d_inv_X.log().coeffs())
    #
    # plt.figure()
    # plt.plot(orientation_store[0:])
    # plt.title('orientation difference')
    # plt.show()



if __name__ == '__main__':
    test_pose_regulation()



