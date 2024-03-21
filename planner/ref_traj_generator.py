import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import casadi as ca
import math
from utils.enum_class import TrajType

class PlannerManager:

    def __init__(self, config):
        self.traj_generator = TrajGenerator(config)
        self.ref_state, self.ref_control, self.dt = self.traj_generator.get_traj()
        self.nTraj = self.ref_state.shape[1]


"""
Simple trajectory generator for unicycle model, given the type of trajectory,
it generates the reference velocity and state for the controller to track.
the state is using forward Euler integration to calculate the next state.

Example config:
For circle trajectory:
config = {'type': TrajType.CIRCLE,
                      'param': {'start_state': np.array([0, 0, 0]),
                                'linear_vel': 0.5,
                                'angular_vel': 0.5,
                                'dt': 0.02,
                                'nTraj': 170}}
For eight trajectory:
config = {'type': TrajType.EIGHT,
          'param': {'start_state': np.array([0, 0, 0]),
                    'dt': 0.05,
                    'v_scale': 1,
                    'nTraj': 170}}
"""
class TrajGenerator:
    def __init__(self, config):
        self.nState = 3
        self.nControl = 2
        if not config:
            config = {'type': TrajType.CIRCLE,
                      'param': {'start_state': np.array([0, 0, 0]),
                                'linear_vel': 0.5,
                                'angular_vel': 0.5,
                                'dt': 0.02,
                                'nTraj': 170}}
        if config['type'] == TrajType.CIRCLE:
            self.generate_circle_traj(config['param'])
        elif config['type'] == TrajType.EIGHT:
            self.generate_eight_traj(config['param'])
        elif config['type'] == TrajType.POSE_REGULATION:
            self.generate_pose_regulation_traj(config['param'])
        elif config['type'] == TrajType.CONSTANT:
            self.generate_circle_traj(config['param'])

    def generate_circle_traj(self, config):
        # example
        # config = {'type': TrajType.CIRCLE,
        #           'param': {'start_state': np.array([0, 0, 0]),
        #                     'linear_vel': 0.5,
        #                     'angular_vel': 0.5,
        #                     'dt': 0.02,
        #                     'nTraj': 170}}
        self.dt = config['dt']
        self.nTraj = config['nTraj']
        self.ref_state = np.zeros((self.nState, self.nTraj))  # [x, y, theta]
        self.ref_control = np.zeros((self.nControl, self.nTraj))  # [v, w]
        state = config['start_state']
        self.ref_state[:, 0] = state
        vel_cmd = np.array([config['linear_vel'], config['angular_vel']])
        self.ref_control[:, 0] = vel_cmd
        v = self.vel_cmd_to_local_vel(vel_cmd)  # constant velocity
        for i in range(self.nTraj - 1):  # 0 to nTraj-2
            curr_state = self.ref_state[:, i]
            X = SE2(curr_state[0], curr_state[1], curr_state[2])
            X_next = X + SE2Tangent(v * self.dt)  # X * SE2Tangent(xi * self.dt).exp()
            self.ref_state[:, i + 1] = np.array([X_next.x(), X_next.y(), X_next.angle()])
            self.ref_control[:, i + 1] = vel_cmd

    def generate_eight_traj(self, config):
        self.dt = config.get('dt', 0.05)
        T = round(1/self.dt)
        self.nTraj = config.get('nTraj', 170)
        init_state = config.get('start_state', np.array([0, 0, 0]))
        v_scale = config.get('v_scale', 1)
        w_scale = config.get('w_scale', 1)

        self.ref_state = np.zeros((self.nState, self.nTraj))  # [x, y, theta]
        self.ref_control = np.zeros((self.nControl, self.nTraj))  # [v, w]
        self.ref_state[:, 0] = init_state
        t = 0.0
        for i in range(self.nTraj - 1):  # 0 to nTraj-2
            xdot = v_scale * np.cos(w_scale*4.0 * np.pi * t / T) * 4.0 * np.pi / T
            ydot = v_scale * np.cos(w_scale*2.0 * np.pi * t / T) * 2.0 * np.pi / T
            v = np.sqrt(xdot ** 2 + ydot ** 2)
            # calculate angular velocity
            xdotdot = -v_scale * np.sin(w_scale*4 * np.pi * t / T) * (4.0 * np.pi / T) ** 2
            ydotdot = -v_scale * np.sin(w_scale*2 * np.pi * t / T) * (2.0 * np.pi / T) ** 2
            w = (ydotdot * xdot - xdotdot * ydot) / (xdot ** 2 + ydot ** 2)
            # calculate twist
            vel_cmd = np.array([v, w])
            twist = np.array([v, 0, w])
            X = SE2(self.ref_state[0, i], self.ref_state[1, i], self.ref_state[2, i])
            X_next = X + SE2Tangent(twist * self.dt)
            self.ref_state[:, i + 1] = np.array([X_next.x(), X_next.y(), X_next.angle()])
            self.ref_control[:, i + 1] = vel_cmd
            # increment time
            if t == T:
                t = 0.0
            else:
                t = t + self.dt
        self.ref_control[:, self.nTraj - 1] = self.ref_control[:, self.nTraj - 2]

    def generate_pose_regulation_traj(self, config):
        # example of pose regulation config
        # config = {'type': TrajType.POSE_REGULATION,
        #           'param': {'end_state': np.array([0, 0, 0]),
        #                     'dt': 0.05,
        #                     'nTraj': 170}}
        self.dt = config['dt']
        self.nTraj = config['nTraj']
        self.ref_state = np.zeros((self.nState, self.nTraj))  # [x, y, theta]
        self.ref_control = np.zeros((self.nControl, self.nTraj))  # [v, w]
        end_state = config['end_state']
        for i in range(self.nTraj):
            self.ref_state[:, i] = end_state

    def get_traj(self):
        v_min_, v_max_, w_min_, w_max_ = self.get_vel_bound()
        print("v_min: ", v_min_, "v_max: ", v_max_, "w_min: ", w_min_, "w_max: ", w_max_)
        return self.ref_state, self.ref_control, self.dt

    def get_vel_bound(self):
        v_min = np.min(self.ref_control[0, :])
        v_max = np.max(self.ref_control[0, :])
        w_min = np.min(self.ref_control[1, :])
        w_max = np.max(self.ref_control[1, :])
        return v_min, v_max, w_min, w_max
    def vel_cmd_to_local_vel(self, vel_cmd):
        # non-holonomic constraint
        # vel_cmd: [v, w]
        # return: [v, 0, w]
        return np.array([vel_cmd[0], 0, vel_cmd[1]])


def test_traj_generator():
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'dt': 0.2,
                             'linear_vel': 0.1,
                             'angular_vel': 0.05,  # don't change this
                             'nTraj': 650}}
    traj_generator = TrajGenerator(traj_config)
    ref_traj, ref_v, dt = traj_generator.get_traj()
    plt.figure(1)
    # show grid
    plt.grid()
    plt.plot(ref_traj[0, :], ref_traj[1, :], 'b')
    plt.title('Reference Trajectory')
    plt.show()

    # convert to [x, y, theta]
    traj = np.zeros((3, ref_traj.shape[1]))
    for i in range(ref_traj.shape[1]):
        X = SE2(ref_traj[0, i], ref_traj[1, i], ref_traj[2, i])
        traj[:, i] = np.array([X.x(), X.y(), X.angle()])

    plt.figure(2)
    plt.plot(traj[0, :], traj[1, :], 'b')
    plt.title('Reference Trajectory [x, y, theta]')
    plt.show()


def test_pose_regulation_traj_generator():
    config = {'type': TrajType.POSE_REGULATION,
              'param': {'end_state': np.array([0, 0, 0]),
                        'dt': 0.05,
                        'nTraj': 170}}
    traj_generator = TrajGenerator(config)
    ref_SE2, ref_twist, dt = traj_generator.get_traj()
    plt.figure(1)
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'b')
    plt.title('Reference Trajectory')
    plt.show()

def test_time_varying_traj_generator():
    config = {'type': TrajType.TIME_VARYING,
              'param': {'start_state': np.array([0, 0, 0]),
                        'dt': 0.1,
                        'nTraj': 3000}}
    traj_generator = TrajGenerator(config)
    ref_SE2, ref_twist, dt = traj_generator.get_traj()
    plt.figure(1)
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'b')
    plt.title('Reference Trajectory')
    plt.show()

def test_generate_eight_traj():
    config = {'type': TrajType.EIGHT,
              'param': {'start_state': np.array([1, 1, 0]),
                        'dt': 0.02,
                        'scale': 1,
                        'nTraj': 3000}}
    traj_generator = TrajGenerator(config)
    ref_SE2, ref_twist, dt = traj_generator.get_traj()
    plt.figure(1)
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'b')
    plt.title('Reference Trajectory')
    plt.show()


if __name__ == '__main__':
    test_traj_generator()
