import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent, SO3, SO3Tangent
import casadi as ca
import math
from utils.enum_class import TrajType


class SE3TrajGenerator:
    def __init__(self, config):
        self.nState = 7
        self.nControl = 6
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
        elif config['type'] == TrajType.TIME_VARYING:
            self.generate_time_vary_traj(config['param'])


    def generate_pose_regulation_traj(self, config):
        # example of pose regulation config
        # config = {'type': TrajType.POSE_REGULATION,
        #           'param': {'end_pos': np.array([0, 0, 0]),
        #                     'end_euler': np.array([0, 0, 0]),
        #                     'dt': 0.05,
        #                     'nTraj': 170}}
        end_pos = config['end_pos']
        end_euler = config['end_euler']
        end_SO3 = SO3(end_euler[0], end_euler[1], end_euler[2])
        end_quat = end_SO3.quat().flatten()
        self.dt = config['dt']
        self.nTraj = config['nTraj']
        self.ref_state = np.zeros((self.nState, self.nTraj))  # [x, y, z, qx, qy, qz, qw]
        self.ref_control = np.zeros((self.nControl, self.nTraj))  # [vx, vy, vz, wx, wy, wz]
        for i in range(self.nTraj):
            self.ref_state[:3, i] = end_pos
            self.ref_state[3:, i] = end_quat


    def generate_time_vary_traj(self, config):

        self.dt = config['dt']
        self.nTraj = config['nTraj']
        self.ref_SE2 = np.zeros((self.nSE2, self.nTraj))  # [x, y, cos(theta), sin(theta)]
        self.ref_twist = np.zeros((self.nTwist, self.nTraj))  # [vx, vy, w]
        state = config['start_state']
        self.ref_SE2[:, 0] = SE2(state[0], state[1], state[2]).coeffs()
        vel_cmd = np.array([np.cos(0), np.sin(0)])
        v = self.vel_cmd_to_local_vel(vel_cmd)  # constant velocity
        self.ref_twist[:, 0] = v

        for i in range(self.nTraj - 1):  # 0 to nTraj-2
            SE2_coeffs = self.ref_SE2[:, i]
            twist = self.ref_twist[:, i]
            X = SE2(SE2_coeffs)  # SE2 state
            X = X + SE2Tangent(twist * self.dt)  # X * SE2Tangent(xi * self.dt).exp()
            vel_cmd = np.array([0.8*np.cos((i+1)*self.dt), np.sin(2*(i+1)*self.dt)*2*np.pi])
            print("vel_cmd: ", vel_cmd)
            print("vel_cmd_to_local_vel: ", self.vel_cmd_to_local_vel(vel_cmd))
            self.ref_SE2[:, i + 1] = X.coeffs()
            self.ref_twist[:, i + 1] = self.vel_cmd_to_local_vel(vel_cmd)

    # def generate_eight_traj(self, config):
    #     raise NotImplementedError

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
    config = {'type': TrajType.POSE_REGULATION,
              'param': {'end_pos': np.array([1, 2, 3]),
                        'end_euler': np.array([1, 3, 2]),
                        'dt': 0.05,
                        'nTraj': 170}}
    traj_generator = SE3TrajGenerator(config)
    ref_state, ref_control, dt = traj_generator.get_traj()
    print("ref_state: ", ref_state)
    print("ref_control: ", ref_control)
    print("dt: ", dt)

if __name__ == '__main__':
    test_traj_generator()
