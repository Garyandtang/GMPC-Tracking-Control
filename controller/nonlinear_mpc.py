import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import math
from planner.ref_traj_generator import TrajGenerator
from utils.enum_class import TrajType, CostType, DynamicsType
import casadi as ca
from utils.enum_class import CostType, DynamicsType, ControllerType
from utils.symbolic_system import FirstOrderModel
from liecasadi import SO3
import time

"""
naive MPC for unicycle model
"""


class UnicycleModel:
    def __init__(self, config: dict = {}, pyb_freq: int = 50, **kwargs):
        self.nState = 3
        self.nControl = 2

        self.control_freq = pyb_freq
        self.dt = 1. / self.control_freq

        # setup configuration
        self.config = config
        if not config:
            self.config = {'cost_type': CostType.POSITION, 'dynamics_type': DynamicsType.EULER_FIRST_ORDER}
        if self.config['dynamics_type'] == DynamicsType.EULER_FIRST_ORDER:
            self.set_up_euler_first_order_dynamics()
        elif self.config['dynamics_type'] == DynamicsType.EULER_SECOND_ORDER:
            pass
        elif self.config['dynamics_type'] == DynamicsType.DIFF_FLAT:
            pass
        print(config.get("dynamics_type"))

    def set_up_euler_first_order_dynamics(self):
        print("Setting up Euler first order dynamics")
        nx = self.nState
        nu = self.nControl
        # state
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        theta = ca.MX.sym('theta')
        X = ca.vertcat(x, y, theta)
        # control

        v = ca.MX.sym('v')  # linear velocity
        w = ca.MX.sym('w')  # angular velocity
        U = ca.vertcat(v, w)

        # state derivative
        x_dot = ca.cos(theta) * v
        y_dot = ca.sin(theta) * v
        theta_dot = w
        X_dot = ca.vertcat(x_dot, y_dot, theta_dot)

        # cost function
        self.costType = self.config['cost_type']
        print("Cost type: {}".format(self.costType))
        Q = ca.MX.sym('Q', nx, nx)
        R = ca.MX.sym('R', nu, nu)
        Xr = ca.MX.sym('Xr', nx, 1)
        Ur = ca.MX.sym('Ur', nu, 1)
        if self.costType == CostType.POSITION:
            cost_func = 0.5 * (X[:2] - Xr[:2]).T @ Q[:2, :2] @ (X[:2] - Xr[:2]) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        elif self.costType == CostType.POSITION_QUATERNION:
            pos_cost = 0.5 * (X[:2] - Xr[:2]).T @ Q[:2, :2] @ (X[:2] - Xr[:2])
            theta = X[2]
            theta_target = Xr[2]
            so3 = SO3.from_euler(ca.vertcat(0, 0, theta))
            so3_target = SO3.from_euler(ca.vertcat(0, 0, theta_target))
            quat_diff = 1 - ca.power(ca.dot(so3.quat, so3_target.quat), 2)
            quat_cost = 0.5 * quat_diff.T @ Q[2:, 2:] @ quat_diff
            cost_func = pos_cost + quat_cost + 0.5 * (U - Ur).T @ R @ (U - Ur)
        elif self.costType == CostType.POSITION_EULER:
            pos_cost = 0.5 * (X[:2] - Xr[:2]).T @ Q[:2, :2] @ (X[:2] - Xr[:2])
            theta = X[2]
            theta_target = Xr[2]
            euler_diff = 1 - ca.cos(theta - theta_target)
            euler_cost = 0.5 * euler_diff.T @ Q[2:, 2:] @ euler_diff
            cost_func = pos_cost + euler_cost + 0.5 * (U - Ur).T @ R @ (U - Ur)
        elif self.costType == CostType.NAIVE:
            cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        else:
            raise ValueError('[ERROR] in turtlebot._setup_symbolic(), cost_type: {}'.format(self.costType))
        cost = {'cost_func': cost_func, 'vars': {'X': X, 'Xr': Xr, 'U': U, 'Ur': Ur, 'Q': Q, 'R': R}}

        # define dynamics and cost dict
        dynamics = {'dyn_eqn': X_dot, 'vars': {'X': X, 'U': U}}
        params = {
            'X_EQ': np.zeros(self.nState),  # np.atleast_2d(self.X_GOAL)[0, :],
            'U_EQ': np.zeros(self.nControl)  # np.atleast_2d(self.U_GOAL)[0, :],
        }
        self.symbolic = FirstOrderModel(dynamics, cost, self.dt, params)


class NonlinearMPC:
    def __init__(self, ref_traj_config, model_config={}):
        self.controllerType = ControllerType.NMPC
        config = model_config
        # dynamics
        self.model = UnicycleModel(config).symbolic
        self.nState = self.model.nx  # 3 (x, y, theta)
        self.nControl = self.model.nu  # 2 (v, w)
        self.solve_time = 0.0
        self.set_ref_traj(ref_traj_config)
        self.setup_solver()
        self.set_control_bound()
        self.cost_func = self.model.cost_func

    def setup_solver(self, q=[200, 200, 0], R=0.8, N=10):
        self.Q = np.diag(q)
        self.R = R * np.eye(self.model.nu)
        self.N = N

    def set_ref_traj(self, traj_config):
        traj_generator = TrajGenerator(traj_config)
        self.ref_state, self.ref_control, self.dt = traj_generator.get_traj()
        self.nTraj = self.ref_state.shape[1]

    def get_curr_ref(self,t):
        k = round(t / self.dt)
        curr_ref_SE2_coeffs = self.ref_state[:, k]
        curr_ref_twist_coeffs = self.ref_control[:, k]
        return curr_ref_SE2_coeffs, curr_ref_twist_coeffs

    def set_control_bound(self, v_min = -100, v_max= 100, w_min = -100, w_max= 100):
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max

    def solve(self, state, t):
        """
        state: [x, y, theta]
        t: time -> index of reference trajectory (t = k * dt)
        """
        start_time = time.time()
        if self.ref_state is None:
            raise ValueError('Reference trajectory is not set up yet!')

        nu = self.nControl
        nx = self.nState
        k = round(t / self.dt)
        N = self.N
        index_end = min(k + N, self.nTraj - 1)
        X = self.ref_state[:, index_end]
        x_goal = X
        opti = ca.Opti()
        x_var = opti.variable(nx, N + 1)
        u_var = opti.variable(nu, N)

        # initial state constraint
        opti.subject_to(x_var[:, 0] == state)

        # dynamics constraint
        for i in range(N):
            # Euler first order
            x_next = x_var[:, i] + self.dt * self.model.fc_func(x_var[:, i], u_var[:, i])
            opti.subject_to(x_var[:, i + 1] == x_next)

        # cost function
        cost = 0

        for i in range(N):
            index = min(k + i, self.nTraj - 1)
            x_target = self.ref_state[:, index]
            u_target = self.ref_control[:, index]
            # u_target = np.zeros((2, 1))
            cost += self.cost_func(x_var[:, i], x_target, u_var[:, i], u_target, self.Q, self.R)

        cost += self.cost_func(x_var[:, N], x_goal, np.zeros((nu,1)), np.zeros((nu, 1)), 100*self.Q, self.R)
        # control bound
        opti.subject_to(u_var[0, :] >= self.v_min)
        opti.subject_to(u_var[0, :] <= self.v_max)
        opti.subject_to(u_var[1, :] >= self.w_min)
        opti.subject_to(u_var[1, :] <= self.w_max)


        opti.minimize(cost)
        opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        opti.solver('ipopt', opts_setting)
        sol = opti.solve()
        u = sol.value(u_var[:, 0])
        self.solve_time = time.time() - start_time
        return u

    def get_solve_time(self):
        return self.solve_time

    def vel_cmd_to_local_twist(self, vel_cmd):
        return ca.vertcat(vel_cmd[0], 0, vel_cmd[1])

    def local_twist_to_vel_cmd(self, local_vel):
        return ca.vertcat(local_vel[0], local_vel[2])

    @property
    def get_controller_type(self):
        return self.controllerType




if __name__ == '__main__':
    traj_config = {'type': TrajType.CIRCLE,
                     'param': {'start_state': np.array([0, 0, 0]),
                              'dt': 0.2,
                              'linear_vel': 0.1,
                              'angular_vel': 0.05,  # don't change this
                              'nTraj': 650}}
    controller = NonlinearMPC(traj_config)

