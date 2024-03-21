"""
Symoblic model of the turtlebot
"""

import numpy as np
import casadi as ca
from utils.enum_class import CostType, DynamicsType
from utils.symbolic_system import FirstOrderModel
from liecasadi import SO3


class TurtlebotModel:
    def __init__(self, config: dict = {}, pyb_freq: int = 50, **kwargs):
        self.nState = 3
        self.nControl = 2
        self.length = 0.23  # length of the turtlebot
        self.width = 0.025  # width of the wheel of the turtlebot

        self.control_freq = pyb_freq
        self.dt = 1. / self.control_freq

        # setup configuration
        self.config = config
        if not config:
            self.config = {'cost_type': CostType.POSITION_EULER, 'dynamics_type': DynamicsType.EULER_FIRST_ORDER}
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
        l = self.length
        # state
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        theta = ca.MX.sym('theta')
        X = ca.vertcat(x, y, theta)
        # control
        v_l = ca.MX.sym('v_l')  # left wheel velocity
        v_r = ca.MX.sym('v_r')  # right wheel velocity
        v = (v_l + v_r) / 2  # linear velocity
        w = (v_r - v_l) / l  # angular velocity
        U = ca.vertcat(v_l, v_r)

        # state derivative
        x_dot = ca.cos(theta) * v
        y_dot = ca.sin(theta) * v
        theta_dot = w
        X_dot = ca.vertcat(x_dot, y_dot, theta_dot)
        # cost = self.set_cost(X, U)

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


if __name__ == '__main__':
    config = {'dynamics_type': DynamicsType.EULER_FIRST_ORDER,
              'cost_type': CostType.POSITION}
    turtlebot_model = TurtlebotModel(config=config)
