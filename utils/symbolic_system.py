"""Symbolic Models."""

import casadi as cs


class FirstOrderModel:
    """Implements first-order dynamics model with symbolic variables.

        x_dot = f(x,u), y = h(x,u), with other pre-defined, symbolic functions
        (e.g. cost, constraints), serve as priors for the controllers.

        for second-order system q_dot_dot = f2(q, q_dot, u)
        the state is defined as x = (q, q_dot), which should be handled before calling this class

        Notes:
            * naming convention on symbolic variable and functions.
            * for single-letter symbol, use {}_sym, otherwise use underscore for delimiter.
            * for symbolic functions to be exposed, use {}_func.
    """

    def __init__(self,
                 dynamics: dict,
                 cost,
                 dt=None,
                 params: dict=None):
        # Setup for dynamics
        self.x_sym = dynamics['vars']['X']
        self.u_sym = dynamics['vars']['U']
        self.x_dot = dynamics['dyn_eqn']
        self.y_sym = dynamics.get('obs_eqn', self.x_sym)
        # sample time
        self.dt = dt
        # TODO: why add to self.__dict__
        if params is not None:
            for name, param in params.items():
                assert name not in self.__dict__
                self.__dict__[name] = param
        # variable dims
        self.nx = self.x_sym.shape[0]
        self.nu = self.u_sym.shape[0]
        self.ny = self.y_sym.shape[0]
        # setup model
        self.setup_model()
        # setup Jacobian and Hessian of the dynamics and cost function
        self.setup_linearization()

        # setup cost function
        # Setup cost function.
        self.cost = cost['cost_func']
        # print(self.cost_func)
        self.Q = cost['vars']['Q']
        self.R = cost['vars']['R']
        self.Xr = cost['vars']['Xr']
        self.Ur = cost['vars']['Ur']
        self.cost_func = cs.Function('cost_func', [self.x_sym, self.Xr, self.u_sym, self.Ur, self.Q, self.R], [self.cost], ['x', 'xr', 'u', 'ur', 'Q', 'R'], ['cost'])

    def setup_model(self):
        # continuous time dynamics
        self.fc_func = cs.Function('fc', [self.x_sym, self.u_sym], [self.x_dot], ['x', 'u'], ['f1'])

        # discrete time dynamics
        self.fd_func = None

        # observation function
        self.h_func = cs.Function('h_func', [self.x_sym, self.u_sym], [self.y_sym], ['x', 'u'], ['y'])

    def setup_linearization(self):
        # jacobian w.r.t state
        self.dfdx = cs.jacobian(self.x_dot, self.x_sym)
        self.dfdu = cs.jacobian(self.x_dot, self.u_sym)
        self.df_func = cs.Function('df', [self.x_sym, self.u_sym],
                                   [self.dfdx, self.dfdu], ['x', 'u'], ['dfdx', 'dfdu'])

