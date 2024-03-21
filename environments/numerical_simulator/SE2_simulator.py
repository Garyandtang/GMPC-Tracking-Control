import numpy as np
from manifpy import SE2, SO2, SE2Tangent, SO2Tangent
class SE2Simulator:
    def __init__(self, dt=0.02):
        self.dt = dt
        self.nTwist = 3
        self.nState = 3
        self.curr_state = np.array([0, 0, 0]) # [x, y, theta]

    def set_init_state(self, init_state):
        self.curr_state = init_state

    def step(self, twist):
        curr_SE2 = SE2(self.curr_state[0], self.curr_state[1], self.curr_state[2])
        next_SE2 = curr_SE2 * SE2Tangent(twist * self.dt).exp()
        self.curr_state = np.array([next_SE2.x(), next_SE2.y(), next_SE2.angle()])
        return self.curr_state

