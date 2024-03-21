import numpy as np
from manifpy import SE2, SO2, SE2Tangent, SO2Tangent
from environments.numerical_simulator.SE2_simulator import SE2Simulator
class WMRSimulator(SE2Simulator):
    def __init__(self, dt=0.02):
        super().__init__(dt)
        self.length = 0.23
        self.radius = 0.036
        self.nControl = 2

    def set_init_state(self, init_state):
        super().set_init_state(init_state)

    def step(self, control):
        twist = self.control_to_twist(control)
        return super().step(twist), None, None, None, None

    def control_to_twist(self, control):
        r = self.radius
        l = self.length
        w_l = control[0]
        w_r = control[1]
        v = (r * w_l + r * w_r) / 2
        w = (r * w_r - r * w_l) / l
        return np.array([v, 0, w])

    def set_length(self, length):
        self.length = length

    def set_radius(self, radius):
        self.radius = radius

    def get_state(self):
        return self.curr_state

    def twist_to_control(self, twist):
        r = self.radius
        l = self.length
        v = twist[0]
        w = twist[2]
        w_l = (2 * v - l * w) / (2 * r)
        w_r = (2 * v + l * w) / (2 * r)
        return np.array([w_l, w_r])


def test_simulator():
    dt = 0.02
    simulator = WMRSimulator(dt)
    init_state = np.array([0, 0, 0])
    simulator.set_init_state(init_state)
    control = np.array([0, 0.2])
    print(simulator.control_to_twist(control))
    state_container = init_state
    for i in range(20000):
        simulator.step(control)
        # print(simulator.curr_state)
        state_container = np.vstack((state_container, simulator.get_state()))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(state_container[:, 0], state_container[:, 1])
    plt.show()

if __name__ == '__main__':
    test_simulator()