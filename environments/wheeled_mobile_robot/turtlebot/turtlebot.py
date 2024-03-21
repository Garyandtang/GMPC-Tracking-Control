"""Turtlebot environment using PyBullet physica.


"""
import os

import time
from liecasadi import SO3, SO3Tangent
import casadi as ca
import numpy as np
import pybullet as p
import pybullet_data
from environments.wheeled_mobile_robot.wheeled_mobile_robot_base import WheeledMobileRobot
from utils.symbolic_system import FirstOrderModel
from gymnasium import spaces
from functools import partial
from utils.enum_class import CostType, DynamicsType

class Turtlebot(WheeledMobileRobot):
    NAME = 'turtlebot'
    URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'turtlebot.urdf')

    def __init__(self,
                 init_state: np.ndarray = None,
                 gui: bool = False,
                 debug: bool = False,
                 **kwargs):
        super().__init__(gui=gui, debug=debug, init_state=init_state, **kwargs)
        # set the init state
        self.nState = 3
        self.nControl = 2
        # turtlebot model parameters
        self.length = 0.23  # length of the turtlebot
        self.radius = 0.036  # width of the wheel of the turtlebot                                                                                            len(init_state)))
        self.reset()
        self.action_space = spaces.Box(low=np.array([-10,-10]), high=np.array([10,10]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.array([-10,-10,-10]), high=np.array([10,10,10]), dtype=np.float32)

    def reset(self, seed=None):
        # reset the simulation
        self._set_action_space()
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.PYB_CLIENT)

        # turtlebot setting
        self.init_pos = np.array([self.init_state[0], self.init_state[1], 0])
        self.init_quat = p.getQuaternionFromEuler([0, 0, self.init_state[2]])
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.PYB_CLIENT)
        self.robot = p.loadURDF(self.URDF_PATH, self.init_pos, self.init_quat, physicsClientId=self.PYB_CLIENT)
        p.resetJointState(self.robot, 0, 0, 0, physicsClientId=self.PYB_CLIENT)
        p.resetJointState(self.robot, 1, 0, 0, physicsClientId=self.PYB_CLIENT)

        return self.get_state(), {}

    def calc_twist(self, action):
        # action: [v_l, v_r] in m/s left and right wheel velocity
        # twist: [v, w] in m/s linear and angular velocity
        v_l = action[0]
        v_r = action[1]
        v = (v_l + v_r) / 2
        w = (v_r - v_l) / self.length
        twist = np.array([v, w])
        return twist

    def twist_to_control(self, twist):
        # twist: [v, 0, w] in m/s linear and angular velocity
        # action: [v_l, v_r] in m/s left and right wheel velocity
        vel_cmd = np.array([twist[0], twist[2]])
        vel_cmd = self.saturate_vel_cmd(vel_cmd)
        v = vel_cmd[0]
        w = vel_cmd[1]
        v_l = v - self.length * w / 2
        v_r = v + self.length * w / 2
        v_l = v_l / self.radius
        v_r = v_r / self.radius
        action = np.array([v_l, v_r])
        return action

    def action_to_vel_cmd(self, action):
        # action: [v_l, v_r] in m/s left and right wheel velocity
        # vel_cmd: [v, w] in m/s linear and angular velocity
        v_l = action[0] * self.radius
        v_r = action[1] * self.radius
        v = (v_l + v_r) / 2
        w = (v_r - v_l) / self.length
        vel_cmd = np.array([v, w])
        return vel_cmd

    def step(self, action):
        # action: [v_l, v_r] in m/s left and right wheel velocity
        v_l = action[0]
        v_r = action[1]
        # TODO: understand how to step per control
        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.setJointMotorControl2(self.robot, 0, p.VELOCITY_CONTROL, targetVelocity=v_l, force=1000,
                                    physicsClientId=self.PYB_CLIENT)
            p.setJointMotorControl2(self.robot, 1, p.VELOCITY_CONTROL, targetVelocity=v_r, force=1000,
                                    physicsClientId=self.PYB_CLIENT)
            p.stepSimulation(physicsClientId=self.PYB_CLIENT)


        self.state = self.get_state()

        self.draw_point(self.state)
        return self.state, 1, None, None, {}

    def get_wheel_vel(self):
        v_l = p.getJointState(self.robot, 0, physicsClientId=self.PYB_CLIENT)[1]
        v_r = p.getJointState(self.robot, 1, physicsClientId=self.PYB_CLIENT)[1]
        return np.array([v_l, v_r])

    def _denormalize_action(self, action):
        """ converts a normalized action into a physical action, only need in RL-based action
        :param action (ndarray):
        :return: action (ndarray):
        """
        return action

    def _set_action_space(self):
        self.v_min = 0
        self.v_max = 0.22
        self.w_min = -2.86
        self.w_max = 2.86




if __name__ == '__main__':
    key_word = {'gui': False}
    env_func = partial(Turtlebot, **key_word)
    turtle_env = Turtlebot(gui=True)
    turtle_env.reset()
    while 1:
        state = turtle_env.step([1, 1])
        print(p.getJointState(1, 1)[1])
        print(state[0])
        print()
        time.sleep(0.02)
