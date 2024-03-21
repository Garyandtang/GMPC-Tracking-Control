"""Turtlebot environment using PyBullet physica.


"""
import os

import time
from liecasadi import SO3, SO3Tangent
import casadi as ca
import numpy as np
import pybullet as p
import pybullet_data

from utils.symbolic_system import FirstOrderModel
from environments.wheeled_mobile_robot.wheeled_mobile_robot_base import WheeledMobileRobot


from functools import partial
from utils.enum_class import CostType, DynamicsType

class ScoutMini(WheeledMobileRobot):
    NAME = 'scout_mini'
    URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scout_description', 'urdf', 'scout_mini.urdf')

    def __init__(self,
                 init_state: np.ndarray = None,
                 gui: bool = True,
                 debug: bool = True,
                 **kwargs):
        super().__init__(gui=gui, debug=debug, init_state=init_state, **kwargs)


        # set the init state
        self.nState = 3
        self.nControl = 2
        # turtlebot model parameters
        self.robot_WHEELBASE = 0.55 # *2
        self.robot_WHEEL_RADIUS = 0.175 #/2
        self.robot_HEIGHT = 0.181368485
        self.reset()

    def reset(self, seed=None):
        # reset the simulation
        self._set_action_space()
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.PYB_CLIENT)

        # turtlebot setting
        self.init_pos = np.array([self.init_state[0], self.init_state[1], self.robot_HEIGHT])
        self.init_quat = p.getQuaternionFromEuler([0, 0, self.init_state[2]])
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.PYB_CLIENT)
        self.robot = p.loadURDF(self.URDF_PATH, self.init_pos, self.init_quat,  physicsClientId=self.PYB_CLIENT)
        for i in range(1, 5):
            p.resetJointState(self.robot, i, 0, 0, physicsClientId=self.PYB_CLIENT)
        return self.get_state()




    def twist_to_control(self, twist):
        # vel_cmd: [v, w] in m/s linear and angular velocity
        # action: [v_rear_r, v_front_r, v_rear_l, v_rear_r] in m/s left and right wheel velocity
        # the dir of left and right wheel is opposite
        v = -twist[0]
        w = twist[2]
        v, w = self.saturate_vel_cmd(np.array([v, w]))
        left_side_vel = v - w * self.robot_WHEELBASE / 2
        right_side_vel = v + w * self.robot_WHEELBASE / 2
        left_side_vel = left_side_vel / self.robot_WHEEL_RADIUS
        right_side_vel = right_side_vel / self.robot_WHEEL_RADIUS
        action = np.array([right_side_vel, right_side_vel, -left_side_vel, -left_side_vel])
        return action


    def step(self, action):
        # action: [v_l, v_r] in m/s left and right wheel velocity
        v0 = action[0]
        v1 = action[1]
        v2 = action[2]
        v3 = action[3]
        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.setJointMotorControl2(self.robot, 1, p.VELOCITY_CONTROL, targetVelocity=v0, force=1000,
                                    physicsClientId=self.PYB_CLIENT)
            p.setJointMotorControl2(self.robot, 2, p.VELOCITY_CONTROL, targetVelocity=v1, force=1000,
                                    physicsClientId=self.PYB_CLIENT)
            p.setJointMotorControl2(self.robot, 3, p.VELOCITY_CONTROL, targetVelocity=v2, force=1000,
                                    physicsClientId=self.PYB_CLIENT)
            p.setJointMotorControl2(self.robot, 4, p.VELOCITY_CONTROL, targetVelocity=v3, force=1000,
                                    physicsClientId=self.PYB_CLIENT)
            p.stepSimulation(physicsClientId=self.PYB_CLIENT)

        self.state = self.get_state()
        self.draw_point(self.state)
        return self.state, None, None, None

    def _set_action_space(self):
        self.v_min = -3
        self.v_max = 3
        self.w_min = -5
        self.w_max = 5

    def get_wheel_vel(self):
        pass
        return None




if __name__ == '__main__':
    key_word = {'gui': False}
    env_func = partial(ScoutMini, **key_word)
    turtle_env = ScoutMini(gui=True)
    turtle_env.reset()
    while 1:
        vel_cmd = np.array([1,0])
        action = turtle_env.vel_cmd_to_action(vel_cmd)

        state = turtle_env.step(action)
        # print(p.getJointState())
        # print(p.getJointState(1, 1)[1])
        print(state[0])
        print()
        # time.sleep(0.02)
