from abc import ABC, abstractmethod
import pybullet as p
import pybullet_data
import numpy as np
import gym


class WheeledMobileRobot(gym.Env, ABC):
    def __init__(self,
                 init_state: np.ndarray = None,
                 gui: bool = True,
                 debug: bool = True):
        self.GUI = gui
        self.DEBUG = debug
        self.CTRL_FREQ = 50  # control frequency
        self.PYB_FREQ = 50  # simulator fr
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ

        # create a PyBullet physics simulation
        self.PYB_CLIENT = -1
        if self.GUI:
            self.PYB_CLIENT = p.connect(p.GUI)
        else:
            self.PYB_CLIENT = p.connect(p.DIRECT)

        # disable urdf auto-loading
        p.setPhysicsEngineParameter(enableFileCaching=1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # set gui and rendering size
        self.RENDER_HEIGHT = int(200)
        self.RENDER_WIDTH = int(320)
        if init_state is None:
            self.init_state = np.array([0, 0, 0])
        elif isinstance(init_state, np.ndarray):
            self.init_state = init_state
        else:
            raise ValueError('[ERROR] in turtlebot.__init__(), init_state, type: {}, size: {}'.format(type(init_state),
                                                                                                      len(init_state)))
        self.v_min, self.v_max, self.w_min, self.w_max = None, None, None, None

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    def get_state(self):
        # [x, y, theta]
        pos, quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.PYB_CLIENT)
        euler = p.getEulerFromQuaternion(quat)  # [-pi, pi]
        self.state = np.array([pos[0], pos[1], euler[2]])
        return self.state

    def get_twist(self):
        # [v, w]
        vel = p.getBaseVelocity(self.robot, physicsClientId=self.PYB_CLIENT)
        self.twist = np.array([np.sqrt(vel[0][0] ** 2 + vel[0][1] ** 2), vel[1][2]])
        return self.twist

    @abstractmethod
    def get_wheel_vel(self):
        raise NotImplementedError

    def draw_point(self, point):
        # if not self.DEBUG or not self.GUI:
        #     return
        p.addUserDebugPoints(
            [[point[0], point[1], 0.12]], [[0.1, 0, 0]], pointSize=3, lifeTime=0.5, physicsClientId=self.PYB_CLIENT)

    def draw_ref_traj(self, ref_SE2):
        # todo: change to draw line, put SE2 outside this class
        # ref_se2: [x, y, cos(theta), sin(theta)]
        if not self.DEBUG or not self.GUI:
            return
        ref_traj = np.zeros((3, ref_SE2.shape[1]))
        ref_traj[0:2, :] = ref_SE2[0:2, :]
        ref_traj[2, :] = 0.1
        for i in range(ref_SE2.shape[1] - 1):
            p1 = ref_traj[:, i]
            p2 = ref_traj[:, i + 1]
            p.addUserDebugLine(p1, p2, [1, 0, 0], 3, physicsClientId=self.PYB_CLIENT)
        return ref_traj

    @abstractmethod
    def _set_action_space(self):
        raise NotImplementedError

    def _actionSpace(self):
        raise NotImplementedError

    def get_vel_cmd_limit(self):
        return self.v_min, self.v_max, self.w_min, self.w_max

    def saturate_vel_cmd(self, vel_cmd):
        v, w = vel_cmd
        if v < self.v_min:
            v = self.v_min
        elif v > self.v_max:
            v = self.v_max
        if w < self.w_min:
            w = self.w_min
        elif w > self.w_max:
            w = self.w_max
        return v, w
