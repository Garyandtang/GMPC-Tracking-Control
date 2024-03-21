import numpy as np
from utils.enum_class import TrajType, ControllerType, EnvType
from controller.nonlinear_mpc import NonlinearMPC
from controller.feedback_linearization import FBLinearizationController
from controller.geometric_mpc import GeometricMPC
from planner.ref_traj_generator import TrajGenerator
from monte_carlo_test_turtlebot import calulate_trajecotry_error, simulation
import os


def butterfly_tracking(env_type, controller_type):
    if env_type == EnvType.TURTLEBOT:
        scale = 0.2
    elif env_type == EnvType.SCOUT_MINI:
        scale = 1.8
    init_state = np.array([0, 0, 0])
    traj_config = {'type': TrajType.EIGHT,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'dt': 0.02,
                             'v_scale': scale,
                             'w_scale': 1,
                             'nTraj': 2500}}

    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()
    if controller_type == ControllerType.NMPC:
        controller = NonlinearMPC(traj_config)
    elif controller_type == ControllerType.GMPC:
        controller = GeometricMPC(traj_config)
    elif controller_type == ControllerType.FEEDBACK_LINEARIZATION:
        controller = FBLinearizationController()
    else:
        return

    store_SE2, store_twist, _ = simulation(init_state, controller, traj_gen, env_type, gui=False)

    dir_name = env_type.value + '_' + controller_type.value
    position_error, orientation_error = calulate_trajecotry_error(ref_state, store_SE2)
    # mkdir if not exist
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data', 'butterfly_tracking', dir_name)):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'data', 'butterfly_tracking', dir_name))

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'butterfly_tracking', dir_name)
    file_path = os.path.join(data_path, 'position_error.npy')
    np.save(file_path, position_error)
    file_path = os.path.join(data_path, 'orientation_error.npy')
    np.save(file_path, orientation_error)
    file_path = os.path.join(data_path, 'ref_SE2.npy')
    np.save(file_path, ref_state)
    file_path = os.path.join(data_path, 'store_SE2.npy')
    np.save(file_path, store_SE2)
    file_path = os.path.join(data_path, 'store_twist.npy')
    np.save(file_path, store_twist)
    file_path = os.path.join(data_path, 'ref_twist.npy')
    np.save(file_path, ref_control)


def main():
    for env_type in EnvType:
        for controller_type in ControllerType:
            print('env_type: {}, controller_type: {}'.format(env_type.value, controller_type.value))
            butterfly_tracking(env_type, controller_type)


def test():
    controller_type = ControllerType.NMPC
    env_type = EnvType.SCOUT_MINI
    butterfly_tracking(env_type, controller_type)

if __name__ == '__main__':
    test()
