import numpy as np
from utils.enum_class import TrajType, ControllerType, EnvType
from controller.naive_mpc import NaiveMPC
from controller.feedback_linearization import FBLinearizationController
from controller.error_dynamics_mpc import ErrorDynamicsMPC
from controller.ref_traj_generator import TrajGenerator
from monte_carlo_test_turtlebot import simulation, calulate_trajecotry_error
from matplotlib import pyplot as plt

def main():
    init_state = np.array([-0.1, -0.1, 0])
    controller_type = ControllerType.GMPC
    env_type = EnvType.TURTLEBOT
    # set solver
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.3,
                             'angular_vel': 0.3,
                             'nTraj': 800,
                             'dt': 0.02}}

    # figure of eight
    # traj_config = {'type': TrajType.EIGHT,
    #           'param': {'start_state': np.array([0, 0, 0]),
    #                     'dt': 0.02,
    #                     'nTraj': 1700}}

    if controller_type == ControllerType.NMPC:
        controller = NaiveMPC(traj_config)
    elif controller_type == ControllerType.GMPC:
        controller = ErrorDynamicsMPC(traj_config)
    elif controller_type == ControllerType.FEEDBACK_LINEARIZATION:
        controller = FBLinearizationController()

    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()

    store_state, store_control, store_solve_time = simulation(init_state, controller, traj_gen, env_type, gui=True)

    # plot linear velocity
    plt.figure()
    plt.plot(store_control[0, :], label='linear velocity')
    plt.plot(store_control[1, :], label='angular velocity')
    plt.legend()
    plt.show()

    # plot trajectory
    plt.figure()
    plt.plot(store_state[0, :], store_state[1, :], label='trajectory')
    plt.plot(ref_state[0, :], ref_state[1, :], label='reference trajectory')
    plt.legend()
    plt.show()

    # plot error
    position_error, orientation_error = calulate_trajecotry_error(store_state, ref_state)

    plt.figure()
    plt.plot(position_error, label='position error')
    plt.show()
    plt.figure()
    plt.plot(orientation_error, label='orientation error')
    plt.show()




if __name__ == '__main__':
    main()