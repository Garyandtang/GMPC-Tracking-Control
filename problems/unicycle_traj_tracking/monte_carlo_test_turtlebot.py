from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
from environments.wheeled_mobile_robot.scout.scout_mini import ScoutMini
import numpy as np
from utils.enum_class import TrajType, ControllerType, EnvType
from controller.nonlinear_mpc import NonlinearMPC
from controller.feedback_linearization import FBLinearizationController
from controller.geometric_mpc import GeometricMPC
from planner.ref_traj_generator import TrajGenerator
from manifpy import SE2, SO2
import matplotlib.pyplot as plt
import scipy


def main():
    mc_num = 2
    env_type = EnvType.TURTLEBOT
    # set init state
    # set trajetory
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.15,
                             'angular_vel': 0.2,
                             'nTraj': 1000,
                             'dt': 0.02}}
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()
    # store error
    edmpc_position_error = np.zeros((mc_num, ref_state.shape[1]))
    edmpc_orientation_error = np.zeros((mc_num, ref_state.shape[1]))

    nmpc_position_error = np.zeros((mc_num, ref_state.shape[1]))
    nmpc_orientation_error = np.zeros((mc_num, ref_state.shape[1]))

    fb_position_error = np.zeros((mc_num, ref_state.shape[1]))
    fb_orientation_error = np.zeros((mc_num, ref_state.shape[1]))

    for i in range(mc_num):
        # random init state
        init_x = np.random.uniform(-0.2, 0)
        init_y = np.random.uniform(-0.2, 0.0)
        init_theta = np.random.uniform(-np.pi /6, 0)
        init_state = np.array([init_x, init_y, init_theta])
        print('mc_num: ', i)
        controller = GeometricMPC(traj_config)
        store_state, store_control, _ = simulation(init_state, controller, traj_gen, env_type)
        edmpc_position_error[i, :], edmpc_orientation_error[i, :] = calulate_trajecotry_error(ref_state, store_state)

        controller = NonlinearMPC(traj_config)
        store_state, store_control, _ = simulation(init_state, controller, traj_gen, env_type)
        nmpc_position_error[i, :], nmpc_orientation_error[i, :] = calulate_trajecotry_error(ref_state, store_state)

        controller = FBLinearizationController()
        store_state, store_control, _ = simulation(init_state, controller, traj_gen, env_type)
        fb_position_error[i, :], fb_orientation_error[i, :] = calulate_trajecotry_error(ref_state, store_state)

    # plot
    plt.figure()
    plt.plot(edmpc_position_error.T, label='edmpc')
    plt.title("edmpc position error")
    plt.xlabel("N")
    plt.ylabel("position error")
    plt.show()

    plt.figure()
    plt.plot(edmpc_orientation_error.T, label='edmpc')
    plt.title("edmpc orientation error")
    plt.xlabel("N")
    plt.ylabel("orientation error")
    plt.show()

    plt.figure()
    plt.plot(nmpc_position_error.T, label='nmpc')
    plt.title("nmpc position error")
    plt.xlabel("N")
    plt.ylabel("position error")
    plt.show()

    plt.figure()
    plt.plot(nmpc_orientation_error.T, label='nmpc')
    plt.title("nmpc orientation error")
    plt.xlabel("N")
    plt.ylabel("orientation error")
    plt.show()

    plt.figure()
    plt.plot(fb_position_error.T, label='fb')
    plt.title("fb position error")
    plt.xlabel("N")
    plt.ylabel("position error")
    plt.show()

    plt.figure()
    plt.plot(fb_orientation_error.T, label='fb')
    plt.title("fb orientation error")
    plt.xlabel("N")
    plt.ylabel("orientation error")
    plt.show()

    print('end')
    np.save('data/edmpc_position_error.npy', edmpc_position_error)
    np.save('data/edmpc_orientation_error.npy', edmpc_orientation_error)
    np.save('data/nmpc_position_error.npy', nmpc_position_error)
    np.save('data/nmpc_orientation_error.npy', nmpc_orientation_error)
    np.save('data/fb_position_error.npy', fb_position_error)
    np.save('data/fb_orientation_error.npy', fb_orientation_error)


def calulate_trajecotry_error(ref_SE2, store_SE2):
    position_error = np.linalg.norm(store_SE2[:2, :] - ref_SE2[:2, :], axis=0)
    orientation_error = np.zeros(ref_SE2.shape[1])
    for i in range(ref_SE2.shape[1]):
        ref_angle = ref_SE2[2, i]
        curr_angle = store_SE2[2, i]
        so2_error = SO2(store_SE2[2, i]).between(SO2(ref_angle)).log().coeffs()
        orientation_error[i] = scipy.linalg.norm(so2_error)
    return position_error, orientation_error


def simulation(init_state, controller, traj_gen, env_type, gui=False):
    # set env and traj
    if env_type == EnvType.TURTLEBOT:
        env = Turtlebot(gui=gui, debug=True, init_state=init_state)
    elif env_type == EnvType.SCOUT_MINI:
        env = ScoutMini(gui=gui, debug=True, init_state=init_state)
    else:
        raise NotImplementedError
    v_min, v_max, w_min, w_max = env.get_vel_cmd_limit()
    ref_state, ref_control, dt = traj_gen.get_traj()

    # set controller limits
    controller.set_control_bound(v_min, v_max, w_min, w_max)

    # store simulation traj
    nTraj = ref_state.shape[1]
    store_state = np.zeros((3, nTraj))
    store_control = np.zeros((2, nTraj))
    store_solve_time = np.zeros(nTraj - 1)
    env.draw_ref_traj(ref_state)
    t = 0
    for i in range(nTraj - 1):
        curr_state = env.get_state()
        store_state[:, i] = curr_state
        store_control[:, i] = env.get_twist()
        if controller.controllerType == ControllerType.NMPC:
            vel_cmd = controller.solve(curr_state, t)
        elif controller.controllerType == ControllerType.GMPC:
            vel_cmd = controller.solve(curr_state, t)
        elif controller.controllerType == ControllerType.FEEDBACK_LINEARIZATION:
            curr_ref_state = ref_state[:, i]
            curr_ref_vel_cmd = ref_control[:, i]
            vel_cmd = controller.feedback_control(curr_state, curr_ref_state, curr_ref_vel_cmd)
        store_solve_time[i] = controller.get_solve_time()
        print('curr_state: ', curr_state)
        print('xi: ', vel_cmd)
        print('curr_twist:', env.get_twist())
        print('ref_twist:', ref_control[:, i])
        t += dt
        twist = np.array([vel_cmd[0], 0, vel_cmd[1]])
        env.step(env.twist_to_control(twist))

    store_state[:, -1] = env.get_state()
    store_control[:, -1] = env.get_twist()

    return store_state, store_control, store_solve_time


if __name__ == '__main__':
    main()
