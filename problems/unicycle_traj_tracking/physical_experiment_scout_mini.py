import sys
sys.path.append("../..")
from environments.wheeled_mobile_robot.scout.scout_mini import ScoutMini
import numpy as np
from utils.enum_class import TrajType, ControllerType
from controller.nonlinear_mpc import NonlinearMPC
from controller.geometric_mpc import GeometricMPC
from planner.ref_traj_generator import TrajGenerator
from manifpy import SE2
import rospy
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

state_init = False
x_bias = 0.0
y_bias = 0.0
theta_bias = 0.0
init_state = np.array([0.0, 0.0, 0.0])


def process_vicon_data(data):
    global state_init, x_bias, y_bias, theta_bias, init_state
    if not state_init:
        state_init = True
        x_bias = int(data.pose.pose.position.x * 10000) / 10000.0
        y_bias = int(data.pose.pose.position.y * 10000) / 10000.0
        theta_bias = math.atan2(2 * (
                    data.pose.pose.orientation.w * data.pose.pose.orientation.z + data.pose.pose.orientation.x * data.pose.pose.orientation.y),
                                1 - 2 * (
                                            data.pose.pose.orientation.y * data.pose.pose.orientation.y + data.pose.pose.orientation.z * data.pose.pose.orientation.z))
    init_state[0] = int(data.pose.pose.position.x * 10000) / 10000.0 - x_bias
    init_state[1] = int(data.pose.pose.position.y * 10000) / 10000.0 - y_bias
    init_state[2] = math.atan2(2 * (
                data.pose.pose.orientation.w * data.pose.pose.orientation.z + data.pose.pose.orientation.x * data.pose.pose.orientation.y),
                               1 - 2 * (
                                           data.pose.pose.orientation.y * data.pose.pose.orientation.y + data.pose.pose.orientation.z * data.pose.pose.orientation.z)) - theta_bias
    print(init_state)


def main():
    # time.sleep(1)
    global init_state
    # init node
    rospy.init_node('physical_exp')
    rospy.Subscriber("/pos_vel_mocap/odom_TA", Odometry, process_vicon_data)
    cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    odom_pub = rospy.Publisher("vel_odom", Odometry, queue_size=10)
    rate = rospy.Rate(5)
    mpc_cmd = Twist()
    vehicle_odom = Odometry()

    # rospy.spin()

    # set init state (real experiment should get from motion capture system)
    # init_state = np.array([0, 0, 0])        # change to global variable in exp, comment this line during exp
    # set ref trajectory
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'dt': 0.2,
                             'linear_vel': 0.1,
                             'angular_vel': 0.05,  # don't change this
                             'nTraj': 650}}
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()
    # set environment
    env = ScoutMini(gui=False, debug=True, init_state=init_state)
    env.draw_ref_traj(ref_state)

    # set controller
    ctrl_type = ControllerType.NMPC
    if ctrl_type == ControllerType.NMPC:
        controller = NonlinearMPC(traj_config)
    elif ctrl_type == ControllerType.GMPC:
        controller = GeometricMPC(traj_config)
    controller.set_control_bound()
    t = 0
    store_solve_time = np.zeros(ref_state.shape[1] - 1)
    for i in range(ref_state.shape[1] - 1):
        # in physical experiment replace this with result from motion capture system
        # curr_state = env.get_state()
        curr_state = init_state.copy()

        if controller.controllerType == ControllerType.NMPC:
            vel_cmd = controller.solve(curr_state, t)
        elif controller.controllerType == ControllerType.GMPC:
            vel_cmd = controller.solve(curr_state, t)
        store_solve_time[i] = controller.get_solve_time()

        print('curr_state: ', curr_state)
        print('xi: ', vel_cmd)
        print('curr_twist:', env.get_twist())

        t += dt
        # in physical experiment replace this with sending velocity command to robot and sleep for dt
        env.step(env.vel_cmd_to_action(vel_cmd))
        mpc_cmd.linear.x = vel_cmd[0]
        mpc_cmd.angular.z = vel_cmd[1]
        cmd_pub.publish(mpc_cmd)
        vehicle_odom.header.stamp = rospy.Time.now()
        vehicle_odom.header.frame_id = "world"
        vehicle_odom.pose.pose.position.x = curr_state[0]
        vehicle_odom.pose.pose.position.y = curr_state[1]
        vehicle_odom.pose.pose.position.z = curr_state[2]

        vehicle_odom.twist.twist.linear.x = vel_cmd[0]
        vehicle_odom.twist.twist.linear.y = store_solve_time[i]
        vehicle_odom.twist.twist.angular.z = vel_cmd[0]
        odom_pub.publish(vehicle_odom)

        # set command
        # time.sleep(0.02)
        rate.sleep()

    np.save(ctrl_type.value + '_' + 'store_solve_time.npy', store_solve_time)


if __name__ == '__main__':
    main()
