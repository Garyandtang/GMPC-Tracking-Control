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
import pybullet as p
import os
state_init = False
x_bias = 0.0
y_bias = 0.0
theta_bias = 0.0
init_state = np.array([0.0, 0.0, 0.0])


def odom_callback(msg):

    global init_state, is_init
    position = msg.pose.pose.position
    x = position.x
    y = position.y
    z = position.z
    pos = np.array([x, y, z])

    orientation = msg.pose.pose.orientation
    qx = orientation.x
    qy = orientation.y
    qz = orientation.z
    qw = orientation.w
    quat = np.array([qx, qy, qz, qw])
    euler = p.getEulerFromQuaternion(quat)  # [-pi, pi]

    init_state[0] = x
    init_state[1] = y
    init_state[2] = euler[2]

    # Process the data as needed
    # ...
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
    # print(init_state)


def main():
    # time.sleep(1)
    global init_state
    # init node
    rospy.init_node('physical_exp')
    cmd_pub = rospy.Publisher("/robot2/cmd_vel", Twist, queue_size=1)
    # data_pub = rospy.Publisher("vel_odom", Odometry, queue_size=10)
    rospy.Subscriber("/robot2/odom", Odometry, odom_callback)
    freq = 50.
    rate = rospy.Rate(freq)
    mpc_cmd = Twist()
    vehicle_odom = Odometry()
    # container to store the state



    # set ref trajectory
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.02,
                             'angular_vel': 0.05,
                             'nTraj': 1000,
                             'dt': 1./freq}}
    traj_config = {'type': TrajType.EIGHT,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'dt': 0.02,
                             'v_scale': 0.2,
                             'nTraj': 2500}}
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()
    state_container = np.zeros((3, ref_state.shape[1] - 1))
    # set controller
    ctrl_type = ControllerType.GMPC
    if ctrl_type == ControllerType.NMPC:
        controller = NonlinearMPC(traj_config)
    elif ctrl_type == ControllerType.GMPC:
        controller = GeometricMPC(traj_config)
    controller.set_control_bound(0, 0.06, -0.8, 0.8)
    t = 0
    store_solve_time = np.zeros(ref_state.shape[1] - 1)

    for i in range(ref_state.shape[1] - 1):
        curr_state = init_state.copy()

        state_container[:, i] = curr_state
        if controller.controllerType == ControllerType.NMPC:
            vel_cmd = controller.solve(curr_state, t)
        elif controller.controllerType == ControllerType.GMPC:
            vel_cmd = controller.solve(curr_state, t)
        store_solve_time[i] = controller.get_solve_time()


        t += dt
        mpc_cmd.linear.x = vel_cmd[0]
        mpc_cmd.angular.z = vel_cmd[1]
        cmd_pub.publish(mpc_cmd)
        rate.sleep()

    # save to data/physical_experiorment
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data', 'physical_experiment')):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'data', 'physical_experiment'))
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'physical_experiment')
    file_path = os.path.join(data_path, ctrl_type.value + '_' + 'store_state.npy')
    np.save(file_path, state_container)


if __name__ == '__main__':
    main()
