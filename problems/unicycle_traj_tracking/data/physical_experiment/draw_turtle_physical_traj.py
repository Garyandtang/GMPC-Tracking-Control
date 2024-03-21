import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from controller.ref_traj_generator import TrajGenerator
from utils.enum_class import TrajType
traj_config = {'type': TrajType.EIGHT,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'dt': 0.02,
                             'v_scale': 0.2,
                             'nTraj': 2500}}
traj_gen = TrajGenerator(traj_config)
ref_state, ref_control, dt = traj_gen.get_traj()
gmpc_state_contain = np.load('Geomtric_model_predictive_control_store_state.npy')
nmpc_state_contain = np.load('nonlinear_model_predictive_control_store_state.npy')

# plot traj
plt.figure()
plt.plot(gmpc_state_contain[0, :], gmpc_state_contain[1, :], label='gmpc')
plt.plot(ref_state[0, :], ref_state[1, :], label='ref')
# plt.plot(nmpc_state_contain[0, :], nmpc_state_contain[1, :], label='nmpc')
plt.title("trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()