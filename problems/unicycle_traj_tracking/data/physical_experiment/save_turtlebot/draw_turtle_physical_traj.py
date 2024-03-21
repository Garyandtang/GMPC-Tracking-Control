import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio
from controller.ref_traj_generator import TrajGenerator
from utils.enum_class import TrajType
matplotlib.use('TkAgg')
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
# move legend to the right upper corner


font_size = 18  # font size
line_width = 2
plt.figure()
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)


plt.plot(ref_state[0, :], ref_state[1, :],  label='reference trajectory', linewidth=line_width)
plt.plot(gmpc_state_contain[0, :], gmpc_state_contain[1, :], label='simulation trajectory', linewidth=line_width)

plt.xlabel("$x~(m)$",fontsize=font_size)
plt.ylabel("$y~(m)$",fontsize=font_size)
plt.tight_layout()
plt.legend(fontsize=font_size, loc='lower left')

plt.show()
plt.savefig('turtle_gmpc_traj.jpg')
# plot traj
plt.figure()
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)


plt.plot(ref_state[0, :], ref_state[1, :],  label='reference trajectory', linewidth=line_width)
plt.plot(nmpc_state_contain[0, :], nmpc_state_contain[1, :], label='simulation trajectory', linewidth=line_width)
plt.xlabel("$x~(m)$",fontsize=font_size)
plt.ylabel("$y~(m)$",fontsize=font_size)
plt.tight_layout()
plt.legend(fontsize=font_size, loc='lower left')

plt.show()
plt.savefig('turtle_nmpc_traj.jpg')
