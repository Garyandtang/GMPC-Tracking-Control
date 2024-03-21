import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from problems.unicycle_traj_tracking.data.butterfly_tracking.draw import load_data_and_save_figure

def main():
    gmpc_solver_time = np.load('Geomtric_model_predictive_control_store_solve_time.npy')
    nmpc_solver_time = np.load('Nonlinear_model_predictive_control_store_solve_time.npy')
    # save as mat file
    sio.savemat('solve_time.mat', {'gmpc_solver_time': gmpc_solver_time, 'nmpc_solver_time': nmpc_solver_time})
    # calculate mean and std
    gmpc_solver_time_mean = np.mean(gmpc_solver_time)
    gmpc_solver_time_std = np.std(gmpc_solver_time-gmpc_solver_time_mean)
    nmpc_solver_time_mean = np.mean(nmpc_solver_time)
    nmpc_solver_time_std = np.std(nmpc_solver_time-nmpc_solver_time_mean)
    # max and min
    gmpc_solver_time_max = np.max(gmpc_solver_time)
    gmpc_solver_time_min = np.min(gmpc_solver_time)
    nmpc_solver_time_max = np.max(nmpc_solver_time)
    nmpc_solver_time_min = np.min(nmpc_solver_time)
    print('gmpc_solver_time_mean: ', gmpc_solver_time_mean)
    print('gmpc_solver_time_std: ', gmpc_solver_time_std)
    print('nmpc_solver_time_mean: ', nmpc_solver_time_mean)
    print('nmpc_solver_time_std: ', nmpc_solver_time_std)
    print('gmpc_solver_time_max: ', gmpc_solver_time_max)
    print('gmpc_solver_time_min: ', gmpc_solver_time_min)
    print('nmpc_solver_time_max: ', nmpc_solver_time_max)
    print('nmpc_solver_time_min: ', nmpc_solver_time_min)

    plt.figure()
    plt.boxplot([gmpc_solver_time, nmpc_solver_time], labels=['GMPC', 'NMPC'],showfliers=False, patch_artist=True )
    plt.ylabel('Solve time (s)')
    plt.tight_layout()
    plt.savefig('solve_time.jpg')
    plt.show()

if __name__ == '__main__':
    main()
