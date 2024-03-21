import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # traverse all the directories
    root_dir = os.path.join(os.getcwd())
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'orientation_error.npy' not in filenames:
            continue
        print(dirpath)
        load_data_and_save_figure(dirpath)




# load data and save the figure
def load_data_and_save_figure(data_path):

    # get current directory name
    name = os.path.basename(data_path)
    orientation_error_path = os.path.join(data_path, 'orientation_error.npy')
    position_error_path = os.path.join(data_path, 'position_error.npy')
    orientation_error = np.load(orientation_error_path)
    position_error = np.load(position_error_path)

    ref_SE2_path = os.path.join(data_path, 'ref_SE2.npy')
    store_SE2_path = os.path.join(data_path, 'store_SE2.npy')
    ref_SE2 = np.load(ref_SE2_path)
    store_SE2 = np.load(store_SE2_path)

    ref_twist_path = os.path.join(data_path, 'ref_twist.npy')
    store_twist_path = os.path.join(data_path, 'store_twist.npy')
    ref_twist = np.load(ref_twist_path)
    store_twist = np.load(store_twist_path)

    t = np.arange(0, ref_SE2.shape[1] * 0.02, 0.02)
    # plot
    plt.figure()
    font_size = 18
    line_width = 2
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    pos_lim = 0.4
    ori_lim = 1.4
    plt.plot(t, orientation_error.T,label='orientation error', linewidth=line_width)
    plt.xlabel("$t~(s)$",fontsize=font_size)
    plt.ylabel("$e_R~(rad)$",fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.ylim(0, ori_lim)
    plt.savefig(os.path.join(data_path, name + '_orientation_error.jpg'))
    plt.show()

    plt.figure()
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.plot(t, position_error.T,label='position error', linewidth=line_width)
    plt.xlabel("$t~(s)$",fontsize=font_size)
    plt.ylabel("$e_p~(m)$",fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.ylim(0, pos_lim)
    plt.savefig(os.path.join(data_path, name + '_position_error.jpg'))
    plt.show()

    plt.figure()
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], label='reference trajectory', linewidth=line_width)
    plt.plot(store_SE2[0, :], store_SE2[1, :], label='actual trajectory', linewidth=line_width)
    plt.xlabel("$x~(m)$",fontsize=font_size)
    plt.ylabel("$y~(m)$",fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, name + '_trajectory.jpg'))
    plt.show()

    plt.figure()
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.plot(t, ref_twist[0, :], label='reference linear velocity', linewidth=line_width)
    plt.plot(t, store_twist[0, :], label='actual linear velocity', linewidth=line_width)
    plt.xlabel("$t~(s)$",fontsize=font_size)
    plt.ylabel("$v~(m/s)$",fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, name + '_linear_velocity.jpg'))
    plt.show()

    plt.figure()
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    if store_twist.shape[0] == 2:
        plt.plot(t, ref_twist[1, :], label='reference angular velocity', linewidth=line_width)
        plt.plot(t, store_twist[1, :], label='actual angular velocity', linewidth=line_width)
        plt.xlabel("$t~(s)$", fontsize=font_size)
        plt.ylabel("$w~(rad/s)$", fontsize=font_size)
        plt.legend(fontsize=font_size)
        plt.tight_layout()
        plt.savefig(os.path.join(data_path, name + '_angular_velocity.jpg'))
        plt.show()
    elif store_twist.shape[0] == 3:
        plt.plot(t, ref_twist[2, :], label='reference angular velocity', linewidth=line_width)
        plt.plot(t, store_twist[2, :],  label='actual angular velocity', linewidth=line_width)
        plt.xlabel("$t~(s)$",fontsize=font_size)
        plt.ylabel("$w~(rad/s)$",fontsize=font_size)
        plt.legend(fontsize=font_size)
        plt.tight_layout()
        plt.savefig(os.path.join(data_path, name + '_angular_velocity.jpg'))
        plt.show()


if __name__ == '__main__':
    main()