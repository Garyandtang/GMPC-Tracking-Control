import numpy as np
import matplotlib.pyplot as plt


edmpc_position_error = np.load("edmpc_position_error.npy")
nmpc_position_error = np.load("nmpc_position_error.npy")
fb_position_error = np.load("fb_position_error.npy")

edmpc_orientation_error = np.load("edmpc_orientation_error.npy")
nmpc_orientation_error = np.load("nmpc_orientation_error.npy")
fb_orientation_error = np.load("fb_orientation_error.npy")

# plot
end = 1000
t = np.arange(0, edmpc_position_error.shape[1] * 0.02, 0.02)
font_size = 16
line_width = 2
plt.figure()
plt.plot(t[:end], edmpc_position_error.T[:end, :], label='edmpc', linewidth=line_width)

plt.ylim(0, 0.15)
 
plt.grid()
plt.xlabel("$t~(s)$",fontsize=font_size+2)
plt.ylabel("$e_p~(m)$",fontsize=font_size+2)
plt.savefig("edmpc_position_error.jpg")
plt.show()

plt.figure()
plt.plot(t[:end], edmpc_orientation_error.T[:end, :], label='edmpc')

plt.ylim(0, 2.0)
plt.grid()
 
plt.xlabel("$t~(s)$",fontsize=font_size+2)
plt.ylabel("$e_R~(rad)$",fontsize=font_size+2)
plt.savefig("edmpc_orientation_error.jpg")
plt.show()

plt.figure()
plt.plot(t[:end], nmpc_position_error.T[:end, :], label='nmpc')
plt.ylim(0, 0.15)
plt.grid()
 

plt.xlabel("$t~(s)$",fontsize=font_size)
plt.ylabel("$e_p~(m)$",fontsize=font_size)
plt.savefig("nmpc_position_error.jpg")
plt.show()

plt.figure()
plt.plot(t[:end], nmpc_orientation_error.T[:end, :], label='nmpc')
plt.ylim(0, 2.0)
plt.grid()
 
plt.xlabel("$t~(s)$",fontsize=font_size)
plt.ylabel("$e_R~(rad)$",fontsize=font_size)
plt.savefig("nmpc_orientation_error.jpg")
plt.show()

plt.figure()
plt.plot(t[:end], fb_position_error.T[:end, :], label='fb')
plt.ylim(0, 0.15)
plt.grid()
 
plt.xlabel("$t~(s)$",fontsize=font_size)
plt.ylabel("$e_p~(m)$", fontsize=font_size)
plt.savefig("fb_position_error.jpg")
plt.show()

plt.figure()
plt.plot(t[:end], fb_orientation_error.T[:end, :], label='fb')

plt.ylim(0, 2.0)
plt.grid()
 
# use latex
plt.xlabel("$t~(s)$",fontsize=font_size)
plt.ylabel("$e_R~(rad)$",fontsize=font_size)
plt.savefig("fb_orientation_error.jpg")
plt.show()

