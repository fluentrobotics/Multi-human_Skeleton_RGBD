import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

T = 10
fig_init = False

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for _ in range(T):
    data = np.random.random((2,17,3))
    mask = np.random.choice(a=[False, True], size=(2, 17), p=[0.5, 0.5])

    plot_mat = data
    mask_mat= mask

    scatter_mat = plot_mat.reshape(-1,3)        # [HK,3]
    scatter_mask = mask_mat.reshape(-1)         # [HK]

    scatter_mat = scatter_mat[scatter_mask]     # [HK-,3]

    if not fig_init:
        sc = ax.scatter(scatter_mat[:, 0], scatter_mat[:, 1], scatter_mat[:, 2])
        fig.show()
        fig_init = True

    else:
        sc._offsets3d = (scatter_mat[:,0], scatter_mat[:,1], scatter_mat[:,2])
        plt.draw()

    plt.pause(0.5)