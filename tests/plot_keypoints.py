import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


file = open('/home/xmo/socialnav_xmo/feature_extractor/tests/test_Minimal_false.pkl', 'rb')
keypoints_list = pickle.load(file)
mask_list = pickle.load(file)
keypoints_no_Kalman_list = pickle.load(file)

file.close()
freq = 15

# print(len(keypoints_list))
# print(len(mask_list))

fig_init = False

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.axes.set_xlim3d(-0.5e6,2e6)
ax.set_xlabel('X-axis')

ax.axes.set_ylim3d(-5,0) 
ax.set_ylabel('Depth-axis')

ax.axes.set_zlim3d(-3e6,0e6) 
ax.set_zlabel('Y-axis')

sc1 = ax.scatter(2e6, 0, 0, color='r')
sc2 = ax.scatter(2e6, 0, 0, color='b')

fig.show()
count = 0
for data, data_no_Kalman, mask in zip(keypoints_list, keypoints_no_Kalman_list, mask_list):

    if data is None or mask is None:
        # sc1._offsets3d = None
        # sc2._offsets3d = None
        # plt.draw()
        # plt.pause(1/freq)

        # print("no detection")
        continue

    fig.savefig(f"/home/xmo/socialnav_xmo/feature_extractor/img/minimal_false/minimal_false{count}")
    count += 1
    plot_mat = data.astype(float)
    plot_mat2 = data_no_Kalman.astype(float)
    
    mask_mat= mask.astype(bool)

    scatter_mat = plot_mat.reshape(-1,3)        # [HK,3]
    scatter_mat2 = plot_mat2.reshape(-1,3)        # [HK,3]

    scatter_mask = mask_mat.reshape(-1)         # [HK]
    # print(scatter_mask)
    # print(type(scatter_mask[0]))

    scatter_mat = scatter_mat[scatter_mask]     # [HK-,3]
    scatter_mat2 = scatter_mat2[scatter_mask]     # [HK-,3]

    sc1._offsets3d = (scatter_mat[:,0], -scatter_mat[:,2], -scatter_mat[:,1])
    sc2._offsets3d = (scatter_mat2[:,0], -scatter_mat2[:,2], -scatter_mat2[:,1])

    plt.draw()
    plt.pause(1/freq)

input()