import pickle
import numpy as np
from matplotlib import pyplot as plt
import time

from feature_extractor.config import *

fig_path = DATA_DIR_PATH / "figure" / TEST_NAME
fig_path = fig_path.absolute().as_posix() + "_fig"

video_path = DATA_DIR_PATH / "video" / TEST_NAME
video_path = video_path.absolute().as_posix() + "_video.mp4"

if SAVE_DATA:
    pickle_path = DATA_DIR_PATH / "pickle" / TEST_NAME
    pickle_path = pickle_path.absolute().as_posix() + "_keypoints.pkl"

    file = open(pickle_path, 'rb')
    keypoints_list = pickle.load(file)
    mask_list = pickle.load(file)
    keypoints_no_Kalman_list = pickle.load(file)
    file.close()

if SAVE_YOLO_IMG:
    pickle_path = DATA_DIR_PATH / "pickle" / TEST_NAME
    pickle_path = pickle_path.absolute().as_posix() + "_YOLOimgs.pkl"

    file = open(pickle_path, 'rb')
    YOLO_plot_list = pickle.load(file)
    file.close()


plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax2 = fig.add_subplot(1,2,2)

ax1.axes.set_xlim3d(-0.5e6,2e6)
ax1.set_xlabel('X-axis')

ax1.axes.set_ylim3d(-5,0) 
ax1.set_ylabel('Depth-axis')

ax1.axes.set_zlim3d(-3e6,0e6) 
ax1.set_zlabel('Y-axis')

sc1 = ax1.scatter(2e6, 0, 0, color='r', label='Kalman')
sc2 = ax1.scatter(2e6, 0, 0, color='b', label='no Kalman')

ax1.legend()

fig.show()
count = 0
# for data_Kalman, data_no_Kalman, mask, plot_2d in zip(keypoints_list, keypoints_no_Kalman_list, mask_list, YOLO_plot_list):

#     if data_Kalman is None or mask is None or plot_2d is None:
#         print(f"data:{data_Kalman}, mask:{mask}, Img2d:{plot_2d}, count:{count}")
#         count += 1
#         continue

#     # fig.savefig(f"/home/xmo/socialnav_xmo/feature_extractor/img/minimal_false/minimal_false{count}")

#     plot_mat_Kalman = data_Kalman.astype(float)
#     plot_mat_no_Kalman = data_no_Kalman.astype(float)
#     mask_mat= mask.astype(bool)

#     scatter_mat_Kalman = plot_mat_Kalman.reshape(-1,3)              # [HK,3]
#     scatter_mat_no_Kalman = plot_mat_no_Kalman.reshape(-1,3)        # [HK,3]

#     scatter_mask = mask_mat.reshape(-1)         # [HK]
#     # print(scatter_mask)
#     # print(type(scatter_mask[0]))

#     scatter_mat_Kalman = scatter_mat_Kalman[scatter_mask]     # [HK-,3]
#     scatter_mat_no_Kalman = scatter_mat_no_Kalman[scatter_mask]     # [HK-,3]

#     sc1._offsets3d = (scatter_mat_Kalman[:,0], -scatter_mat_Kalman[:,2], -scatter_mat_Kalman[:,1])
#     sc2._offsets3d = (scatter_mat_no_Kalman[:,0], -scatter_mat_no_Kalman[:,2], -scatter_mat_no_Kalman[:,1])

#     plt.draw()
#     plt.pause(1/PUB_FREQ)

#     count += 1


# input()