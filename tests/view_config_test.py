import os
import pickle
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from feature_extractor.config import *

idx = 113

def main():

    if not os.path.exists(DATA_DIR_PATH / "figure"):
        os.mkdir(DATA_DIR_PATH / "figure")
    if not os.path.exists(DATA_DIR_PATH / "video"):
        os.mkdir(DATA_DIR_PATH / "video")


    fig_path = DATA_DIR_PATH / "figure" / TEST_NAME
    fig_path = fig_path.absolute().as_posix() + "_fig"

    video_path = DATA_DIR_PATH / "video" / TEST_NAME
    video_path = video_path.absolute().as_posix() + "_video.mp4"

    if SAVE_DATA:
        pickle_path = DATA_DIR_PATH / "pickle" / TEST_NAME
        pickle_path = pickle_path.absolute().as_posix() + "_keypoints.pkl"

        file = open(pickle_path, 'rb')
        keypoints_list: list[np.ndarray] = pickle.load(file)
        mask_list: list[np.ndarray] = pickle.load(file)
        keypoints_no_Kalman_list: list[np.ndarray] = pickle.load(file)
        file.close()
    
    plt.ion()
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(1,1,1, projection='3d')

    # TODO: Adjust the scale
    ax1.axes.set_xlim3d(-0.5,1.5)
    ax1.set_xlabel('X-axis')

    ax1.axes.set_ylim3d(0,2) 
    ax1.set_ylabel('Y-axis')

    ax1.axes.set_zlim3d(1,4) 
    ax1.set_zlabel('Depth-axis')

    sc1 = ax1.scatter(0, 0, 0, color='r', label='Kalman')

    ax1.view_init(elev=-30, azim=-60)

    plot_mat_Kalman = keypoints_list[idx].astype(float)
    plot_mat_no_Kalman = keypoints_no_Kalman_list[idx].astype(float)
    mask_mat= mask_list[idx].astype(bool)

    scatter_mat_Kalman = plot_mat_Kalman.reshape(-1,3)              # [HK,3]
    scatter_mat_no_Kalman = plot_mat_no_Kalman.reshape(-1,3)        # [HK,3]
    scatter_mask = mask_mat.reshape(-1)         # [HK]

    scatter_mat_Kalman = scatter_mat_Kalman[scatter_mask]     # [HK-,3]
    scatter_mat_no_Kalman = scatter_mat_no_Kalman[scatter_mask]     # [HK-,3]

    sc1._offsets3d = (-scatter_mat_Kalman[:,0], scatter_mat_Kalman[:,1], scatter_mat_Kalman[:,2])

    plt.draw()
    fig.show()



if __name__ == "__main__":
    main()
    input()

