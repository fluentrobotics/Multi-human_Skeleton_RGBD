import os
import pickle
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import time
from tqdm import tqdm

from feature_extractor.config import *




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

    if SAVE_YOLO_IMG:
        pickle_path = DATA_DIR_PATH / "pickle" / TEST_NAME
        pickle_path = pickle_path.absolute().as_posix() + "_YOLOimgs.pkl"

        file = open(pickle_path, 'rb')
        YOLO_plot_list: list[np.ndarray] = pickle.load(file)
        file.close()


    plt.ion()
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,2,1, projection='3d')
    ax2 = fig.add_subplot(2,2,2, projection='3d')
    ax3 = fig.add_subplot(2,2,3, projection='3d')
    ax4 = fig.add_subplot(2,2,4, projection='3d')
    # ax0 = fig.add_subplot(2,3,1)

    # TODO: Adjust the scale
    ax1.axes.set_xlim3d(-0.5,1.5)
    ax1.set_xlabel('X-axis')
    ax1.axes.set_ylim3d(0,2) 
    ax1.set_ylabel('Y-axis')
    ax1.axes.set_zlim3d(1,4) 
    ax1.set_zlabel('Depth-axis')

    # ax2.axes.set_xlim3d(-2,2)
    # ax2.set_xlabel('X-axis')
    # ax2.axes.set_ylim3d(-2,2) 
    # ax2.set_ylabel('Y-axis')
    # ax2.axes.set_zlim3d(0,4) 
    # ax2.set_zlabel('Depth-axis')

    ax2.axes.set_xlim3d(-0.5,1.5)
    ax2.set_xlabel('X-axis')
    ax2.axes.set_ylim3d(0,2) 
    ax2.set_ylabel('Y-axis')
    ax2.axes.set_zlim3d(1,4) 
    ax2.set_zlabel('Depth-axis')

    ax3.axes.set_xlim3d(-0.5,1.5)
    ax3.set_xlabel('X-axis')
    ax3.axes.set_ylim3d(0,2) 
    ax3.set_ylabel('Y-axis')
    ax3.axes.set_zlim3d(1,4) 
    ax3.set_zlabel('Depth-axis')

    ax4.axes.set_xlim3d(-0.5,1.5)
    ax4.set_xlabel('X-axis')
    ax4.axes.set_ylim3d(0,2) 
    ax4.set_ylabel('Y-axis')
    ax4.axes.set_zlim3d(1,4) 
    ax4.set_zlabel('Depth-axis')

    sc1 = ax1.scatter(0, 0, 0, color='r', label='Kalman')
    sc2 = ax2.scatter(0, 0, 0, color='b', label='no Kalman')
    sc3 = ax3.scatter(0, 0, 0, color='r', label='Kalman')
    sc4 = ax4.scatter(0, 0, 0, color='b', label='no Kalman')

    # view position
    ax1.view_init(elev=-60, azim=-90)
    ax1.legend()
    ax2.view_init(elev=-60, azim=-90)
    ax2.legend()
    ax3.view_init(elev=-15, azim=-90)
    ax3.legend()
    ax4.view_init(elev=-15, azim=-90)
    ax4.legend()


    # fig.show()
    count = 0

    for data_Kalman, data_no_Kalman, mask, plot_2d in tqdm(zip(keypoints_list, keypoints_no_Kalman_list, mask_list, YOLO_plot_list),
                                                        ncols=80,
                                                        colour="red",
                                                        total=len(mask_list)
                                                        ):

        if data_Kalman is None and plot_2d is None:
            tqdm.write(f"no keypoints detection at {count}")
            # print(f"no keypoints detection at {count}")
            count += 1
            continue
        
            
        # if plot_2d is not None:
        #     ax0.set_title(f"{count}: YOLO result")
        #     ax0.imshow(plot_2d[:,:,::-1])
        
        if data_Kalman is not None and mask is not None:
            plot_mat_Kalman = data_Kalman.astype(float)
            plot_mat_no_Kalman = data_no_Kalman.astype(float)
            mask_mat= mask.astype(bool)

            scatter_mat_Kalman = plot_mat_Kalman.reshape(-1,3)              # [HK,3]
            scatter_mat_no_Kalman = plot_mat_no_Kalman.reshape(-1,3)        # [HK,3]

            scatter_mask = mask_mat.reshape(-1)         # [HK]
            # print(scatter_mask)
            # print(type(scatter_mask[0]))

            scatter_mat_Kalman = scatter_mat_Kalman[scatter_mask]     # [HK-,3]
            scatter_mat_no_Kalman = scatter_mat_no_Kalman[scatter_mask]     # [HK-,3]

            sc1._offsets3d = (-scatter_mat_Kalman[:,0], scatter_mat_Kalman[:,1], scatter_mat_Kalman[:,2])
            sc2._offsets3d = (-scatter_mat_no_Kalman[:,0], scatter_mat_no_Kalman[:,1], scatter_mat_no_Kalman[:,2])
            sc3._offsets3d = (-scatter_mat_Kalman[:,0], scatter_mat_Kalman[:,1], scatter_mat_Kalman[:,2])
            sc4._offsets3d = (-scatter_mat_no_Kalman[:,0], scatter_mat_no_Kalman[:,1], scatter_mat_no_Kalman[:,2])

            # ax1.set_title(f"{count}: 3D keypoints")
            # ax2.set_title(f"{count}: 3D keypoints")
            plt.draw()
            # plt.pause(1/PUB_FREQ)
            
        
        # input()
        fig.suptitle(f'Fig {count}\nMinimal Filter:{MINIMAL_FILTER}')
        fig.savefig(f"{fig_path}{count}.png", format='png')
        # tqdm.write(f"plot fig{count}")
        # print(f"plot fig{count}")
        count += 1
        

if __name__ == "__main__":

    main()