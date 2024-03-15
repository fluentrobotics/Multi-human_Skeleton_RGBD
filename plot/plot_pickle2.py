import os
import pickle
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import time
from tqdm import tqdm

from skeleton_extractor.config import *
from skeleton_extractor.utils import delete_files_in_directory

# Local Config
XLIM3D = (-0.5,3)
YLIM3D = (-0.5,3)
ZLIM3D = (1,4)


def main():

    if not os.path.exists(DATA_DIR_PATH / "figure"):
        os.mkdir(DATA_DIR_PATH / "figure")
    if not os.path.exists(DATA_DIR_PATH / "video"):
        os.mkdir(DATA_DIR_PATH / "video")
    
    # delete last plotting result
    delete_files_in_directory( (DATA_DIR_PATH / "figure").absolute().as_posix() )


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
    plt.tight_layout()
    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[4, 3])

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')  # First row, first column
    ax2 = fig.add_subplot(gs[1, 0], projection='3d')  # Second row, first column
    ax3 = fig.add_subplot(gs[:, 1])  # Both rows, second column

    # TODO: Adjust the scale
    ax1.axes.set_xlim3d(*XLIM3D)
    ax1.set_xlabel('X-axis')
    ax1.axes.set_ylim3d(*YLIM3D) 
    ax1.set_ylabel('Y-axis')
    ax1.axes.set_zlim3d(*ZLIM3D) 
    ax1.set_zlabel('Depth-axis')

    ax2.axes.set_xlim3d(*XLIM3D)
    ax2.set_xlabel('X-axis')
    ax2.axes.set_ylim3d(*YLIM3D) 
    ax2.set_ylabel('Y-axis')
    ax2.axes.set_zlim3d(*ZLIM3D) 
    ax2.set_zlabel('Depth-axis')

    sc1 = ax1.scatter(0, 0, 0, color='r', label='Human Keypoints')
    sc2 = ax2.scatter(0, 0, 0, color='r', label='Human Keypoints')

    # view position
    ax1.view_init(elev=-60, azim=-90)
    ax1.set_title('Front View', fontstyle='italic')
    ax1.legend()

    ax2.view_init(elev=-15, azim=-90)
    ax2.set_title('Top View', fontstyle='italic')
    ax2.legend()


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
        
            
        if plot_2d is not None:
            ax3.set_title(f"{count}: YOLO result")
            ax3.imshow(plot_2d[:,:,::-1])
        
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
            sc2._offsets3d = (-scatter_mat_Kalman[:,0], scatter_mat_Kalman[:,1], scatter_mat_Kalman[:,2])

            ax1.set_title(f"{count}: 3D keypoints")
            ax2.set_title(f"{count}: 3D keypoints")
            plt.draw()
            # plt.pause(1/PUB_FREQ)
            
        
        # input()
        fig.suptitle(f'Fig {count}')
        fig.savefig(f"{fig_path}{count}.png", format='png')
        # tqdm.write(f"plot fig{count}")
        # print(f"plot fig{count}")
        count += 1
        

if __name__ == "__main__":
    
    main()