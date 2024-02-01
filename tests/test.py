import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
# from tqdm import tqdm

from feature_extractor.config import *
from feature_extractor.Outlier_filter import find_inliers

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
    
error_idx_list = [112, 115, 192, 203, 207, 210, 217, 220, 233, 270]
idx = 115

data_KD = keypoints_list[idx][0]
mask_K = mask_list[idx][0]

inliers_mask = find_inliers(data_KD, mask_K)

print("inliers:\n",inliers_mask)