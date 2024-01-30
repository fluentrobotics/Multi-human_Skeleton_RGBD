import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
# from tqdm import tqdm

from feature_extractor.config import *

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
# # keypoints_list, keypoints_no_Kalman_list, mask_list
# for idx in error_idx_list:
#     print(f"{idx}:\nMask:{mask_list[idx]}\nKalman:\n{keypoints_list[idx]}\nNo Kalman:\n{keypoints_no_Kalman_list[idx]}")
#     valid_keypoints = keypoints_list[idx][0, mask_list[idx][0],...]        # 0 because single person
#     print(f"\nvalid keypoints:\n{valid_keypoints}")
#     center = np.mean(valid_keypoints, axis=1)
#     print(f"center:{center}")
#     error_idx:int = input()


while(True):
    print("\nEnter your Query Idx:")
    idx: int = eval(input())
    print(f"{idx}:\nMask:{mask_list[idx]}\nKalman: shape{keypoints_list[idx].shape}\n{keypoints_list[idx]}\nNo Kalman: shape{keypoints_no_Kalman_list[idx].shape}\n{keypoints_no_Kalman_list[idx]}")
    valid_body_position = SKELETAL2BODY[mask_list[idx][0]]
    valid_keypoints = keypoints_list[idx][0, mask_list[idx][0],...]        # 0 because single person
    print(f"\nvalid body:\n{valid_body_position}\nvalid keypoints: shape{valid_keypoints.shape}\n{valid_keypoints}")
    center = np.mean(valid_keypoints, axis=1)
    print(f"center:{center}")