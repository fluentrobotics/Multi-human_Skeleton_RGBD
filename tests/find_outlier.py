import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm

from feature_extractor.config import *
from feature_extractor.utils import logger

if SAVE_DATA:
    pickle_path = DATA_DIR_PATH / "pickle" / TEST_NAME
    pickle_path = pickle_path.absolute().as_posix() + "_keypoints.pkl"

    file = open(pickle_path, 'rb')
    keypoints_list: list[np.ndarray] = pickle.load(file)
    mask_list: list[np.ndarray] = pickle.load(file)
    keypoints_no_Kalman_list: list[np.ndarray] = pickle.load(file)
    file.close()

else:
    logger.error("Cannot find pickle")
    exit()

count = 0
for data_Kalman, data_no_Kalman, mask in zip(keypoints_list, keypoints_no_Kalman_list, mask_list):

    if data_Kalman is None or mask is None:
        count += 1
        logger.warning(f"{count}: no data")
    
    else:

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        count += 1
        
        
        
        
        
        
        
        
        
        
        