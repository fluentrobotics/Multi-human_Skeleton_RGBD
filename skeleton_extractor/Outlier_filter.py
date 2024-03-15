import numpy as np

# Global Config
from skeleton_extractor.config import *

# Local Config
THRESHOLD = 1     # 1.0 meter
MAX_ITER = 5
DEPTH_THRESHOULD = 0.3

def find_inliers(data_KD: np.ndarray, mask_K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    @ data_KD: [K,D] float
    @ mask_K: [K,] bool
    
    return new_mask, geometric center
    """
    K, D = data_KD.shape
    inliers_mask = np.ones(K, dtype=bool)
    
    for _ in range(MAX_ITER):
        mask = mask_K & inliers_mask

        # Since keypoints 0 to 4 often drift, we just exclude them when calculate the geo_center.
        geo_mask = mask.copy()
        geo_mask[:5] = False
        geo_mask[7:11] = False
        geo_mask[13:] = False
        geo_center = np.mean(data_KD[geo_mask, ...], axis=0)
        assert geo_center.shape == (D,) , "Wrong Dimension"

        err_squareSum = np.sum( np.square(data_KD - geo_center.reshape(1,3)), axis=1)
        # print(err_squareSum)
        new_inliers_mask = err_squareSum < THRESHOLD
        depth_inliers_mask = np.absolute(data_KD[:,2]-geo_center[2]) < DEPTH_THRESHOULD
        # new_inliers_mask[:5] = False
        
        new_mask = new_inliers_mask & mask_K
        # new_mask = depth_inliers_mask & new_mask

        if np.array_equal(new_mask, mask):
            # print("equal")
            # print("new:",new_mask)
            # print("pre:",mask)
            # print("\nmask:\n", mask_K)
            break
        else:
            # print("Not equal")
            # print("new:",new_mask)
            # print("pre:",mask)
            inliers_mask = new_inliers_mask

    # print("reach max iter")
    # print("\nmask:\n", mask_K)
    return new_mask, geo_center