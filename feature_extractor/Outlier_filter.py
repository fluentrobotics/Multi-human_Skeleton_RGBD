import numpy as np

# Global Config
from feature_extractor.config import *

# Local Config
THRESHOLD = 1.0     # 1.0 meter
MAX_ITER = 5

def find_inliers(data_KD: np.ndarray, mask_K: np.ndarray) -> np.ndarray:
    """
    @ data_KD: [K,D] float
    @ mask_K: [K,] bool

    return inlier_mask: inlier: True, outlier: False
    """
    K, D = data_KD.shape
    # Since keypoints 0 to 4 often drift, we just exclude them.
    inliers_mask = np.ones(K, dtype=bool)
    inliers_mask[:5] = False
    for _ in range(MAX_ITER):
        mask = mask_K & inliers_mask
        geo_center = np.mean(data_KD[mask, ...], axis=0)
        assert geo_center.shape == (D,) , "Wrong Dimension"

        err_squareSum = np.sum( np.square(data_KD - geo_center.reshape(1,3)), axis=1)
        # print(err_squareSum)
        new_inliers_mask = err_squareSum < THRESHOLD
        new_inliers_mask[:5] = False
        
        new_mask = new_inliers_mask & mask_K

        if np.array_equal(new_mask, mask):
            # print("equal")
            # print("new:",new_mask)
            # print("pre:",mask)
            # print("\nmask:\n", mask_K)
            return new_mask
        else:
            # print("Not equal")
            # print("new:",new_mask)
            # print("pre:",mask)
            inliers_mask = new_inliers_mask

    # print("reach max iter")
    # print("\nmask:\n", mask_K)
    return new_mask