import numpy as np
import cv2
from feature_extractor.KalmanFilter import KalmanFilter
from feature_extractor.utils import logger

from feature_extractor.config import MAX_MISSING

class HumanKeypointsFilter:
    """
    1. Gaussian blur (optional)
    2. Minimal Filter
    3. Projection to camera coodinate
    4. Kalman Filter
    
    NOTE: Vectorization calculation would be much faster
    """
    def __init__(self, id, 
                 gaussian_blur: bool = True,
                 minimal_filter: bool = True,
                 num_keypoints = None,
                 ) -> None:
        """
        required: id
        optional: gaussian_blur, minimal_filter
        """
        self.id = id
        self.keypoints_filtered: np.ndarray = None              # 3D keypoints [K,3]
        self.missing_count: int = 0
        self.valid_keypoints: np.ndarray                        # valid keypoint mask
        
        
        self.gaussian_blur = gaussian_blur
        self.minimal_filter = minimal_filter
        
        self.filters = np.empty(num_keypoints, dtype=object)
        self.missing_keypoints = np.zeros(num_keypoints, dtype=int)

    def gaussianBlur_depth(self, depth_frame: np.ndarray, kernel_size=(11,11)) -> np.ndarray:
        """
        @depth_frame: [axis_0, axis_1]
        @kernel_size: (m,m)
        """
        return cv2.GaussianBlur(depth_frame, kernel_size, 0)
    
    def minimalFilter(self, depth_frame, keypoints_pixel, kernel_size=(11,11)) -> np.ndarray:
        """
        @keypoint_pixels: [K-,3]
        """
        # shape = cv2.MORPH_RECT
        # kernel = cv2.getStructuringElement(shape, kernel_size)
        img_shape = depth_frame.shape
        # logger.debug(f"depth img shape: {img_shape}")
        # valid mask
        keypoints_pixel = keypoints_pixel[self.valid_keypoints,...]
        for keypoint_pos in keypoints_pixel[:, :2]:
            left = keypoint_pos[1] - kernel_size[1] // 2
            top = keypoint_pos[0] - kernel_size[0] // 2
            right = left + kernel_size[1]
            bottom = top + kernel_size[0]
            # logger.debug(f"left right top bottom:{left,right,top,bottom}")
            # check if out of boundary
            if left < 0 or right > img_shape[1] or top < 0 or bottom > img_shape[0]:
                break
            # depth_frame = cv2.erode(depth_frame[left:right, top:bottom], kernel)

            depth_frame[top:bottom,left:right] = np.min(depth_frame[top:bottom,left:right])
            
        return depth_frame
        
        
    def align_depth_with_color(self, keypoints_2d, depth_frame, intrinsic_mat, rotate = cv2.ROTATE_90_CLOCKWISE):
        """
        @keypoints_2d: [K, 3(x,y,conf)]. x=y=0.00 with low conf means the keypoint does not exist
        @depth_frame: []
        @intrinsic_mat: cameara intrinsic K
        @rotate = 0 | None
        
        Mask: self.valid_keypoints
        return keypoints in camera coordinate [K,3]
        """
        # valid keypoints mask
        # logger.debug(f"keypoints_2d:\n{keypoints_2d}")
        valid_xy = keypoints_2d[:,:2] != 0.00        # bool [K,2]
        self.valid_keypoints = valid_xy[:,0] & valid_xy[:,1]        # bool vector [K,]
        self.valid_keypoints = self.valid_keypoints.astype(bool)
        # logger.debug(f"\nkeypoint mask:\n{self.valid_keypoints}")
        # logger.debug(f"\nkeypoint:\n{keypoints_2d}")
        # convert keypoints to the original coordinate
        if rotate == cv2.ROTATE_90_CLOCKWISE:
            axis_0 = depth_frame.shape[0]-1 - keypoints_2d[:, 0:1]  # vector [K,1], -1 to within the idx range
            axis_1 = keypoints_2d[:, 1:2]                           # vector [K,1]

        else:
            # no ratation
            axis_0 = keypoints_2d[:, 1:2]
            axis_1 = keypoints_2d[:, 0:1]
        
        axis_2 = np.ones_like(axis_0)

        keypoints_pixel = np.concatenate((axis_0, axis_1, axis_2), axis=1).astype(np.int16)     # [K, 3]
        
        # Preprocessing Filters
        if self.gaussian_blur:
            depth_frame = self.gaussianBlur_depth(depth_frame)
        if self.minimal_filter:
            depth_frame = self.minimalFilter(depth_frame, keypoints_pixel)
        
        keypoints_depth = depth_frame[keypoints_pixel[:,0], keypoints_pixel[:,1]]               # [K,]
        #                   [3,3]           [3,K]               [K,K]
        raw_keypoints_cam = np.linalg.inv(intrinsic_mat) @ keypoints_pixel.T @ np.diag(keypoints_depth)            # [3, K]     inverse intrinsic
        return raw_keypoints_cam.T    #[K,3]
    
    
    def kalmanfilter_cam(self, raw_keypoints_cam: np.ndarray, freq: float = 30.0):
        """
        @ raw_keypoints_cam: keypoints to be filtered [K,3]
        @ freq: Kalman Filter updation frequency
        return: keypoints_filtered [K,3]
        """
        K, D = raw_keypoints_cam.shape
        keypoints_filtered = np.zeros_like(raw_keypoints_cam)
        

        # NOTE: need vectorization?
        for k in range(K):
            if self.filters[k] is None:
                if self.valid_keypoints[k]:
                    self.missing_keypoints[k] = 0
                    self.filters[k] = KalmanFilter(freq=freq)
                    self.filters[k].initialize(raw_keypoints_cam[k,...])        # raw_keypoints_cam[k] is a xyz vector
                    keypoints_filtered[k] = self.filters[k].getMeasAfterInitialize()
                else:
                    self.missing_keypoints[k] += 1

            else:
                if self.valid_keypoints[k]:
                    self.missing_keypoints[k] = 0
                    keypoints_filtered[k] = self.filters[k].update(raw_keypoints_cam[k])
                else:
                    self.missing_keypoints[k] += 1
                    keypoints_filtered[k] = self.filters[k].updateOpenLoop()
                    if self.missing_keypoints[k] >= MAX_MISSING:
                        self.filters[k] = None

        return keypoints_filtered


    def inlier_filter(keypoints_cam: np.ndarray):
        """
        @ keypoints_cam: keypoints with outliers [K,3]

        return @ inlier_mask: [K,] inlier is True, outlier is False
        """
        K, D = keypoints_cam.shape

        inlier_mask = np.ones(K)

        #  TODO: find the center of valid data, find idx of outliers, inlier_mask = inlier AND valid
