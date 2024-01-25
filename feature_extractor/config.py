import numpy as np

from feature_extractor.utils import get_project_dir_path

# SKELETAL NODE config #############################################################

# SUB topics
COLOR_FRAME_TOPIC = '/camera/color/image_raw'
COLOR_COMPRESSED_FRAME_TOPIC = '/camera/color/image_raw/compressed'
DEPTH_ALIGNED_TOPIC = '/camera/aligned_depth_to_color/image_raw'
DEPTH_ALIGNED_COMPRESSED_TOPIC = '/camera/aligned_depth_to_color/image_raw/compressed'
CAMERA_INFO_TOPIC = '/camera/aligned_depth_to_color/camera_info'
CAMERA_INTRINSIC = [906.7041625976562, 0.0, 653.4981689453125, 0.0, 906.7589111328125, 375.4635009765625, 0.0, 0.0, 1.0]
# PUB topic
SKELETON_HUMAN_ID_TOPIC = 'skeleton/numpy_msg/human_id'
SKELETON_MASK_MAT_TOPIC = 'skeleton/numpy_msg/mask'
RAW_SKELETON_TOPIC = '/skeleton/numpy_msg/raw_keypoints_3d'
FILTERED_SKELETON_TOPIC = '/skeleton/numpy_msg/filtered_keypoints_3d'
RVIZ_IMG2D_SKELETON_TOPIC = '/skeleton/vis/keypoints_2d_img'
RVIZ_MARKER3D_SKELETON_TOPIC = '/skeleton/vis/keypoints_3d_makers'
PUB_FREQ:float = 20.0

CAMERA_FRAME = "camera_color_optical_frame"

# DATA TYPE
ID_TYPE = np.int32

# Node PARAMETERS
SKELETON_NODE = "skeleton"
COMPRESSED_TOPICS = {'rgb': True, 'depth': False}
MAX_MISSING = 5
SKELETAL_LINE_PAIRS_LIST = [(4,2),(2,0),(0,1),(1,3),
                            (10,8),(8,6),(6,5),(5,7),(7,8),
                            (6,12),(12,14),(14,16),(5,11),(11,13),(13,15),(12,11)]
SAVE_DATA = True

# Pre-trained Model
POSE_MODEL = 'yolov8m-pose.pt'

# RViz Visualization (code errors)
RVIZ_VIS = False
HUMAN_MARKER_ID_VOL = 10
KEYPOINT_ID_OFFSET = 0
LINE_ID_OFFSET = 1

# PATH
PROJECT_PATH = get_project_dir_path()
DATA_DIR_PATH = PROJECT_PATH / "data"
###################################################################################

# TODO: Changeable Parameters
TEST_NAME = "test_Minimal_False"
SAVE_YOLO_IMG = True

# Filters
USE_KALMAN = True
MINIMAL_FILTER = False