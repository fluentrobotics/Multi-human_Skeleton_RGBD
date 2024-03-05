"""
Convert human keypoints and mask into ROS2 markers
"""
import numpy as np
# Message type
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Header, ColorRGBA, MultiArrayDimension
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

from feature_extractor.config import *

def add_human_skeletal_keypoint_Marker(human_id: ID_TYPE, 
                                       keypoint_KD: np.ndarray,
                                       keypoint_mask_K: np.ndarray,
                                       frame_id = "skeleton_frame",
                                       ns = "skeleton",
                                       offset = KEYPOINT_ID_OFFSET,
                                       ) -> Marker:
    """
    @human_id: ID_TYPE(np.int32)
    @keypoint_KD: [K,3]
    @keypoint_mask_K: [K,]

    return single human skeletal points MarkerArray
    """
    
    marker = Marker()
    marker.header.frame_id = frame_id
    # marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = human_id * HUMAN_MARKER_ID_VOL + offset     # 0 * 10 + 0
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
    
    K, D = keypoint_KD.shape
    for keypoint_id in range(K):
        if keypoint_mask_K[keypoint_id]:
            marker_point = Point(keypoint_KD[keypoint_id, 0], keypoint_KD[keypoint_id, 1], keypoint_KD[keypoint_id, 2])
            marker.points.append(marker_point)

    return marker


def add_human_skeletal_line_Marker(human_id: ID_TYPE, 
                                   keypoint_KD: np.ndarray,
                                   keypoint_mask_K: np.ndarray,
                                   frame_id = "skeleton_frame",
                                   ns = "skeleton",
                                   offset = LINE_ID_OFFSET) -> Marker:
    """
    @human_id: ID_TYPE(np.int32)
    @keypoint_KD: [K,3]
    @keypoint_mask_K: [K,]

    return single human skeletal lines MarkerArray
    """
    K, D = keypoint_KD.shape
    include_list = np.arange(K)[keypoint_mask_K].astype(np.int32)
    filtered_list = [pair for pair in SKELETAL_LINE_PAIRS_LIST 
                     if pair[0] in include_list and pair[1] in include_list]

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = human_id * HUMAN_MARKER_ID_VOL + offset
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.02
    marker.color = ColorRGBA(0.0, 1.0, 0.0, 1.0)

    for valid_pair in filtered_list:
        start_pos = Point(keypoint_KD[valid_pair[0],0], 
                          keypoint_KD[valid_pair[0],1],
                          keypoint_KD[valid_pair[0],2])
        
        end_pos = Point(keypoint_KD[valid_pair[1],0], 
                        keypoint_KD[valid_pair[1],1],
                        keypoint_KD[valid_pair[1],2])
        
        marker.points.extend([start_pos, end_pos])

    return marker
        

def delete_human_marker(human_id: ID_TYPE,
                        frame_id = "skeleton_frame",
                        ns = "skeleton",
                        offset_list: list = [KEYPOINT_ID_OFFSET, LINE_ID_OFFSET]
                        ) -> list[Marker]:
    
    remove_list: list[Marker] = []
    for offset in offset_list:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.ns = ns
        marker.id = human_id * HUMAN_MARKER_ID_VOL + offset
        marker.action = Marker.DELETE
        marker.color.a = 0.0
        
        remove_list.append(marker)
    
    return remove_list