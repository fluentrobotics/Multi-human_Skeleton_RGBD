"""Extract images and publish skeletons.
"""

import os
import argparse
import cv2
import rosbag
import rospy
import message_filters
import numpy as np
from typing import Optional, TypedDict
from rospy.numpy_msg import numpy_msg

from std_msgs.msg import Int32MultiArray, Float32MultiArray, Header, ColorRGBA
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

# from feature_extractor.KalmanFilter import KalmanFilter
from feature_extractor.HumanKeypointsFilter import HumanKeypointsFilter
from feature_extractor.utils import get_pose_model_dir, logger

# sub
COLOR_FRAME_TOPIC = '/camera/color/image_raw'
DEPTH_ALIGNED_TOPIC = '/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/aligned_depth_to_color/camera_info'

# pub
SKELETON_HUMAN_ID_TOPIC = 'skeleton/numpy_msg/human_id'
SKELETON_MASK_MAT_TOPIC = 'skeleton/numpy_msg/mask'
RAW_SKELETON_TOPIC = '/skeleton/numpy_msg/raw_keypoints_3d'
FILTERED_SKELETON_TOPIC = '/skeleton/numpy_msg/filtered_keypoints_3d'
VIS_IMG2D_SKELETON_TOPIC = '/skeleton/compressed/keypoints_2d'
VIS_MARKER3D_SKELETON_TOPIC = '/skeleton/markers/keypoints_3d'
PUB_FREQ : float = 30.0

CAMERA_FRAME = "camera_color_optical_frame"

SKELETON_NODE = "skeleton"

ID_TYPE = np.int32
HUMAN_MARKER_ID_VOL = 10
KEYPOINT_ID_OFFSET = 0
LINE_ID_OFFSET = 1

bridge = CvBridge()
MAX_missing_count = 5

SKELETAL_LINE_PAIRS_LIST = [(4,2),(2,0),(0,1),(1,3),
                            (10,8),(8,6),(6,5),(5,7),(7,8),
                            (6,12),(12,14),(14,16),(5,11),(11,13),(13,15),(12,11)]

class skeletal_extractor_node():
    def __init__(self, 
                 rotate: int = cv2.ROTATE_90_CLOCKWISE, 
                 compressed: dict = {'rgb': True, 'depth': False}, 
                 syn: bool = False, 
                 pose_model: str = 'yolov8m-pose.pt',
                 use_kalman: bool = True,
                 vis: bool = True,
    ):
        """
        @rotate: default =  cv2.ROTATE_90_CLOCKWISE
        @compressed: default = {'rgb': True, 'depth': False}
        @syn: syncronization, default = False
        @pose_model: default = 'yolov8m-pose.pt'
        @use_kalman: Kalman Filter, default = True
        @vis: RViz to show skeletal keypoints and lines
        """
        self.rotate = rotate        # rotate: rotate for yolov8 then rotate inversely   
        # ROTATE_90_CLOCKWISE=0, ROTATE_180=1, ROTATE_90_COUNTERCLOCKWISE=2 
        self.compressed = compressed
        self.syn = syn
        self.vis = vis
        self.use_kalman = use_kalman

        logger.info(f"Initialization\n rotate={self.rotate}\n \
                    compressed={self.compressed}\n \
                    syncronization={self.syn}\n \
                    pose_model={self._POSE_model}\n \
                    KalmanFilter={self.use_kalman}\n \
                    Visualization={self.vis}\n \
                    PubFreqency={PUB_FREQ}\n \
                    NodeName={SKELETON_NODE}\n \
                    MaxMissingCount={MAX_missing_count}\n \
                    ")
        
        # yolov8n-pose, yolov8s-pose, yolov8m-pose, yolov8l-pose, yolov8x-pose, yolov8x-pose-p6
        self._POSE_model = YOLO(get_pose_model_dir() / pose_model)
        self._POSE_KEYPOINTS = 17
        self.human_dict = dict()
        self.keypoints_HKD: np.ndarray

        # get ROS param ####
        # ########################################
        

        # Subscriber ##########################################################
        self._rgb_sub = message_filters.Subscriber(
            COLOR_FRAME_TOPIC, CompressedImage, self._rgb_callback, queue_size=1
        )
        self._depth_sub = message_filters.Subscriber(
            DEPTH_ALIGNED_TOPIC, Image, self._depth_callback, queue_size=1
        )
        self._rgb_msg: Optional[CompressedImage] = None
        self._depth_msg: Optional[Image] = None
        
        # TODO: built-in syncronization or customized syn, now built-in
        if self.syn:
            self._subSync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub],
                                                                queue_size=1)
            self._subSync.registerCallback(self._syn_callback)
        # #####################################################################
        
        
        # Publisher ###########################################################     
        # TODO: verify the msg type
        self._skeleton_human_id_pub = rospy.Publisher(
            SKELETON_HUMAN_ID_TOPIC, numpy_msg(Int32MultiArray), queue_size=1
        )
        self._skeleton_mask_mat_pub = rospy.Publisher(
            SKELETON_MASK_MAT_TOPIC, numpy_msg(Int32MultiArray), queue_size=1
        )
        self._raw_skeleton_pub = rospy.Publisher(
            RAW_SKELETON_TOPIC, numpy_msg(Float32MultiArray), queue_size=1
        )
        self._filtered_skeleton_pub = rospy.Publisher(
            FILTERED_SKELETON_TOPIC, numpy_msg(Float32MultiArray), queue_size=1
        )
        # VISUALIZATION ####################################################
        if self.vis:
            self._vis_keypoints_img2d_pub = rospy.Publisher(
                VIS_IMG2D_SKELETON_TOPIC, CompressedImage, queue_size=1
            )
            self._vis_keypoints_marker3d_pub = rospy.Publisher(
                VIS_MARKER3D_SKELETON_TOPIC, MarkerArray, queue_size=1
            )
        # #####################################################################
        
        
        # Camera Calibration Info ################
        camera_info_msg: CameraInfo = rospy.wait_for_message(
            CAMERA_INFO_TOPIC, CameraInfo, timeout=5
        )
        self._intrinsic_matrix = np.array(camera_info_msg.K).reshape(3,3)
        # ########################################
        
    


    def _rgb_callback(self, msg: Image | CompressedImage) -> None:
        self._rgb_msg = msg

    def _depth_callback(self, msg: Image | CompressedImage) -> None:
        self._depth_msg = msg
    
    # TODO: def subscriber callback if necessary
    # ########################################

    
    def _syn_callback(self):
        self._skeleton_inference_and_publish()


    def _skeleton_inference_and_publish(self) -> None:
        if self._rgb_msg is None or self._depth_msg is None:
            return
        
        rgb_msg_header = self._rgb_msg.header
        
        # compressed img
        if self.compressed['rgb']:
            bgr_frame = bridge.compressed_imgmsg_to_cv2(self._rgb_msg)
        else:
            bgr_frame = bridge.imgmsg_to_cv2(self._rgb_msg)
            
        if self.compressed['depth']:
            depth_frame = bridge.compressed_imgmsg_to_cv2(self._depth_msg)
        else:
            depth_frame = bridge.imgmsg_to_cv2(self._depth_msg)
        depth_frame = depth_frame.astype(np.float32) / 1e3          # millimeter to meter
            
        # rotate img
        if self.rotate is not None:
            bgr_frame = cv2.rotate(bgr_frame, self.rotate)
            depth_frame = cv2.rotate(depth_frame, self.rotate)
        
        # 2D-keypoints
        # res.keypoints.data : tensor [human, keypoints, 3 (x, y, confidence) ] float
        #                      0.00 if the keypoint does not exist
        # res.boxes.id : [human_id, ] float
        results: list[Results] = self._POSE_model.track(bgr_frame, persist=True, verbose=False)
        if not results:
            return
        yolo_res = results[0]
        
        # 2D -> 3D ################################################################
        num_human, num_keypoints, num_dim = yolo_res.keypoints.data.shape
        conf_keypoints = yolo_res.keypoints.conf        # [H,K]
        keypoints_2d = yolo_res.keypoints.data             # [H,K,D(x,y,conf)] [H, 17, 3]

        conf_boxes = yolo_res.boxes.conf                # [H]
        id_human = yolo_res.boxes.id.astype(ID_TYPE)    # [H]

        # if no detections, gets [1,0,51]
        if num_keypoints == 0 or id_human == None:
            return

        # TODO: warning if too many human in the dictionary
        # if len(self.human_dict) >= MAX_dict_vol:
        all_marker_list = []
        # delele missing pedestrians
        for key, value in self.human_dict.items():
            value.missing_count += 1
            if value.missing_count > MAX_missing_count:
                self.human_dict[key] = None
                del self.human_dict[key]
                # python memory management system will release deleted space in the future
                
                # VISUALIZATION ####################################################
                if self.vis:
                    delete_list = delete_human_marker(key, offset_list=[KEYPOINT_ID_OFFSET, LINE_ID_OFFSET])
                    all_marker_list.extend(delete_list)
                
   
        keypoints_3d = np.zeros((num_human, num_keypoints, 3))  # [H,K,3]
        keypoints_raw = np.zeros((num_human, num_keypoints, 3)) # [H,K,3]
        keypoints_mask = np.zeros((num_human, num_keypoints))   # [H,K]
        for idx, id in enumerate(id_human, start=0):
            # query or set if non-existing
            self.human_dict.setdefault(id, HumanKeypointsFilter(id=id, gaussian_blur=True, minimal_filter=True))
            self.human_dict[id].missing_count = 0       # reset missing_count of existing human

            # [K,3]
            keypoints_cam = self.human_dict[id].align_depth_with_color(keypoints_2d, depth_frame, self._intrinsic_matrix, rotate=self.rotate)
            keypoints_raw[idx,...] = keypoints_cam      # [K,3]
            if self.use_kalman:
                keypoints_cam = self.human_dict[id].kalmanfilter_cam(keypoints_cam)     # [K,3]
            
            assert keypoints_cam.shape[0] == self._POSE_KEYPOINTS, 'There must be K keypoints'
            
            self.human_dict[id].keypoints_filtered = keypoints_cam
            keypoints_3d[idx,...] = keypoints_cam     # [K,3]
            keypoints_mask[idx,...] = self.human_dict[id].valid_keypoints
        
        
            # VISUALIZATION ####################################################
            if self.vis:
                add_keypoint_marker = add_human_skeletal_keypoint_Marker(human_id= id, 
                                                                         keypoint_KD= keypoints_cam, 
                                                                         keypoint_mask_K= self.human_dict[id].valid_keypoints,
                                                                         offset= KEYPOINT_ID_OFFSET,
                                                                         )
                
                add_line_marker = add_human_skeletal_line_Marker(human_id= id,
                                                                 keypoint_KD= keypoints_cam,
                                                                 keypoint_mask_K= self.human_dict[id].valid_keypoints,
                                                                 offset= LINE_ID_OFFSET)
                
                all_marker_list.extend([add_keypoint_marker, add_line_marker])
                
        # Publish keypoints and markers
        # TODO: show missing pedestrians?

        # id_msg = Int32MultiArray()
        # id_msg.data = id_human.tolist()     # much faster than for loop
        
        # NOTE: numpy_msg
        self._skeleton_human_id_pub.publish(id_human)           # [H,]
        self._skeleton_mask_mat_pub.publish(keypoints_mask)     # [H,K]
        self._raw_skeleton_pub.publish(keypoints_raw)           # [H,K,D]
        self._filtered_skeleton_pub.publish(keypoints_3d)       # [H,K,D]

        # VISUALIZATION ####################################################
        if self.vis:
            self._vis_keypoints_img2d_pub.publish(yolo_res.plot())

            all_marker_array = MarkerArray()
            all_marker_array.markers = all_marker_list
            self._vis_keypoints_marker3d_pub.publish(all_marker_array)


##############################################################################################
# VISUALIZATION function #####################################################################
def add_human_skeletal_keypoint_Marker(human_id: ID_TYPE, 
                                       keypoint_KD: np.ndarray,
                                       keypoint_mask_K: np.ndarray, 
                                       frame_id = "skeleton_frame",
                                       ns = "skeleton",
                                       offset = KEYPOINT_ID_OFFSET) -> Marker:
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
        

# class yolo_extractor_2D():
#     """generate 2d keypoints from a single image or images
#     """

#     def __init__(self,
#                 #  bag_file: str = "/home/xmo/bagfiles/RGBD.bag", 
#                 #  output_dir: str = "/home/xmo/bagfiles/extract/",
#                 ):
        
#         # self.bag = rosbag.Bag(bag_file, "r")
#         self.topics = [COLOR_FRAME_TOPIC, DEPTH_ALIGNED_TOPIC, CAMERA_INFO_TOPIC]


#     def extract_keypoints_2d(self, bgr_frame: np.ndarray) -> list[Results]:
#         """ 
#         YOLOv8 2d keypoints
#         return res
#          @res.keypoints.data : tensor [human, keypoints, 3 (x, y, confidence) ] float
#          @res.boxes.id : [human_id, ] float
#         """
#         results: list[Results] = POSE_model.predict(bgr_frame)

#         res = results[0]
#         # res.keypoints.data : tensor [human, keypoints, 3 (x, y, confidence) ] float
#         # res.boxes.id : [human_id, ] float
#         return res


# class pedestrian:
#     """
#     required: id, keypoints
#     optional: conf_boxes, conf_keypoints
    
#     target: keypoints_filtered(3d keypoints)
#     """
#     def __init__(self, id, keypoints, conf_boxes = None, conf_keypoints = None) -> None:
#         self.id = id
#         self.keypoints = keypoints
#         self.conf_boxes = conf_boxes
#         self.conf_keypoints = conf_keypoints

#         self.missing_count = 0
#         self.filter : KalmanFilter = None
#         self.keypoints_filtered: np.ndarray = None               # 3D keypoints [3, K]


def main() -> None:
    rospy.init_node(SKELETON_NODE)
    node = skeletal_extractor_node()
    logger.success("Skeleton Node initialized")

    rospy.spin()
    
    while not rospy.is_shutdown():
        # TODO: manage publisher frequency
        node._skeleton_inference_and_publish()
        rospy.sleep(1/PUB_FREQ)

if __name__ == '__main__':

    main()