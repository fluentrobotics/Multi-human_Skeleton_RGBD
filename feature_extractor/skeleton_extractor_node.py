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

from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

# from feature_extractor.KalmanFilter import KalmanFilter
from feature_extractor.HumanKeypointsFilter import HumanKeypointsFilter
from feature_extractor.utils import get_pose_model_dir

# sub
COLOR_FRAME_TOPIC = '/camera/color/image_raw'
DEPTH_ALIGNED_TOPIC = '/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/aligned_depth_to_color/camera_info'

# pub
SKELETON_TOPIC = '/skeleton/keypoints_3d'
DEBUG_MARKER_SKELETON = '/skeleton/markers/data'
DEBUG_VIS_SKELETON = '/skeleton/vis/keypoints_vis'
PUB_FREQ : float = 20.0

CAMERA_FRAME = "camera_color_optical_frame"

SKELETON_NODE = "skeleton"

ID_TYPE = np.int16

bridge = CvBridge()
MAX_missing_count = 10

class skeletal_extractor_node():
    def __init__(self, 
                 rotate: int = cv2.ROTATE_90_CLOCKWISE, 
                 compressed: dict = {'rgb': True, 'depth': False}, 
                 syn: bool = False, 
                 pose_model: str = 'yolov8m-pose.pt',
                 use_kalman: bool = True,
                 vis = True,
    ):
        """
        
        """
        self.rotate = rotate        # rotate: rotate for yolov8 then rotate inversely   
        # ROTATE_90_CLOCKWISE=0, ROTATE_180=1, ROTATE_90_COUNTERCLOCKWISE=2 
        self.compressed = compressed
        self.syn = syn
        self.vis = vis
        self.use_kalman = use_kalman

        # yolov8n-pose, yolov8s-pose, yolov8m-pose, yolov8l-pose, yolov8x-pose, yolov8x-pose-p6
        self._POSE_model = YOLO(get_pose_model_dir() / pose_model)
        self._POSE_KEYPOINTS = 17
        self.human_dict = dict()

        # get ROS param ####
        # ########################################
        

        # Subscriber #############################
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
            self._subSync.registerCallback(self._callback)
        # ########################################
        
        
        # Publisher ##############################     
        # TODO: verify the msg type
        self._skeleton_pub = rospy.Publisher(
            SKELETON_TOPIC, PoseArray,queue_size=1
        )
        
        # self._debug_marker_skeleton = rospy.Publisher(
        #     DEBUG_MARKER_SKELETON, Marker,queue_size=1
        # )
        if vis:
            self._debug_vis_skeleton = rospy.Publisher(
                DEBUG_VIS_SKELETON, CompressedImage, queue_size=1
            )
        # ########################################
        
        
        # Camera Calibration Info ################
        camera_info_msg: CameraInfo = rospy.wait_for_message(
            CAMERA_INFO_TOPIC, CameraInfo, timeout=5
        )
        self._intrinsic_matrix = np.array(camera_info_msg.K).reshape(3,3)
        # ########################################
        
        rospy.spin()
    
    
    def _rgb_callback(self, msg: Image | CompressedImage) -> None:
        self._rgb_msg = msg

    def _depth_callback(self, msg: Image | CompressedImage) -> None:
        self._depth_msg = msg
    
    # TODO: def subscriber callback if necessary
    # ########################################

    
    def _callback(self):
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


        # delele missing pedestrians
        for key, value in self.human_dict.items():
            value.missing_count += 1
            if value.missing_count > MAX_missing_count:
                self.human_dict[key] = None
                del self.human_dict[key]
                # python memory management system will release deleted space in the future
   

        for id in id_human:
            # query or set if non-existing
            self.human_dict.setdefault(id, HumanKeypointsFilter(id=id, gaussian_blur=True, minimal_filter=True))
            self.human_dict[id].missing_count = 0       # reset missing_count of existing human

            # [K,3]
            keypoints_cam = self.human_dict[id].align_depth_with_color(keypoints_2d, depth_frame, self._intrinsic_matrix, rotate=self.rotate)

            if self.use_kalman:
                keypoints_cam = self.human_dict[id].kalmanfilter_cam(keypoints_cam)
            
            assert keypoints_cam.shape[0] == self._POSE_KEYPOINTS, 'There must be K keypoints'
            
            self.human_dict[id].keypoints_filtered = keypoints_cam
            



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


def main():
    rospy.init_node(SKELETON_NODE)
    node = skeletal_extractor_node()
    
    while not rospy.is_shutdown():
        # TODO: manage publisher frequency
        node._skeleton_inference_and_publish()
        rospy.sleep(1/PUB_FREQ)

if __name__ == '__main__':
    
    main()