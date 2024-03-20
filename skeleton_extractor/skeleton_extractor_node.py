#! /usr/bin/env python
# Multi-human_Skeleton_RGB-D the fluent robotics lab, AGPL-3.0 license

"""Extract images and publish skeletons.
"""
import os
import threading
import argparse
import cv2
import pickle
import numpy as np
from typing import Optional, TypedDict
from pathlib import Path

# import rosbag
import rclpy
from rclpy.node import Node
# import message_filters
# from rospy.numpy_msg import numpy_msg

# Message type
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Header, ColorRGBA, MultiArrayDimension
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from skeleton_extractor.HumanKeypointsFilter import HumanKeypointsFilter
from skeleton_extractor.Outlier_filter import find_inliers
from skeleton_extractor.rviz2_maker import *
from skeleton_extractor.utils import *

# Global Config
from skeleton_extractor.config import *
# Local Config
bridge = CvBridge()




class skeletal_extractor_node(Node):
    def __init__(self, 
                 rotate: int = cv2.ROTATE_90_CLOCKWISE, 
                 compressed: dict = {'rgb': True, 'depth': False}, 
                 syn: bool = False, 
                 pose_model: str = 'yolov8m-pose.pt',
                 use_kalman: bool = True,
                 gaussian_blur: bool = True,
                 minimal_filter: bool = True,
                 outlier_filter: bool = True,
                 rviz: bool = False,
                 init_rviz: bool = True,
                 save: bool = True,
                 ref_link: str = 'link_head'
    ):
        """
        @rotate: default =  cv2.ROTATE_90_CLOCKWISE
        @compressed: default = {'rgb': True, 'depth': False}
        @syn: syncronization, default = False
        @pose_model: default = 'yolov8m-pose.pt'
        @use_kalman: Kalman Filter, default = True
        @rviz: RViz to show skeletal keypoints and lines
        """
        super().__init__(SKELETON_NODE)
        self.rotate = rotate        # rotate: rotate for yolov8 then rotate inversely   
        # ROTATE_90_CLOCKWISE=0, ROTATE_180=1, ROTATE_90_COUNTERCLOCKWISE=2 
        self.compressed = compressed
        self.syn = syn
        self.rviz = rviz
        self.use_kalman = use_kalman
        self.use_gaussian_blur = gaussian_blur
        self.use_minimal_filter = minimal_filter
        self.use_outlier_filter = outlier_filter
        self.save = save
        self.ns = "skeleton"
        self.frame_id = ref_link

        logger.info(f"\nInitialization\n rotate={self.rotate}\n \
                    compressed={self.compressed}\n \
                    syncronization={self.syn}\n \
                    pose_model={pose_model}\n \
                    KalmanFilter={self.use_kalman}\n \
                    Rviz={self.rviz}\n \
                    PubFreqency={PUB_FREQ}\n \
                    NodeName={SKELETON_NODE}\n \
                    MaxMissingCount={MAX_MISSING}\n \
                    Gaussian_blur={gaussian_blur}\n \
                    Minimal_filter={minimal_filter}\n \
                    Outlier_filter={outlier_filter}\n \
                    ")
        
        # yolov8n-pose, yolov8s-pose, yolov8m-pose, yolov8l-pose, yolov8x-pose, yolov8x-pose-p6
        self._POSE_model = YOLO(get_pose_model_dir() / pose_model)
        self._POSE_KEYPOINTS = 17
        self.human_dict = dict()
        self.keypoints_HKD: np.ndarray
        self.step = 0


        # logger debugger
        self.no_human_debugger = False
        self.no_message_debugger = False


        
        # Subscriber ##########################################################
        if self.compressed['rgb']:
            self._rgb_sub = self.create_subscription(
                CompressedImage, COLOR_COMPRESSED_FRAME_TOPIC, self._rgb_callback, 1
            )
            self._rgb_msg: CompressedImage = None
        else:
            self._rgb_sub = self.create_subscription(
                Image, COLOR_FRAME_TOPIC, self._rgb_callback, 1
            )
            self._rgb_msg: Image = None
        if self.compressed['depth']:
            self._depth_sub = self.create_subscription(
            CompressedImage, DEPTH_ALIGNED_COMPRESSED_TOPIC, self._depth_callback, 1
        )
            self._depth_msg: CompressedImage = None
        else:
            self._depth_sub = self.create_subscription(
            Image, DEPTH_ALIGNED_TOPIC, self._depth_callback, 1
        )
            self._depth_msg: Image = None
            


        # Timer for Publisher ################################################
        timer_period = 1/PUB_FREQ    
        self.pub_timer = self.create_timer(timer_period, self._timer_callback)

        # Publisher ###########################################################   
        # # TODO: now we only publish RViz firstly
        # self._skeleton_human_id_pub = self.create_publisher(
        #     Int32MultiArray, SKELETON_HUMAN_ID_TOPIC, 1
        # )
        # self._skeleton_mask_mat_pub = self.create_publisher(
        #     Int32MultiArray, SKELETON_MASK_MAT_TOPIC, 1
        # )
        # self._raw_skeleton_pub = self.create_publisher(
        #     Float32MultiArray, RAW_SKELETON_TOPIC, 1
        # )
        # self._filtered_skeleton_pub = self.create_publisher(
        #     Float32MultiArray, FILTERED_SKELETON_TOPIC, 1
        # )

        # RViz Skeletons ####################################################
        if self.rviz:
            self._rviz_keypoints_img2d_pub = self.create_publisher(
                Image, RVIZ_IMG2D_SKELETON_TOPIC, 5
            )
            self._rviz_keypoints_marker3d_pub = self.create_publisher(
                MarkerArray, RVIZ_MARKER3D_SKELETON_TOPIC, 5
            )
            self._reset_rviz(ns=self.ns)
            logger.debug(f"clear markers in {self.ns}")
        
        if self.save:
            # save keypoints
            self.filtered_keypoints = None
            self.keypoints_mask = None
            self.keypoints_no_Kalman= None
            # save YOLO plot
            self.video_img = None

        # #####################################################################
        

        # Camera Calibration Info Subscriber ##################################
        self.receive_from_camera: bool = False
        self.waiting_cam_count: int = 0
        self._cam_info_sub = self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._cam_info_callback, 1)

        self._msg_lock = threading.Lock()   # lock the data resource
        self._reset_sub_msg()


        # #####################################################################



        # Test variable #############
        self.current_human: list = list()
        #############################
        
    def _timer_callback(self) -> None:
        if self.receive_from_camera:
            self._skeleton_inference_and_publish()
        else:
            self.waiting_cam_count += 1
            if self.waiting_cam_count % 40 == 0:
                logger.warning(f"waiting for camera information, {self.waiting_cam_count // 20}, now: {self.get_clock().now().nanoseconds}")

            # wait too long
            # if self.waiting_cam_count >= 400:
            #     sys.exit()

    def _rgb_callback(self, msg: Image | CompressedImage) -> None:
        with self._msg_lock:
            self._rgb_msg = msg

    def _depth_callback(self, msg: Image | CompressedImage) -> None:
        with self._msg_lock:
            self._depth_msg = msg

    def _cam_info_callback(self, msg: CameraInfo) -> None:
        with self._msg_lock:
            self._intrinsic_matrix = np.array(msg.k).reshape(3,3)
            if not self.receive_from_camera:
                self.receive_from_camera = True
                logger.info(f"\nCamera Intrinsic Matrix:\n{self._intrinsic_matrix}")
            

    def _reset_sub_msg(self):
        with self._msg_lock:
            self._rgb_msg = None
            self._depth_msg = None

    def _reset_rviz(self, ns='skeleton') -> None:
        deleteall_list = deleteall_marker(ns=ns, frame_id=self.frame_id)
        marker_array = MarkerArray()
        marker_array.markers = deleteall_list
        self._rviz_keypoints_marker3d_pub.publish(marker_array)


    def _skeleton_inference_and_publish(self) -> None:

        self.step += 1
        if self.save:
            self.filtered_keypoints = None
            self.keypoints_mask = None
            self.keypoints_no_Kalman= None
            self.YOLO_plot = None
        
        
        # no rgb-d message ##########################################
        if self._rgb_msg is None or self._depth_msg is None:
            if not self.no_message_debugger:
                logger.debug(f"No RGB-D messages")
                self.no_message_debugger = True
            return
        self.no_message_debugger = False

        #############################################################
        #############################################################

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
        # logger.debug(f"bgr shape:{bgr_frame.shape}, depth shape:{depth_frame.shape}")
        # rotate img
        if self.rotate is not None:
            bgr_frame = cv2.rotate(bgr_frame, self.rotate)
            # no need to rotate depth frame because we should rotate the bgr frame inversely afterwards
        
        # 2D-keypoints
        # res.keypoints.data : tensor [human, keypoints, 3 (x, y, confidence) ] float
        #                      0.00 if the keypoint does not exist
        # res.boxes.id : [human_id, ] float
        results: list[Results] = self._POSE_model.track(bgr_frame, persist=True, verbose=False)

        if not results:
            logger.debug(f"no yolo results")
            return
        
        yolo_res = results[0]
        
        # show YOLO v8 tracking image
        yolo_plot = yolo_res.plot()
        if self.rviz:
            rviz_keypoint_2d_img = bridge.cv2_to_imgmsg(yolo_plot)
            self._rviz_keypoints_img2d_pub.publish(rviz_keypoint_2d_img)
            all_marker_list = []

        if self.save:
            # video
            self.YOLO_plot = yolo_plot
        
        # 2D -> 3D ################################################################
        num_human, num_keypoints, num_dim = yolo_res.keypoints.data.shape
        id_human = yolo_res.boxes.id                    # Tensor [H]
        
        # if no detections, gets [1,0,51], yolo_res.boxes.id=None
        if id_human is None or num_keypoints == 0 or num_keypoints == 0:
            if self.no_human_debugger is False:
                logger.debug('no human or no keypoints, reset rviz')
                self.no_human_debugger = True

                self._reset_rviz(ns=self.ns)

            return
        else:
            self.no_human_debugger = False
            conf_keypoints = yolo_res.keypoints.conf.cpu().numpy()        # Tensor [H,K]
            keypoints_2d = yolo_res.keypoints.data.cpu().numpy()          # Tensor [H,K,D(x,y,conf)] [H, 17, 3]

            conf_boxes = yolo_res.boxes.conf.cpu().numpy()                # Tensor [H,]
            id_human = id_human.cpu().numpy().astype(ID_TYPE)             # Tensor [H,]

        # logger.debug(f"\ndata shape:{yolo_res.keypoints.data.shape}\nhuman ID:{id_human}")
        # logger.info("Find Keypoints")


        # TODO: warning if too many human in the dictionary
        # if len(self.human_dict) >= MAX_dict_vol:
        
        # delele missing pedestrians
        del_list = []
        for key in self.human_dict.keys():
            self.human_dict[key].missing_count += 1
            if self.human_dict[key].missing_count > MAX_MISSING:
                del_list.append(key)


        if self.current_human != list(self.human_dict.keys()):
            self.current_human = list(self.human_dict.keys())
            logger.debug(f"current human id: {self.current_human}")

        for key in del_list:
            logger.debug(f"DEL human_{key} from human_dict")
            self.human_dict[key] = None
            del self.human_dict[key]
            # python memory management system will release deleted space in the future
            
            # RVIZ ####################################################
            if self.rviz:
                logger.info(f"DELETE human id: {key} in RViz")
                delete_list = delete_human_marker(key, offset_list=[KEYPOINT_ID_OFFSET, LINE_ID_OFFSET], frame_id=self.frame_id)
                all_marker_list.extend(delete_list)


        keypoints_3d = np.zeros((num_human, num_keypoints, 3))  # [H,K,3]
        keypoints_no_Kalman = np.zeros((num_human, num_keypoints, 3)) # [H,K,3]
        keypoints_mask = np.zeros((num_human, num_keypoints),dtype=bool)   # [H,K]

        for idx, id in enumerate(id_human, start=0):
            # query or set if non-existing
            if id not in self.human_dict.keys():
                logger.info(f"ADD human id: {id}")
            self.human_dict.setdefault(id, HumanKeypointsFilter(id=id, 
                                                                gaussian_blur=self.use_gaussian_blur, 
                                                                minimal_filter=self.use_minimal_filter,
                                                                num_keypoints=num_keypoints,
                                                                ))
            self.human_dict[id].missing_count = 0       # reset missing_count of existing human

            # [K,3]
            keypoints_cam = self.human_dict[id].align_depth_with_color(keypoints_2d[idx,...], 
                                                                       depth_frame, 
                                                                       self._intrinsic_matrix, 
                                                                       rotate=self.rotate)
            keypoints_no_Kalman[idx,...] = keypoints_cam                                # [K,3]
            if self.use_kalman:
                keypoints_cam = self.human_dict[id].kalmanfilter_cam(keypoints_cam)     # [K,3]
            
            assert keypoints_cam.shape[0] == self._POSE_KEYPOINTS, 'There must be K keypoints'
            
            self.human_dict[id].keypoints_filtered = keypoints_cam

            if self.use_outlier_filter:
                # logger.debug(f"{keypoints_cam}\n{self.human_dict[id].valid_keypoints}")
                new_mask, geo_center = find_inliers(data_KD=keypoints_cam, mask_K= self.human_dict[id].valid_keypoints)
                logger.debug(geo_center)
            else:
                new_mask = self.human_dict[id].valid_keypoints
            
            keypoints_3d[idx,...] = keypoints_cam     # [K,3]
            keypoints_mask[idx,...] = new_mask
            # id_human[idx] = id
            # keypoints_xx[idx] = corelated data
        
            # RVIZ ####################################################
            if self.rviz:
                add_keypoint_marker = add_human_skeletal_keypoint_Marker(human_id= id,
                                                                         keypoint_KD= keypoints_cam, 
                                                                         keypoint_mask_K=new_mask,
                                                                         offset= KEYPOINT_ID_OFFSET,
                                                                         frame_id=self.frame_id,
                                                                         )
                add_geo_center_marker = add_human_geo_center_Marker(human_id=id,
                                                                    geo_center=geo_center,
                                                                    frame_id=self.frame_id,
                                                                    )

                add_line_marker = add_human_skeletal_line_Marker(human_id= id,
                                                                 keypoint_KD= keypoints_cam,
                                                                 keypoint_mask_K=new_mask,
                                                                 offset= LINE_ID_OFFSET,
                                                                 frame_id = self.frame_id,
                                                                 )
                
                # all_marker_list.extend([add_keypoint_marker])
                all_marker_list.extend([add_keypoint_marker, add_geo_center_marker, add_line_marker])
        
        if self.save:
            self.filtered_keypoints = keypoints_3d
            self.keypoints_mask = keypoints_mask
            self.keypoints_no_Kalman = keypoints_no_Kalman

        # Publish keypoints and markers

        # id_msg = Int32MultiArray()
        # id_msg.data = id_human.tolist()     # much faster than for loop
        
        # NOTE: numpy_msg
        # self._skeleton_human_id_pub.publish(data=id_human)           # [H,]
        # self._skeleton_mask_mat_pub.publish(data=keypoints_mask)     # [H,K]
        # self._raw_skeleton_pub.publish(data=keypoints_no_Kalman)           # [H,K,D]
        # self._filtered_skeleton_pub.publish(data=keypoints_3d)       # [H,K,D]

        # RVIZ ####################################################
        if self.rviz:
            all_marker_array = MarkerArray()
            all_marker_array.markers = all_marker_list
            self._rviz_keypoints_marker3d_pub.publish(all_marker_array)
            # logger.success("Publish MakerArray")
            
        self._reset_sub_msg()   # if no new sub msg, pause this fun



def main(args=None) -> None:

    rclpy.init(args=args)
    node = skeletal_extractor_node(compressed=COMPRESSED_TOPICS,
                                   pose_model=POSE_MODEL,
                                   use_kalman=USE_KALMAN,
                                   minimal_filter=MINIMAL_FILTER,
                                   outlier_filter=OUTLIER_FILTER,
                                   rviz=RVIZ_VIS,
                                   save=SAVE_DATA,
                                   ref_link='link_head',
                                   )
    
    logger.success(f"{SKELETON_NODE} Node initialized")
    rclpy.spin(node)
        

if __name__ == '__main__':
    # if SAVE_DATA:
    #     if not os.path.exists(DATA_DIR_PATH / "pickle"):
    #         os.mkdir(DATA_DIR_PATH / "pickle")
    main()