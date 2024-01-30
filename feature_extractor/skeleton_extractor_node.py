"""Extract images and publish skeletons.
"""
import os
import argparse
import cv2
import pickle
import numpy as np
from typing import Optional, TypedDict
from pathlib import Path

import rosbag
import rospy
import message_filters
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Header, ColorRGBA, MultiArrayDimension
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from feature_extractor.HumanKeypointsFilter import HumanKeypointsFilter
from feature_extractor.utils import *

# Global Config
from feature_extractor.config import *
# Local Config
bridge = CvBridge()




class skeletal_extractor_node():
    def __init__(self, 
                 rotate: int = cv2.ROTATE_90_CLOCKWISE, 
                 compressed: dict = {'rgb': True, 'depth': False}, 
                 syn: bool = False, 
                 pose_model: str = 'yolov8m-pose.pt',
                 use_kalman: bool = True,
                 gaussian_blur: bool = True,
                 minimal_filter: bool = True,
                 rviz: bool = True,
                 save: bool = True,
    ):
        """
        @rotate: default =  cv2.ROTATE_90_CLOCKWISE
        @compressed: default = {'rgb': True, 'depth': False}
        @syn: syncronization, default = False
        @pose_model: default = 'yolov8m-pose.pt'
        @use_kalman: Kalman Filter, default = True
        @rviz: RViz to show skeletal keypoints and lines
        """
        self.rotate = rotate        # rotate: rotate for yolov8 then rotate inversely   
        # ROTATE_90_CLOCKWISE=0, ROTATE_180=1, ROTATE_90_COUNTERCLOCKWISE=2 
        self.compressed = compressed
        self.syn = syn
        self.rviz = rviz
        self.use_kalman = use_kalman
        self.use_gaussian_blur = gaussian_blur
        self.use_minimal_filter = minimal_filter
        self.save = save

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
                    ")
        
        # yolov8n-pose, yolov8s-pose, yolov8m-pose, yolov8l-pose, yolov8x-pose, yolov8x-pose-p6
        self._POSE_model = YOLO(get_pose_model_dir() / pose_model)
        self._POSE_KEYPOINTS = 17
        self.human_dict = dict()
        self.keypoints_HKD: np.ndarray


        # Subscriber ##########################################################
        if self.compressed['rgb']:
            self._rgb_sub = rospy.Subscriber(
                COLOR_COMPRESSED_FRAME_TOPIC, CompressedImage, self._rgb_callback, queue_size=1
            )
            self._rgb_msg: CompressedImage = None
        else:
            self._rgb_sub = rospy.Subscriber(
                COLOR_FRAME_TOPIC, Image, self._rgb_callback, queue_size=1
            )
            self._rgb_msg: Image = None
        if self.compressed['depth']:
            self._depth_sub = rospy.Subscriber(
            DEPTH_ALIGNED_COMPRESSED_TOPIC, CompressedImage, self._depth_callback, queue_size=1
        )
            self._depth_msg: CompressedImage = None
            
        else:
            self._depth_sub = rospy.Subscriber(
            DEPTH_ALIGNED_TOPIC, Image, self._depth_callback, queue_size=1
        )
            self._depth_msg: Image = None
            
        
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


        # RViz ####################################################
        if self.rviz:
            self._rviz_keypoints_img2d_pub = rospy.Publisher(
                RVIZ_IMG2D_SKELETON_TOPIC, Image, queue_size=1
            )
            self._rviz_keypoints_marker3d_pub = rospy.Publisher(
                RVIZ_MARKER3D_SKELETON_TOPIC, MarkerArray, queue_size=1
            )
        
        if self.save:
            # save keypoints
            self.filtered_keypoints = None
            self.keypoints_mask = None
            self.keypoints_no_Kalman= None
            # save YOLO plot
            self.video_img = None

        # #####################################################################
        
        
        # Camera Calibration Info ################
        camera_info_msg: CameraInfo = rospy.wait_for_message(
            CAMERA_INFO_TOPIC, CameraInfo, timeout=30
        )
        self._intrinsic_matrix = np.array(camera_info_msg.K).reshape(3,3)
        logger.info(f"\nCamera Intrinsic Matrix:\n{self._intrinsic_matrix}")
        # ########################################
        


    def _rgb_callback(self, msg: Image | CompressedImage) -> None:
        self._rgb_msg = msg

    def _depth_callback(self, msg: Image | CompressedImage) -> None:
        self._depth_msg = msg

    def _reset_sub_msg(self):
        self._rgb_msg = None
        self._depth_msg = None

    def _skeleton_inference_and_publish(self) -> None:

        if self.save:
            self.filtered_keypoints = None
            self.keypoints_mask = None
            self.keypoints_no_Kalman= None
            self.YOLO_plot = None
        
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
        if num_keypoints == 0 or num_keypoints == 0:
            return
        
        id_human = yolo_res.boxes.id                    # Tensor [H]
        
        # if no detections, gets [1,0,51], yolo_res.boxes.id=None
        if id_human is None:
            return
        # logger.debug(f"\ndata shape:{yolo_res.keypoints.data.shape}\nhuman ID:{id_human}")
        # logger.info("Find Keypoints")
        
        conf_keypoints = yolo_res.keypoints.conf.cpu().numpy()        # Tensor [H,K]
        keypoints_2d = yolo_res.keypoints.data.cpu().numpy()          # Tensor [H,K,D(x,y,conf)] [H, 17, 3]

        conf_boxes = yolo_res.boxes.conf.cpu().numpy()                # Tensor [H,]
        id_human = id_human.cpu().numpy().astype(ID_TYPE)             # Tensor


        # TODO: warning if too many human in the dictionary
        # if len(self.human_dict) >= MAX_dict_vol:

        
        # delele missing pedestrians
        for key, value in list(self.human_dict.items()):
            value.missing_count += 1
            if value.missing_count > MAX_MISSING:
                self.human_dict[key] = None
                del self.human_dict[key]
                # python memory management system will release deleted space in the future
                
                # RVIZ ####################################################
                if self.rviz:
                    delete_list = delete_human_marker(key, offset_list=[KEYPOINT_ID_OFFSET, LINE_ID_OFFSET])
                    all_marker_list.extend(delete_list)


        keypoints_3d = np.zeros((num_human, num_keypoints, 3))  # [H,K,3]
        keypoints_no_Kalman = np.zeros((num_human, num_keypoints, 3)) # [H,K,3]
        keypoints_mask = np.zeros((num_human, num_keypoints),dtype=bool)   # [H,K]

        for idx, id in enumerate(id_human, start=0):
            # query or set if non-existing
            self.human_dict.setdefault(id, HumanKeypointsFilter(id=id, 
                                                                gaussian_blur=self.use_gaussian_blur, 
                                                                minimal_filter=self.use_minimal_filter))
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

            keypoints_3d[idx,...] = keypoints_cam     # [K,3]
            keypoints_mask[idx,...] = self.human_dict[id].valid_keypoints
            # id_human[idx] = id
            # keypoints_xx[idx] = corelated data
        
            # RVIZ ####################################################
            if self.rviz:
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
        
        if self.save:
            self.filtered_keypoints = keypoints_3d
            self.keypoints_mask = keypoints_mask
            self.keypoints_no_Kalman = keypoints_no_Kalman

        # Publish keypoints and markers

        # id_msg = Int32MultiArray()
        # id_msg.data = id_human.tolist()     # much faster than for loop
        
        # NOTE: numpy_msg
        self._skeleton_human_id_pub.publish(data=id_human)           # [H,]
        self._skeleton_mask_mat_pub.publish(data=keypoints_mask)     # [H,K]
        self._raw_skeleton_pub.publish(data=keypoints_no_Kalman)           # [H,K,D]
        self._filtered_skeleton_pub.publish(data=keypoints_3d)       # [H,K,D]

        # RVIZ ####################################################
        if self.rviz:
            all_marker_array = MarkerArray()
            all_marker_array.markers = all_marker_list
            self._rviz_keypoints_marker3d_pub.publish(all_marker_array)
            logger.success("Publish MakerArray")
            
        self._reset_sub_msg()   # if no new sub msg, pause this fun

##############################################################################################
# RVIZ visualization function ################################################################
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
        
##############################################################################################
##############################################################################################



def main() -> None:

    rospy.init_node(SKELETON_NODE)
    node = skeletal_extractor_node(compressed=COMPRESSED_TOPICS,
                                   pose_model=POSE_MODEL,
                                   use_kalman=USE_KALMAN, 
                                   minimal_filter=MINIMAL_FILTER,
                                   rviz=RVIZ_VIS,
                                   save=SAVE_DATA,
                                   )
    
    logger.success("Skeleton Node initialized")

    # record data
    keypoints_list = []
    mask_list = []
    keypoints_no_Kalman_list = []
    YOLO_plot_list = []

    while not rospy.is_shutdown():
        
        node._skeleton_inference_and_publish()

        if node.save:
            # record data
            keypoints_list.append(node.filtered_keypoints)                  # List[np.ndarray]
            mask_list.append(node.keypoints_mask)                           # List[np.ndarray]
            keypoints_no_Kalman_list.append(node.keypoints_no_Kalman)       # List[np.ndarray]
            YOLO_plot_list.append(node.YOLO_plot)                           # List[np.ndarray]
            # logger.info(f"recording length: {len(mask_list)}")
        
        # TODO: manage publisher frequency
        rospy.sleep(1/PUB_FREQ)

    logger.success("rospy shutdown")

    if node.save:
        pickle_path = DATA_DIR_PATH / "pickle" / TEST_NAME
        pickle_path = pickle_path.absolute().as_posix() + "_keypoints.pkl"
        
        # numpy ndarray
        file = open(pickle_path, 'wb')
        pickle.dump(keypoints_list, file)
        pickle.dump(mask_list, file)
        pickle.dump(keypoints_no_Kalman_list, file)
        file.close()
        
        logger.info(f"Data length: {len(mask_list)}")
        logger.success("Save Keypoints Array successfully!")


    if SAVE_YOLO_IMG:
        pickle_path = DATA_DIR_PATH / "pickle" / TEST_NAME
        pickle_path = pickle_path.absolute().as_posix() + "_YOLOimgs.pkl"

        file = open(pickle_path, 'wb')
        pickle.dump(YOLO_plot_list, file)
        file.close()
        logger.info(f"Imgs length: {len(YOLO_plot_list)}")
        logger.success("Save YOLO Imgs successfully!")

        # # video
        # video_path = DATA_DIR_PATH / "video" / TEST_NAME
        # video_path = video_path.absolute().as_posix() + ".mp4"
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # height, width, layers = node.video_img_list[0].shape
        # size = (width,height)
        # out = cv2.VideoWriter(filename=video_path,
        #                       fourcc=fourcc,
        #                       fps=PUB_FREQ,
        #                       frameSize=size,
        #                       )
        # for img in node.video_img_list:
        #     out.write(img)

        # cv2.destroyAllWindows()
        # out.release()
        # logger.success("Save Video successfully!")
        

if __name__ == '__main__':
    if not os.path.exists(DATA_DIR_PATH / "pickle"):
        os.mkdir(DATA_DIR_PATH / "pickle")
    main()