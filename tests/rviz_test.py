# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse
import numpy as np
import cv2
import rospy
import rosbag

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
from cv_bridge import CvBridge

from feature_extractor.utils import get_pose_model_dir, logger

model = YOLO('/home/xmo/socialnav_xmo/feature_extractor/models/yolov8m-pose.pt')

COLOR_FRAME_TOPIC = '/camera/color/image_raw'
COLOR_COMPRESSED_FRAME_TOPIC = '/camera/color/image_raw/compressed'
DEPTH_ALIGNED_TOPIC = '/camera/aligned_depth_to_color/image_raw'
DEPTH_ALIGNED_COMPRESSED_TOPIC = '/camera/aligned_depth_to_color/image_raw/compressed'

bridge = CvBridge()

class rviz_test_node:
    def __init__(self) -> None:
        self.img_sub = rospy.Subscriber(COLOR_COMPRESSED_FRAME_TOPIC, CompressedImage, self._img_sub_callback)
        self.original_pub = rospy.Publisher("/rviz_test/raw/color", Image, queue_size=1)
        self.keypoint_pub = rospy.Publisher("/rviz_test/compressed/keypoint_img", Image, queue_size=1)
        self.img = None

    def _img_sub_callback(self, img_msg):
        self.img = img_msg

    def pub_test(self):
        if self.img is None:
            return
        img_mat = bridge.compressed_imgmsg_to_cv2(self.img)
        original_img = bridge.cv2_to_imgmsg(img_mat)
        self.original_pub.publish(original_img)
        # logger.debug("pub original")
        results = model.predict(img_mat, verbose=False)
        res_mat = results[0].plot()
        res_comp_img = bridge.cv2_to_imgmsg(res_mat)
        self.keypoint_pub.publish(res_comp_img)
        # logger.debug("pub keypoint")

def main():
    rospy.init_node("rviz_test")
    node = rviz_test_node()
    
    while not rospy.is_shutdown():
        node.pub_test()
        rospy.sleep(0.1)

if __name__ == '__main__':
    main()