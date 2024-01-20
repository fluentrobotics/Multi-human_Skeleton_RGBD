# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse
import numpy as np
import cv2

import rosbag


# try: 
#     import ros_numpy
# except AttributeError:
#     import numpy as np
#     np.float = np.float64  # temp fix for following import
#     import ros_numpy


from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.engine.results import Results


model = YOLO('/home/xmo/socialnav_xmo/feature_extractor/models/yolov8m-pose.pt')



def main():
    """Extract a folder of images from a rosbag.
    """
    # parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    # parser.add_argument("bag_file", help="Input ROS bag.")
    # parser.add_argument("output_dir", help="Output directory.")
    # parser.add_argument("image_topic", help="Image topic.")

    # args = parser.parse_args()

    # print "Extract images from %s on topic %s into %s" % (args.bag_file,
    #                                                       args.image_topic, args.output_dir)
    bag_file = "/home/xmo/socialnav_xmo/feature_extractor/bagfiles/test_video.bag"
    # image_topic = "/camera/color/image_raw"
    image_topic = "/camera/color/image_raw/compressed"
    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    # depth_topic = "/camera/aligned_depth_to_color/image_raw/compressed"
    cam_info_topic = "/camera/color/camera_info"

    output_dir = "/home/xmo/bagfiles/extract/"

    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    count = 0
    img_c = 0
    dep_c = 0
    for topic, msg, t in bag.read_messages(topics=[image_topic, depth_topic]):
        # print('seq:',msg.header.seq, '      TOPIC:',topic, '     T:', t, )
        # if topic == cam_info_topic:
        #     print("camera info:", msg)

        # if topic == depth_topic:
        #     depth_frame = bridge.imgmsg_to_cv2(msg)
        #     # depth_frame = bridge.compressed_imgmsg_to_cv2(msg)
        #     print('depth:',depth_frame.shape)       # [720,1280]

        if topic == image_topic:
            # print("color")
            # cv_img = bridge.imgmsg_to_cv2(msg)
            cv_img = bridge.compressed_imgmsg_to_cv2(msg)
            # print(type(cv_img))
            # print('color:',cv_img.shape)            # [720,1280,3]

            cv_img_rotate = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
            # # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            # # print('after rotation:',cv_img.shape)            # [720,1280,3]

            # # # print(cv_img.shape)
            
            results = model.track(cv_img_rotate, verbose=False)
            res = results[0]
            if res.boxes.id is not None:
                # print("detetcted")
                print("data shape",res.keypoints.data.shape)
                keypoint0 = res.keypoints.data[0,0,...].cpu().numpy().astype(np.int16)
            # cv_img = results[0].plot()
                cv_img_rotate[keypoint0[1]-5:keypoint0[1]+5,keypoint0[0]-5:keypoint0[0]+5] = np.array([255,0,0])
                print("keypoint0", keypoint0)
                original_keypoint0 = np.array([720-keypoint0[0], keypoint0[1]])
                cv_img[original_keypoint0[0]-5:original_keypoint0[0]+5,original_keypoint0[1]-5:original_keypoint0[1]+5] = np.array([255,0,0])

                # cv_img_rotate[keypoint0[1],keypoint0[0]] = np.array([255,0,0])

            # print("cv_img", cv_img)
            # kyepoint is [508,146], in numpy array should be [146,508]
            # cv_img[146:156,508:518] = np.array([0,0,0])

            cv2.imshow('RealSense RGB',cv_img_rotate)
            cv2.imshow('Original', cv_img)
            # print(res.keypoints.data)
            key = cv2.waitKey(5)
            if key == ord('q'):
                break

            # if results[0].keypoints.shape[0] in (1,2) and results[0].keypoints.shape[1] != 0:
            #     print(res.keypoints.data.shape)
            #     print(type(res.keypoints.data[0,0,0]))
            #     break
    
        
        # cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)
        # print "Wrote image %i" % count

        count += 1

if __name__ == '__main__':
    main()



# YOLOv8 XY coordinate          Original Coordinate(before rotation)
#          720                              720-x
#       ------> x (axis0)              axis0 <------
#       |                                          | y              (720,1280)
#  1280 |                                          |
#       v y(axis1)                                 v axis1
    

# color/depth frame coordinate
# ------> axis1
# |
# |
# v axis0
