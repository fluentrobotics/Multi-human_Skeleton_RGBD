import os
import argparse

import cv2

import rosbag

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.engine.results import Results


model = YOLO('/home/xmo/socialnav_xmo/feature_extractor/models/yolov8m-pose.pt')

def main():
    """Extract a folder of images from a rosbag.
    """
    bag_file = "/home/xmo/socialnav_xmo/feature_extractor/bagfiles/multiHuman2.bag"
    image_topic = "/camera/color/image_raw/compressed"
    # output_dir = "/home/xmo/bagfiles/extract/"

    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[image_topic]):

        cv_img = bridge.compressed_imgmsg_to_cv2(msg)
        cv_img = cv2.rotate(cv_img,cv2.ROTATE_90_CLOCKWISE)

        results = model.track(cv_img, persist=True, verbose=False)
        cv_img = results[0].plot()

        cv2.imshow('RealSense RGB',cv_img)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break
        # cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)
        # print "Wrote image %i" % count

        # count += 1

    bag.close()


if __name__ == '__main__':
    main()