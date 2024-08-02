#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class OusterImageAnalyzer:
    def __init__(self):
        self.bridge = CvBridge()
        self.reflec_image_sub = rospy.Subscriber("/ouster/reflec_image", Image, self.reflec_image_callback)
        self.range_image_sub = rospy.Subscriber("/ouster/range_image", Image, self.range_image_callback)
        self.reflec_image = None
        self.range_image = None

    def reflec_image_callback(self, msg):
        rospy.loginfo("Received reflectance image")
        self.reflec_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def range_image_callback(self, msg):
        rospy.loginfo("Received range image")
        self.range_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def find_point_and_range(self, point):
        if self.reflec_image is None or self.range_image is None:
            rospy.logwarn("Images not received yet")
            return

        x, y = point
        if 0 <= x < self.reflec_image.shape[1] and 0 <= y < self.reflec_image.shape[0]:
            reflec_value = self.reflec_image[y, x]
            range_value = self.range_image[y, x]
            rospy.loginfo(f"Point: ({x}, {y}), Reflectance: {reflec_value}, Range: {range_value}")
        else:
            rospy.logwarn("Point out of image bounds")

if __name__ == "__main__":
    rospy.init_node("ouster_image_analyzer")
    analyzer = OusterImageAnalyzer()

    # 예를 들어, 원하는 점을 (100, 100)으로 지정
    desired_point = (0, 0)

    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        analyzer.find_point_and_range(desired_point)
        rate.sleep()
