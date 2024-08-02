#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d

class ReflectanceAnalyzer:
    def __init__(self):
        self.bridge = CvBridge()
        self.reflec_image_sub = rospy.Subscriber("/ouster/reflec_image", Image, self.reflec_image_callback)
        self.range_image_sub = rospy.Subscriber("/ouster/range_image", Image, self.range_image_callback)
        self.reflec_image = None
        self.range_image = None
        self.low_reflectance_threshold = 50000  # 임계값 설정 (필요에 따라 조정)

    def reflec_image_callback(self, msg):
        rospy.loginfo("Received reflectance image")
        self.reflec_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.process_images()

    def range_image_callback(self, msg):
        rospy.loginfo("Received range image")
        self.range_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.process_images()

    def process_images(self):
        if self.reflec_image is None or self.range_image is None:
            rospy.logwarn("Images not received yet")
            return

        points = []
        height, width = self.reflec_image.shape

        for y in range(height):
            for x in range(width):
                reflec_value = self.reflec_image[y, x]
                if reflec_value < self.low_reflectance_threshold:
                    range_value = self.range_image[y, x]
                    if range_value > 0:  # 유효한 범위 값 확인
                        point = [x, y, range_value]
                        points.append(point)

        rospy.loginfo(f"Number of points with low reflectance: {len(points)}")

        if points:
            self.save_and_visualize_ply(points)

    def save_and_visualize_ply(self, points):
        ply_points = []
        for p in points:
            ply_points.append([p[0], p[1], p[2]])

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ply_points)

        # Save to PLY file
        output_path = "./low_reflectance_points.ply"
        o3d.io.write_point_cloud(output_path, pcd)
        rospy.loginfo(f"Saved PLY file with {len(points)} points to {output_path}")

        # Visualize point cloud
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    rospy.init_node("reflectance_analyzer")
    analyzer = ReflectanceAnalyzer()
    rospy.spin()

