#!/usr/bin/env python3

import rospy
import rosbag
from sensor_msgs.msg import Image, LaserScan
import cv2
import numpy as np
import message_filters


class VideoToBag:
    def __init__(self, topic_name_image, topic_name_lidar, bag_file_name):
        self.topic_name_image = topic_name_image
        self.topic_name_lidar = topic_name_lidar
        
        self.bag = rosbag.Bag(bag_file_name, 'w')
        self.image_sub = message_filters.Subscriber(topic_name_image, Image)
        self.lidar_sub = message_filters.Subscriber(topic_name_lidar, LaserScan)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub], 1, 1)
        self.ts.registerCallback(self.callback)
        
        
    def callback(self, image_msg, lidar_msg):
        # Write the image message to the bag file
        self.bag.write(self.topic_name_image, image_msg)
        self.bag.write(self.topic_name_lidar, lidar_msg)
        print("Capturing video to ROS bag. Press Ctrl+C to stop and save the bag file.")
        
    def close_bag(self):
        self.bag.close()

if __name__ == '__main__':
    rospy.init_node('collect_to_rosbag_node', anonymous=True)
    topic_name_image = 'video_2'  
    topic_name_lidar = 'lidar_scan'  
    bag_file_name = 'camera_lidar.bag'  # The name of the bag file to write to

    video_to_bag = VideoToBag(topic_name_image, topic_name_lidar, bag_file_name)
    rospy.spin()  # Keep the program alive until Ctrl+C is pressed

    video_to_bag.close_bag()
    print("Bag file saved.")



