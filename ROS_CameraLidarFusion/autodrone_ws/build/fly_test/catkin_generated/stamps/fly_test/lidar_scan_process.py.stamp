import rospy
from sensor_msgs.msg import LaserScan
import numpy as np


# ROS Node
class Lidar_process(object):
    def __init__(self):
        self.rate = rospy.Rate(30)
        self.lidar_pub = rospy.Publisher('lidar_scan', LaserScan, queue_size=10)
        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.callback)

    def callback(self, msg):
        self.lidar_intensity_remap(msg)
        print("Lidar Points Publishing......")
        
    def lidar_intensity_remap(self, msg):
        ranges = msg.ranges
        ranges_np = np.array(ranges)
        intensities = ranges_np

        # Replace inf values with 0 intensity
        intensities[np.isinf(intensities)] = 0
        intensities_tuple = tuple(intensities)

        msg_w_intensity = msg
        msg_w_intensity.intensities = intensities_tuple
        msg_w_intensity.header.stamp = rospy.Time.now()
        self.lidar_pub.publish(msg_w_intensity)

if __name__ == '__main__':
    rospy.init_node("Lidar_Process", anonymous=True)
    my_node = Lidar_process()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

