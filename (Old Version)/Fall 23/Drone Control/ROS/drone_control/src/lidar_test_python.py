import rospy
from sensor_msgs.msg import LaserScan
import numpy as np


# ROS Node
class Lidar_process(object):
    def __init__(self):
        self.rate = rospy.Rate(10)
        self.lidar_pub = rospy.Publisher('scan_test', LaserScan, queue_size=10)
        self.lidar_sub = rospy.Subscriber('spur/laser/scan', LaserScan, self.callback)



    def callback(self, msg):
        self.lidar_intensity_remap(msg)

    def lidar_intensity_remap(self, msg):
        ranges = msg.ranges
        ranges_np = np.array(ranges)
        intensities = ranges_np
        
        # Replace inf values with 0 intensity
        intensities[np.isinf(intensities)] = 0
        # intensities_tuple = tuple(intensities)
        
        # msg_w_intensity = msg
        # msg_w_intensity.intensities = intensities_tuple
        # msg_w_intensity.header.stamp = rospy.Time.now()
        # self.lidar_pub.publish(msg_w_intensity)

        # create test lidar_msg for segmentation!!!!!!!!!!!!!!!
        msg_test = msg
        test_range = np.array(msg.ranges)
        test_intensities = np.array(msg.intensities)
        
        # test_intensities[:250] = 10
        # test_intensities[251:500] = 30
        # test_intensities[501:750] = 50
        # test_intensities[751:] = 70
        
        msg_test.intensities = tuple(test_intensities)

        msg_test = msg
        test_range = np.array(msg.ranges)

        # test_range[100:] = np.inf       # Quadrant 3
        test_range[:231] = test_range[281:] = np.inf  # Quadrant 4
        #test_range[:500] = test_range[750:] = np.inf  # Quadrant 1
        # test_range[751:] = np.inf       # Quadrant 2	
        
        msg_test.ranges= tuple(test_range)
        
        header_new = msg.header
        header_new.stamp = rospy.Time.now()
        msg_test.header = header_new
        
        self.lidar_pub.publish(msg_test)
        print("Lidar Test Message Publishing!")



if __name__ == '__main__':
    rospy.init_node("Lidar_Process", anonymous=True)
    my_node = Lidar_process()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")