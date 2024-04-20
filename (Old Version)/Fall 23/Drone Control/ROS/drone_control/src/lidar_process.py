import rospy
from sensor_msgs.msg import LaserScan
import numpy as np


negative_y_lidar_index = 256
detect_range = 20
wall_threshold = 0.5  # meter

# ROS Node
class Lidar_process(object):
    def __init__(self):
        self.rate = rospy.Rate(10)
        self.move_pub = rospy.Publisher('move_to_make', LaserScan, queue_size=10)
        self.lidar_sub = rospy.Subscriber('spur/laser/scan', LaserScan, self.callback)



    def callback(self, msg):
        ranges = msg.ranges
        ranges_np = np.array(ranges)
 
        range_selected = ranges_np[negative_y_lidar_index - detect_range : negative_y_lidar_index + detect_range]
        # range_selected[np.isinf(range_selected)] = -1   #label inf value
        result = np.all(range_selected > wall_threshold)
        print(result)
        msg_move = msg
        msg_move.ranges = []
        

        header_new = msg.header
        header_new.stamp = rospy.Time.now()
        msg_move.header = header_new
        if result:
            msg_move.angle_min = 1
        else:
            msg_move.angle_min = 0

        self.move_pub.publish(msg_move)
        print("Motion Message Publishing!")





if __name__ == '__main__':
    rospy.init_node("Lidar_Process", anonymous=True)
    my_node = Lidar_process()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")