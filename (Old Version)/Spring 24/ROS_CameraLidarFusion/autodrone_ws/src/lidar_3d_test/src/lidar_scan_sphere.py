import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan, PointCloud, Imu
from geometry_msgs.msg import Point32
import message_filters

# PointCloud
# std_msgs/Header header
# geometry_msgs/Point32[] points
# sensor_msgs/ChannelFloat32[] channels


# LaserScan
# std_msgs/Header header
# float32 angle_min
# float32 angle_max
# float32 angle_increment
# float32 time_increment
# float32 scan_time
# float32 range_min
# float32 range_max
# float32[] ranges
# float32[] intensities

# Point32
# float32 x
# float32 ye
# float32 z


ANGLE_MIN= -3.1415927410125732
ANGLE_MAX= 3.1415927410125732
ANGLE_INCREMENT= 0.005482709966599941
TIME_INCREMENT= 0.00012854613305535167
SCAN_TIME= 0.14731386303901672
RANGE_MIN= 0.15000000596046448
RANGE_MAX= 12.0
LIDAR_POINT_SIZE = 1147



# ROS Node
class Lidar_process(object):
    def __init__(self):
        self.rate = rospy.Rate(30)
        self.lidar_pub = rospy.Publisher('lidar_2D_to_3D', PointCloud, queue_size=10)

        self.lidar_sub = message_filters.Subscriber('scan', LaserScan)
        self.imu_sub = message_filters.Subscriber('/mavros/imu/data_raw', Imu)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.lidar_sub, self.imu_sub], 1, 1)
        self.ts.registerCallback(self.callback)

        self.pointCloud_msg_cache = PointCloud()

        self.pointCloud_msg_cache_size = 100

        self.start_time = rospy.Time.now().secs

    def callback(self, lidar_msg, imu_msg):
        rospy.loginfo("Lidar Reconstruct Working!")
        lidar_msg = self.lidar_filtered(lidar_msg)
        self.pointCloud_msg_cache  = self.lidar_reconstruct(lidar_msg, imu_msg, self.pointCloud_msg_cache)


        if len(self.pointCloud_msg_cache.points) > self.pointCloud_msg_cache_size * LIDAR_POINT_SIZE:
            # Remove the first 100 elements
            self.pointCloud_msg_cache.points = self.pointCloud_msg_cache.points[LIDAR_POINT_SIZE:]
        self.lidar_pub.publish(self.pointCloud_msg_cache)


    def lidar_reconstruct(self, lidar_msg, imu_msg, msg_cache):
        msg_reconstructed = PointCloud()
        msg_reconstructed.header = lidar_msg.header

        ranges_np = np.array(lidar_msg.ranges)
        angle_list = np.arange(ANGLE_MIN, ANGLE_MIN + ANGLE_INCREMENT * len(ranges_np), ANGLE_INCREMENT)

        pitch_rad = self.calculate_pitch(imu_msg)

        cartesian_coordinates = self.spherical_to_cartesian(ranges_np, angle_list, pitch_rad)

        point_list = [Point32(x=float(point[0]), y=float(point[1]), z=float(point[2])) for point in cartesian_coordinates]
        if point_list:
            msg_cache.points.extend(point_list)
        msg_cache.header = lidar_msg.header
        return msg_cache
        # Create a PointCloud message


    def lidar_filtered(self, msg):
        ranges = msg.ranges
        ranges_np = np.array(ranges)
        intensities = ranges_np
        range_filtered = ranges_np
        # Replace inf values with 0 intensity
        intensities[np.isinf(intensities)] = 0
        range_filtered[np.isinf(range_filtered)] = 0
        intensities_tuple = tuple(intensities)
        range_filtered_tuple = tuple(range_filtered)

        msg_filtered= msg
        msg_filtered.intensities = intensities_tuple
        msg_filtered.ranges = range_filtered_tuple
        msg_filtered.header.stamp = rospy.Time.now()

        return msg_filtered

    def calculate_pitch(self, imu_msg):
        x = imu_msg.linear_acceleration.x
        y = imu_msg.linear_acceleration.y
        z = imu_msg.linear_acceleration.z

        adjacent = math.sqrt(x * x + z * z)
        pitch_rad = math.atan2(y, adjacent)
        pitch_deg = pitch_rad * 180 / math.pi
        print("Pitch Deg: ", round(pitch_deg,2))
        return pitch_rad

    def spherical_to_cartesian(self, distance, angle_xy_plane, pitch_rad):
        pitch_rad = np.pi/2 - pitch_rad  # since the pitch angle was calculated from xy-plane to the point
        x = distance * np.sin(pitch_rad) * np.cos(angle_xy_plane)
        y = distance * np.sin(pitch_rad) * np.sin(angle_xy_plane)
        z = distance * np.cos(pitch_rad)
        xyz = np.stack((x, y, z), axis=-1)

        return xyz




    # # create test lidar_msg for segmentation!!!!!!!!!!!!!!!
        # msg_test = msg
        # test_range = np.array(msg.ranges)
        # test_intensities = np.array(msg.intensities)
        #
        # test_intensities[:287] = 10
        # test_intensities[287:574] = 30
        # test_intensities[574:861] = 50
        # test_intensities[861:1146] = 70
        #
        # msg_test.intensities = tuple(test_intensities)
        # # test_range[287:] = np.inf
        # # msg_test.ranges= tuple(test_range)


if __name__ == '__main__':
    rospy.init_node("Lidar_process", anonymous=True)
    my_node = Lidar_process()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

