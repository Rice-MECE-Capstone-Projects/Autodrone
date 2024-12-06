import rospy
import numpy as np
import math
import struct
from sensor_msgs.msg import LaserScan, PointCloud, Imu, Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

from geometry_msgs.msg import Point32
import message_filters
import sys
import cv2


# PointCloud
# std_msgs/Header header
# geometry_msgs/Point32[] points
# sensor_msgs/ChannelFloat32[] channels
    # string name
    # float32[] values

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

# Camera Parameters
CAMERA_FOCAL_LENGTH = 0.004     # meters
CAMERA_SENSOR_WIDTH = 0.00358   # meters
CAMERA_SENSOR_HEIGHT = 0.00202  # meters
CAMERA_HORIZONTAL_PIXELS = 1280 # pixels
CAMERA_VERTICAL_PIXELS = 720    # pixels
CAMERA_HORIZONTAL_FOV = 49.58   # degree
CAMERA_VERTICAL_FOV = 29.06   # degree


# Lidar Parameters
ANGLE_MIN= -3.1415927410125732
ANGLE_MAX= 3.1415927410125732
ANGLE_INCREMENT= 0.005482709966599941
TIME_INCREMENT= 0.00012854613305535167
SCAN_TIME= 0.14731386303901672
RANGE_MIN= 0.15000000596046448
RANGE_MAX= 12.0
LIDAR_POINT_SIZE = 1147
LIDAR_OFFSET_DEGREE = -10
LIDAR_POSITIVE_DEGREE = CAMERA_HORIZONTAL_FOV/2   # degree
LIDAR_NEGATIVE_DEGREE = 360 - LIDAR_POSITIVE_DEGREE  # degree



# customized imgmsg_to_cv2 to replace broken function in cv2
def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv


# ROS Node
class Lidar_process(object):
    def __init__(self):
        self.rate = rospy.Rate(30)
        self.lidar_pub = rospy.Publisher('lidar_2D_to_3D_PointCloud2', PointCloud2, queue_size=10)

        self.lidar_sub = message_filters.Subscriber('scan', LaserScan)
        self.image_sub = message_filters.Subscriber('video_2_Reversed', Image)
        self.imu_sub = message_filters.Subscriber('/mavros/imu/data_raw', Imu)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.lidar_sub, self.image_sub, self.imu_sub], 1, 1)
        self.ts.registerCallback(self.callback)

        self.pointCloud_msg_cache = PointCloud()
        self.pointCloud_msg_cache_size = 100

        self.start_time = rospy.Time.now().secs


    def callback(self, lidar_msg, img_msg, imu_msg):
        rospy.loginfo("Lidar Reconstruct and Rendering Working!")
        lidar_msg = self.lidar_filtered(lidar_msg)

        self.pointCloud_msg_cache.header = lidar_msg.header

        (point_list, points_rgb) = self.lidar_reconstruct(lidar_msg, img_msg, imu_msg)
        print(points_rgb.shape)

        # pointCloud_msg = self.load_pointCloud(self.pointCloud_msg_cache, point_list)

        pointcloud2_msg = self.pointcloud_to_pointcloud2_with_rgb(self.pointCloud_msg_cache, point_list, points_rgb)

        self.lidar_pub.publish(pointcloud2_msg)



    def load_pointCloud(self, msg_cache, point_list):
        if point_list:
            msg_cache.points.extend(point_list)
            msg_cache.header.stamp = rospy.Time.now()

        # check of the datapoints reach the cache limit
        print("Points Received: ", len(point_list))
        if len(self.pointCloud_msg_cache.points) > self.pointCloud_msg_cache_size * len(point_list):
            # Remove the first "LIDAR_POINT_SIZE" elements
            self.pointCloud_msg_cache.points = self.pointCloud_msg_cache.points[LIDAR_POINT_SIZE:]

        return self.pointCloud_msg_cache

    def lidar_reconstruct(self, lidar_msg, img_msg, imu_msg):
        msg_reconstructed = PointCloud()
        msg_reconstructed.header = lidar_msg.header

        # lidar_range_merged = self.lidar_sector_remap(lidar_msg, LIDAR_POSITIVE_DEGREE, LIDAR_NEGATIVE_DEGREE)
        range_list = np.array(lidar_msg.ranges)
        angle_list = np.arange(ANGLE_MIN, ANGLE_MIN + ANGLE_INCREMENT * len(range_list), ANGLE_INCREMENT)

        pitch_rad = self.calculate_pitch(imu_msg)
        # (range_list_cut, angle_list_cut) = self.lidar_sector_remap(range_list, angle_list, LIDAR_POSITIVE_DEGREE, LIDAR_NEGATIVE_DEGREE)
        # only preserve the desired angle of range of lidar

        cartesian_coordinates = self.spherical_to_cartesian(range_list, angle_list, pitch_rad)
        cartesian_coordinates_cut = self.lidar_sector_remap(cartesian_coordinates, LIDAR_POSITIVE_DEGREE, LIDAR_NEGATIVE_DEGREE)
        point_list = [Point32(x=float(point[0]), y=float(point[1]), z=float(point[2])) for point in cartesian_coordinates_cut]

        points_rgb = self.render_pointCloud(img_msg, point_list)

        return point_list, points_rgb


    def render_pointCloud(self, img_msg, point_list):
        image = imgmsg_to_cv2(img_msg)
        center_rows = self.extract_center_rows(image, num_rows=5, v_offset=-50, h_offset = 0)

        points_rgb = self.sample_points_from_center_row(center_rows, num_points=len(point_list), neighborhood_size=5)
        # encoded_argb_array = np.apply_along_axis(lambda row: self.encode_rgb_to_single_num(row[0], row[1], row[2]), 1, points_rgb)

        print("Image received!")
        return points_rgb

    def extract_center_rows(self, image, num_rows=3, v_offset=0, h_offset=0):
        """
        Extracts a specified number of rows around the center of the image.

        Parameters:
        - image: The input image as a numpy array (shape: [height, width, channels]).
        - num_rows: The number of rows to extract around the center.

        Returns:
        - A numpy array containing the center rows of the image.
        """
        # Calculate the center row index
        center_row_index = image.shape[0] // 2

        # Calculate the start and end row indices to extract
        start_row = max(center_row_index - num_rows // 2, 0)
        end_row = min(start_row + num_rows, image.shape[0])

        # Extract the center rows
        center_rows = image[start_row + v_offset:end_row + v_offset, :image.shape[1] - h_offset, :]
        return center_rows

    def sample_points_from_center_row(self, center_rows, num_points=0, neighborhood_size=1):
        """
        Samples a specified number of points equally distributed across the center row
        of the extracted rows. For each point, it samples the RGB value from the nearby pixels.

        Parameters:
        - center_rows: The extracted center rows of the image as a numpy array.
        - num_points: The number of points to sample.
        - neighborhood_size: The size of the neighborhood to average the RGB values from.

        Returns:
        - A numpy array of shape (num_points, 3) containing the sampled RGB values.
        """
        # Initialize an array to store RGB values
        sampled_rgb_values = np.zeros((num_points, 3), dtype=int)
        row_index = center_rows.shape[0] // 2  # Use the middle row of the extracted rows
        step = center_rows.shape[1] // num_points  # Determine step size for equally distributed sampling

        for i in range(num_points):
            point_x = min(i * step + step // 2, center_rows.shape[1] - 1)  # Calculate the x-coordinate of the point

            # Calculate start and end indices for the neighborhood around the point
            start_x = max(point_x - neighborhood_size // 2, 0)
            end_x = min(start_x + neighborhood_size, center_rows.shape[1])

            # Sample and average the RGB values from the neighborhood of pixels
            rgb_average = np.mean(center_rows[row_index, start_x:end_x, :], axis=0)
            sampled_rgb_values[i, :] = rgb_average

        return sampled_rgb_values

    def pointcloud_to_pointcloud2_with_rgb(self, point_cloud, point_list, rgb_list):
        """
        Convert a PointCloud to PointCloud2 message, including RGB values for each point.

        :param point_cloud: PointCloud message.
        :param rgb_values: Numpy array of shape (number of points, 3) with RGB values.
        :return: PointCloud2 message.
        """
        # Create PointCloud2 message header
        header = point_cloud.header

        # Define fields for PointCloud2 with x, y, z, and rgb
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        # Initialize empty list to hold point cloud data for PointCloud2
        cloud_data = point_cloud.points

        for i, point in enumerate(point_list):
            # Extract RGB values and pack into a uint32
            r, g, b = rgb_list[i].astype(np.uint8)
            # rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
            rgb = struct.unpack('I', struct.pack('BBBB', r, g, b, 255))[0]

            # Append (x, y, z, rgb) to cloud_data
            cloud_data.append([point.x, point.y, point.z, rgb])

        if len(cloud_data) > self.pointCloud_msg_cache_size * len(point_list):
            # Remove the first number of "LIDAR_POINT_SIZE" elements if reach limits
            cloud_data = cloud_data[LIDAR_POINT_SIZE:]

        # Create PointCloud2 message
        pc2_msg = pc2.create_cloud(header, fields, cloud_data)

        return pc2_msg


    def lidar_sector_remap(self, range_list, positive_deg, negative_deg):
        idx_positive = self.get_lidar_index(positive_deg)
        idx_negative = self.get_lidar_index(negative_deg)

        range_positive = range_list[:idx_positive]
        range_negative = range_list[idx_negative:]
        range_merged = np.concatenate((range_negative, range_positive))
        range_merged = range_merged[::-1]  # change to left to right corresponds to the image
        range_merged = np.array(range_merged)
        # range_merged[np.isinf(range_merged)] = -1

        # angle_positive = angle_list[:idx_positive]
        # angle_negative = angle_list[idx_negative:]
        # angle_merged = angle_negative + angle_positive
        # angle_merged = angle_merged[::-1]  # change to left to right corresponds to the image
        # angle_merged = np.array(angle_merged)
        # angle_merged[np.isinf(angle_merged)] = -1

        return range_merged

    # get index of lidar.msg.ranges by degree(0 to 360), 0 and 360 deg points to negative x-axis, counterclockwise
    def get_lidar_index(self, deg):
        index = round(LIDAR_POINT_SIZE / 360 * deg)  # round to the closest int
        return index


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


    def encode_rgb_to_single_num(self, R, G, B):
        encoded_rgb= (R << 16) | (G << 8) | B
        return encoded_rgb



if __name__ == '__main__':
    rospy.init_node("Lidar_process", anonymous=True)
    my_node = Lidar_process()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

