import rospy
from sensor_msgs.msg import Image, LaserScan
import cv2
import sys
import numpy as np
import message_filters
from fusion_sensor.msg import MyArray
import std_msgs.msg

# LIDAR angle from -pi to pi, increment 0.0054827, 1147 data points total
TF_HORIZONTAL_OFFSET = -0.045   # meters   correspond LIDAR coordinate x-axis is positive
TF_VERTICAL_OFFSET = -0.05      # meters   correspond LIDAR coordinate z-axis is positive

CAMERA_FOCAL_LENGTH = 0.004     # meters
CAMERA_SENSOR_WIDTH = 0.00358   # meters
CAMERA_SENSOR_HEIGHT = 0.00202  # meters
CAMERA_HORIZONTAL_PIXELS = 1280 # pixels
CAMERA_VERTICAL_PIXELS = 720    # pixels
CAMERA_HORIZONTAL_FOV = 58.42   # degree
CAMERA_VERTICAL_FOV = 33.06     # degree

TOTAL_LIDAR_POINTS = 1147
LIDAR_POSITIVE_DEGREE = 20      # degree
LIDAR_NEGATIVE_DEGREE = 340     # degree


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

# customized cv2_to_imgmsg to replace broken function in cv2
def cv2_to_imgmsg(cv_image, header):
    img_msg = Image()
    img_msg.header = header
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height 
    return img_msg


# ROS Node
class Sensor_fusion(object):
    def __init__(self):
        self.rate = rospy.Rate(30)
        self.lidar_pub = rospy.Publisher('scan_distance', LaserScan, queue_size=10)
        self.fusion_pub = rospy.Publisher('fusion', Image, queue_size=10)
        #self.test_pub = rospy.Publisher('lidar_test', LaserScan, queue_size=10)

        self.image_sub = message_filters.Subscriber('video_2', Image)
        self.lidar_sub = message_filters.Subscriber('scan_radius', LaserScan)
        # self.object_sub = message_filters.Subscriber('rect_array', MyArray)
        # self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub, self.object_sub], 1, 1)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub], 1, 1)
        self.ts.registerCallback(self.callback)

    def callback(self, img_msg, lidar_msg):  #object_msg
        print("Sub1:", img_msg.header.stamp)
        print("Sub2:", lidar_msg.header.stamp)
        # print("Sub3:", object_msg)

        image_draw = self.project_points(img_msg, lidar_msg)
        header = img_msg.header
        header.stamp = rospy.Time.now()
        img_draw_msg = cv2_to_imgmsg(image_draw, header)
        self.fusion_pub.publish(img_draw_msg)

    def project_points(self, img_msg, lidar_msg):
        image = imgmsg_to_cv2(img_msg)
        lidar_points_merged = self.lidar_sector_remap(lidar_msg, LIDAR_POSITIVE_DEGREE, LIDAR_NEGATIVE_DEGREE)
        image_draw = self.draw_lidar_points_on_image(image, lidar_points_merged)

        return image_draw

    def lidar_sector_remap(self, lidar_msg, positive_deg, negative_deg):
        idx_positive = self.get_lidar_index(positive_deg)
        idx_negative = self.get_lidar_index(negative_deg)
        lidar_points_positive= lidar_msg.ranges[:idx_positive]

        lidar_points_negative = lidar_msg.ranges[idx_negative:]
        lidar_points_merged = lidar_points_negative + lidar_points_positive
        lidar_points_merged = lidar_points_merged[::-1]  # change to left to right corresponds to the image
        lidar_points_merged = np.array(lidar_points_merged)
        lidar_points_merged[np.isinf(lidar_points_merged)] = -1
        return lidar_points_merged

    # get index of lidar.msg.ranges by degree(0 to 360), 0 and 360 deg points to negative x-axis, counterclockwise
    def get_lidar_index(self, deg):
        index = round(TOTAL_LIDAR_POINTS / 360 * deg)  # round to the closest int
        return index

    def get_lidar_angle(self, index):
        deg = round(index / TOTAL_LIDAR_POINTS * 360, 2)  # round to two decimal
        return deg

    def camera_to_lidar_angle_map(self, deg):
        pass

    def draw_lidar_points_on_image(self, image, lidar_points):
        width = image.shape[1]
        height = image.shape[0]
        gap = int(width / (len(lidar_points) - 1))
        dot_radius = 5
        max_distance = max(lidar_points)
        min_distance = min(lidar_points)
        for i in range(0, len(lidar_points)):
            if lidar_points[i] >= 0:
                center = (gap*i, height // 2)
                dot_color = self.distance_color_map(lidar_points[i], max_distance, min_distance)
                cv2.circle(image, center, dot_radius, dot_color, -1)

        # draw center point of the image with distance shown
        center_point_distance = self.get_center_distance(lidar_points, 0.1)
        if center_point_distance >= 0:
            dot_color = self.distance_color_map(center_point_distance, max_distance, min_distance)
            center_point_location = (width // 2, height // 2)
            cv2.circle(image, center_point_location , dot_radius, dot_color, -1)

            # Specify the text and font settings
            text = str(center_point_distance) + 'm'
            print(text)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2

            # Get the size of the text box
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            # Calculate the position for the lower right corner
            text_position = (center_point_location[0] - text_size[0] // 2, center_point_location[1] + text_size[1] + 20)

            # Draw the text
            cv2.putText(image, text, text_position, font, font_scale, dot_color, font_thickness, cv2.LINE_AA)
        return image

    def distance_color_map(self, value, max_value, min_value):
        # Normalize the value to the range [0, 1]
        normalized_value = (value - min_value) / (max_value - min_value)

        # Map the normalized value to an RGB color
        red = int(255 * normalized_value)
        blue = int(255 * (1 - normalized_value))
        green = 0

        # Return the RGB tuple
        return red, green, blue

    def get_center_distance(self, lidar_points, percentage):  # percentage of center points, 0.1 for 10% center points
        points_num = len(lidar_points)
        points_width = round(points_num * percentage)
        mid_idx = int(points_num // 2)
        center_points = []
        for i in range(points_width):
            center_points.append(lidar_points[mid_idx - (points_width // 2) + i])

        center_points = [x for x in center_points if x >= 0]
        if len(center_points) > 0:
            center_point_distance = round(sum(center_points) / len(center_points), 2)
        else:
            center_point_distance = -1
        return center_point_distance




if __name__ == '__main__':
    rospy.init_node("sensor_fusion", anonymous=True)
    my_node = Sensor_fusion()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

