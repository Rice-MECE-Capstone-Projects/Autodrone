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
TOTAL_LIDAR_POINTS = 1147

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
        #self.test_pub = rospy.Publisher('lidar_test', LaserScan, queue_size=10)

        self.image_sub = message_filters.Subscriber('video_marked_2', Image)
        self.lidar_sub = message_filters.Subscriber('scan_radius', LaserScan)
        self.object_sub = message_filters.Subscriber('rect_array', MyArray)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub, self.object_sub], 1, 1)  # Changed code
        self.ts.registerCallback(self.callback)

    def callback(self, img_msg, lidar_msg, object_msg):
        print("Sub1:", img_msg.header.stamp)
        print("Sub2:", lidar_msg.header.stamp)
        print("Sub3:", object_msg)

    # get index of lidar.msg.ranges based on the rad(-pi to pi)
    def get_index(self, rad, ):
        pass

    def get_angle(self):
        pass


if __name__ == '__main__':
    rospy.init_node("sensor_fusion", anonymous=True)
    my_node = Sensor_fusion()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

