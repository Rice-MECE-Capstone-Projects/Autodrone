# !/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
import numpy as np
import message_filters


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
def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg


class Video_processor():
    def __init__(self):
        rospy.init_node('video_process', anonymous=True)
        self.rate = rospy.Rate(30)

        self.pub1 = rospy.Publisher('video_marked_1', Image, queue_size=10)
        self.pub2 = rospy.Publisher('video_marked_2', Image, queue_size=10)

        self.video_sub_1 = message_filters.Subscriber("video_1", Image, callback)
        self.video_sub_2 = message_filters.Subscriber("video_2", Image, callback)

        ts = message_filters.TimeSynchronizer([self.video_sub_1, self.video_sub_2], 10)
        ts.registerCallback(callback)
        rospy.spin()

def callback(msg):
    rospy.loginfo(rospy.get_caller_id() + "----------Image Receiving----------")
    frame = imgmsg_to_cv2(msg)  # (width, height, 3) numpy array
    frame_processed = tracking_algorithm(frame)


def tracking_algorithm(frame):
    pass




if __name__ == '__main__':
    listener()