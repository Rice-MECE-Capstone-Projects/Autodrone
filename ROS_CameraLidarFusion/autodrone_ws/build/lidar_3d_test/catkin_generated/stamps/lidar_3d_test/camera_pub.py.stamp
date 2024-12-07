#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Header

# customized cv2_to_imgmsg to replace broken function in cv2
def cv2_to_imgmsg(cv_image, header):
    img_msg = Image()
    img_msg.header = header
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg


def publisher():
    pub1 = rospy.Publisher('video_1', Image, queue_size=10)
    pub2 = rospy.Publisher('video_2', Image, queue_size=10)
    pub1_rev = rospy.Publisher('video_1_Reversed', Image, queue_size=10)
    pub2_rev = rospy.Publisher('video_2_Reversed', Image, queue_size=10)
    rospy.init_node('video_pub', anonymous=True)

    rate = rospy.Rate(30) # 10hz

    cap1 = cv2.VideoCapture(2)
    cap2 = cv2.VideoCapture(0)

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 7680)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 4320)

    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 7680)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 4320)
    br = CvBridge()

    # While ROS is still running.
    while not rospy.is_shutdown():

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True:
            header = Header()
            header.stamp = rospy.Time.now()

            pub1.publish(cv2_to_imgmsg(frame1), header)
            pub1_rev.publish(cv2_to_imgmsg(cv2.rotate(frame1, cv2.ROTATE_180), header))

            rospy.loginfo('----------Publishing Video Frame 1----------')
        if ret2 == True:
            header = Header()
            header.stamp = rospy.Time.now()

            pub2.publish(cv2_to_imgmsg(frame2, header))
            pub2_rev.publish(cv2_to_imgmsg(cv2.rotate(frame2, cv2.ROTATE_180), header))
            rospy.loginfo('----------Publishing Video Frame 2----------')

        # Sleep just enough to maintain the desired rate
        rate.sleep()



if __name__ == '__main__':
  try:
    publisher()
  except rospy.ROSInterruptException:
    pass
