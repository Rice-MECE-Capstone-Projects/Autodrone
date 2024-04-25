#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge


# customized cv2_to_imgmsg to replace broken function in cv2
def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    img_msg.header.stamp = rospy.Time.now()
    return img_msg


def publisher():

  pub1 = rospy.Publisher('video_1', Image, queue_size=10)
  pub2 = rospy.Publisher('video_2', Image, queue_size=10)
  
  rospy.init_node('video_pub', anonymous=True)

  rate = rospy.Rate(30) # 10hz

  # Create a VideoCapture object
  # The argument '0' gets the default webcam.
  #cap = cv2.VideoCapture(0)
  cap1 = cv2.VideoCapture(2)
  cap2 = cv2.VideoCapture(0)

  # Used to convert between ROS and OpenCV images
  br = CvBridge()


  # While ROS is still running.
  while not rospy.is_shutdown():

      # Capture frame-by-frame
      # This method returns True/False as well
      # as the video frame.
      ret1, frame1 = cap1.read()
      ret2, frame2 = cap2.read()

      if ret1 == True:
        # Publish the image.
        # The 'cv2_to_imgmsg' method converts an OpenCV
        # image to a ROS image message
        frame1 = cv2.flip(frame1, 0)
        
        # pub1.publish(br.cv2_to_imgmsg(frame1))  #run this first, if not work, use the line blow
        pub1.publish(cv2_to_imgmsg(frame1))
        rospy.loginfo('----------Publishing Video Frame 1----------')
      if ret2 == True:
        # Publish the image.
        # The 'cv2_to_imgmsg' method converts an OpenCV
        # image to a ROS image message
        frame2 = cv2.flip(frame2, 0)
        #pub2.publish(br.cv2_to_imgmsg(frame2))  #run this first, if not work, use the line blow

        pub2.publish(cv2_to_imgmsg(frame2))
        rospy.loginfo('----------Publishing Video Frame 2----------')
      # Sleep just enough to maintain the desired rate
      rate.sleep()



if __name__ == '__main__':
  try:
    publisher()
  except rospy.ROSInterruptException:
    pass
