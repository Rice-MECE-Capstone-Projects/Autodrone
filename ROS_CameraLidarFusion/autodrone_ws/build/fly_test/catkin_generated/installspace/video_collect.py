#!/usr/bin/env python3.11
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rosbag


class VideoSaver:
    def __init__(self, filePath):
        self.bridge = CvBridge()
        self.video_writer = cv2.VideoWriter(filePath, cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (640, 480))
        self.subscriber = rospy.Subscriber("video_2", Image, self.callback, queue_size=1)
        
    def callback(self, data):
        try:
            # Convert from ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        # Write the OpenCV image to a video file
        self.video_writer.write(cv_image)
        
        # Display the resulting frame (optional)
        cv2.imshow('frame', cv_image)
        rospy.loginfo('----------Video Recording----------')
        if cv2.waitKey(3) & 0xFF == ord('q'):
            self.save_and_close()
            print("Shutting down!")
            rospy.signal_shutdown('ROS shutdown...')
            
    def save_and_close(self):
        # When everything done, release the video capture and video write objects
        self.video_writer.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    rospy.init_node('video_saver_node', anonymous=True)
    vs = VideoSaver('camera_1.mp4')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.signal_shutdown('ROS shutdown...')
        print("Shutting down")


