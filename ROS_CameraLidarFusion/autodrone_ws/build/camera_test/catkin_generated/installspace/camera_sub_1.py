# !/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
import sys
import numpy as np
from pyzbar.pyzbar import decode


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


class Video_processor(object):
    def __init__(self):
        self.rate = rospy.Rate(30)
        self.pub = rospy.Publisher('video_marked_1', Image, queue_size=10)
        self.video_sub = rospy.Subscriber("video_1", Image, self.callback)
        self.frame = None

    def start(self):
        while not rospy.is_shutdown():
            #print("subscribling......................")
            # if self.frame is not None:
            #     img_msg = cv2_to_imgmsg(self.frame)
            #     self.pub.publish(img_msg)
                self.rate.sleep()


    def callback(self, msg):
        rospy.loginfo("----------Image Receiving----------\n")
        self.frame = imgmsg_to_cv2(msg)  # (width, height, 3) numpy array
        frame_marked = self.track_qr_code(self.frame)

        if frame_marked is not None:
            img_msg = cv2_to_imgmsg(frame_marked)
            self.pub.publish(img_msg)

        # img_msg = cv2_to_imgmsg(frame_marked)


    def track_qr_code(self, frame):
        #frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Decode the QR code in the frame
        decoded_objs = decode(frame)
        # Draw bounding box around the detected QR code
        for obj in decoded_objs:
            cv2.rectangle(frame, obj.rect, (255, 255, 0), 5)
            # Check if the center of the QR code is in the center of the frame
            cx = obj.rect.left + obj.rect.width // 2
            cy = obj.rect.top + obj.rect.height // 2
            height, width, _ = frame.shape
            if abs(cx - width // 2) < 50 and abs(cy - height // 2) < 50:
                print("[Camera 1] QR code in the CENTER !!!")
            elif cx < frame.shape[1] // 2:
                print("[Camera 1] QR CODE is in the LEFT !!!")
                # don't have a function for rotating left
            else:
                print("[Camera 1] QR CODE is in the RIGHT !!!")
        return frame



if __name__ == '__main__':
    rospy.init_node("video_process", anonymous=True)
    my_node = Video_processor()
    my_node.start()