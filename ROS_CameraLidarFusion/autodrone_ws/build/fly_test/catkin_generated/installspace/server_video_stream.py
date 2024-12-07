import cv2
import socket
import pickle
import struct
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rosbag


class VideoStreamer:
    def __init__(self, host_ip, port):
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host_ip = host_ip  # Server's IP address
        self.port = port  # Port to listen on
        self.socket_address = (self.host_ip, self.port)
        
        # Bind and listen
        self.server_socket.bind(self.socket_address)
        self.server_socket.listen(5)
        print("Listening at:", self.socket_address)
        # Accept connection
        self.client_socket, self.addr = self.server_socket.accept()
        print('Got Connection from:', self.addr)

        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber("video_2", Image, self.callback, queue_size=1)
        
    def callback(self, data):
        try:
            # Convert from ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        # Serialize frame
        data = pickle.dumps(cv_image)
        # Send message length first
        message = struct.pack("Q", len(data)) + data
        # Then data
        self.client_socket.sendall(message)
        rospy.loginfo('----------Video Streaming----------')

            
            

if __name__ == '__main__':
    rospy.init_node('video_streamer_node', anonymous=True)
    ip_host = '168.5.171.218'
    port = 8000
    vs = VideoStreamer(ip_host, port)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        vs.client_socket.close()
        rospy.signal_shutdown('ROS shutdown...')
        print("Shutting down")

