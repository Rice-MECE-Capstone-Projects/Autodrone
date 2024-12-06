import rospy

from std_msgs.msg import Float32

#std_msgs/Float32.msg


class Altitude(object):
    def __init__(self):
        self.rate = rospy.Rate(30)
        self.pub = rospy.Publisher('scan_radius', Float32, queue_size=10)


        while not rospy.is_shutdown():
            # Create a simple message
            message = Float32()
            fake_altitude = 100000
            message.data = fake_altitude
            # Publish the message
            self.pub.publish(message)

            # Log the published message
            rospy.loginfo("Fake altitude Set to: %f", fake_altitude)

            # Wait according to the publishing rate
            self.rate.sleep()



if __name__ == '__main__':
    rospy.init_node("fake_altitude", anonymous=True)
    my_node = Altitude()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
