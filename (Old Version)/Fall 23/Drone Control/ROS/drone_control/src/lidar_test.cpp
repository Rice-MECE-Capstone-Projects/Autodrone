#include <gnc_functions.hpp>
#include <sensor_msgs/LaserScan.h>
//include API 



void callback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
    float lidar_points = msg->ranges;
	float lidar_points_100 = lidar_points[:100];
	sensor_msgs::LaserScan::ConstPtr& new_msg = msg
	new_msg->ranges = lidar_points_100
	lidar_pub.publish(new_msg)

}

int main(int argc, char** argv)
{
	//initialize ros 
	ros::init(argc, argv, "gnc_node");
	ros::NodeHandle gnc_node("~");
	
	//initialize control publisher/subscribers
	init_publisher_subscriber(gnc_node);

  	// wait for FCU connection
	wait4connect();

	//wait for used to switch to mode GUIDED
	wait4start();

	//create local reference frame 
	initialize_local_frame();

	//request takeoff
	takeoff(1.5);
    
    // set initial speed
    set_speed(0.5)  //0.5 meters per second

	ros::Publisher lidar_pub = gnc_node.advertise<ensor_msgs::LaserScan>("/lidar_test", 10);
	ros::Subscriber lidar_sub = gnc_node.subscribe("/spur/laser/scan ", 10, callback);

	//specify control loop rate. We recommend a low frequency to not over load the FCU with messages. Too many messages will cause the drone to be sluggish
	ros::Rate rate(5.0);
	while(ros::ok())
	{
      
        

		ros::spinOnce();
		rate.sleep();
		
	}
	return 0;
}