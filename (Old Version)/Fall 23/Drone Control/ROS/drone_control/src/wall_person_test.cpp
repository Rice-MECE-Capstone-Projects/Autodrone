#include <gnc_functions.hpp>
#include <sensor_msgs/LaserScan.h>
//include API 

float motion_result = 0; //0 for rotation, 1 for forward
geometry_msgs::Point current_location;
;
void callback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
	float motion_result = msg->angle_min;
	current_location = get_current_location();
	float y_current = current_location.y;
	std::cout << motion_result << std::endl;
	if (motion_result) {
		set_destination(0, y_current + 0.5, 1, 0);   //y is forward
		ROS_INFO("Forwarding!");
	}

	else{	
		set_yaw(10, 5, -1, 1);
		ROS_INFO("Rotating!");
	}
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
	takeoff(1);
    
    // set initial speed
    set_speed(0.5);  //0.5 meters per second


	ros::Subscriber lidar_sub = gnc_node.subscribe("/move_to_make", 1, callback);

	//specify control loop rate. We recommend a low frequency to not over load the FCU with messages. Too many messages will cause the drone to be sluggish
	ros::Rate rate(5.0);

        

	ros::spin();

	return 0;
}