#include <gnc_functions.hpp>
#include <geometry_msgs/Point.h>
//include API 

float human_position_angle = -0.5;
float human_position_height = -0.5;
bool person_found = false;

void person_pos_callback(const geometry_msgs::Point::ConstPtr& msg)
{
    human_position_angle = msg->x;
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

	ros::Subscriber person_pos_sub = gnc_node.subscribe("/person_position", 10, person_pos_callback);

	//specify control loop rate. We recommend a low frequency to not over load the FCU with messages. Too many messages will cause the drone to be sluggish
	ros::Rate rate(5.0);
	while(ros::ok())
	{
        if (!person_found) {
            if (std::abs(human_position_angle) < 0.01){
                person_found = true;
            }
            else {
                if (human_position_angle > 0) {
                    ROS_INFO("Current human position %f, rotate clockwise.", human_position_angle);
                    set_yaw(1, 10, 1, 1);
                }
                else if (human_position_angle < 0){
                    ROS_INFO("Current human position %f, rotate counterclockwise.", human_position_angle);
                    set_yaw(1, 10, -1, 1);
                }
            }
        }
        else{
            set_yaw(0, 0, 0, 1);
        }
        

		ros::spinOnce();
		rate.sleep();
		
	}
	return 0;
}