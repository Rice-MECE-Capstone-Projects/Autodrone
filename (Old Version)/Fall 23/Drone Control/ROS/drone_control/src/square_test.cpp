#include <gnc_functions.hpp>
//include API 

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
	takeoff(8);

	//specify some waypoints 
	std::vector<gnc_api_waypoint> waypointList;
	gnc_api_waypoint nextWayPoint;
	nextWayPoint.x = 0;
	nextWayPoint.y = 0;
	nextWayPoint.z = 8;
	nextWayPoint.psi = 0;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = -5;
	nextWayPoint.y = 2.5;
	nextWayPoint.z = 8;
	nextWayPoint.psi = 0;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = -5;
	nextWayPoint.y = 10;
	nextWayPoint.z = 8;
	nextWayPoint.psi = 45;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = -10;
	nextWayPoint.y = 20;
	nextWayPoint.z = 10;
	nextWayPoint.psi = 135;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = -5;
	nextWayPoint.y = 20;
	nextWayPoint.z = 13;
	nextWayPoint.psi = 225;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = -5;
	nextWayPoint.y = 25;
	nextWayPoint.z = 9;
	nextWayPoint.psi = 315;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = -10;
	nextWayPoint.y = 30;
	nextWayPoint.z = 7;
	nextWayPoint.psi = 315;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = -5;
	nextWayPoint.y = 30;
	nextWayPoint.z = 10;
	nextWayPoint.psi = 315;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = 5;
	nextWayPoint.y = 25;
	nextWayPoint.z = 12;
	nextWayPoint.psi = 315;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = 10;
	nextWayPoint.y = 20;
	nextWayPoint.z = 10;
	nextWayPoint.psi = 225;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = 15;
	nextWayPoint.y = 20;
	nextWayPoint.z = 8;
	nextWayPoint.psi = 135;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = 5;
	nextWayPoint.y = 10;
	nextWayPoint.z = 7;
	nextWayPoint.psi = 45;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = 5;
	nextWayPoint.y = 2.5;
	nextWayPoint.z = 6;
	nextWayPoint.psi = 0;
	waypointList.push_back(nextWayPoint);
	nextWayPoint.x = 0;
	nextWayPoint.y = 0;
	nextWayPoint.z = 5;
	nextWayPoint.psi = 0;
	waypointList.push_back(nextWayPoint);


	//specify control loop rate. We recommend a low frequency to not over load the FCU with messages. Too many messages will cause the drone to be sluggish
	ros::Rate rate(2.0);
	int counter = 0;
	while(ros::ok())
	{
		ros::spinOnce();
		rate.sleep();
		if(check_waypoint_reached(.3) == 1)
		{
			if (counter < waypointList.size())
			{
				set_destination(waypointList[counter].x,waypointList[counter].y,waypointList[counter].z, waypointList[counter].psi);
				counter++;
				sleep(0.5);
			}else{
				//land after all waypoints are reached
				land();
			}	
		}	
		
	}
	return 0;
}