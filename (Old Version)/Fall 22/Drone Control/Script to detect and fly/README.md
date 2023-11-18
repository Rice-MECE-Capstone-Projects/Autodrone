# Description
Both of these scripts are implemented using a Mission class from the skymission.mission module, which is a subclass of grpc.server, and uses a DroneController class from the skyengine.drone module to control the drone. The start_mission method is to be the main entry point for the mission, and it takes a list of waypoints as input and makes the drone fly to them in sequence. The start_location_log method logs the drone's location to a log file using a DirectLogger from the skylog.logger module.

The idea of this design is to allow the drone to first fly a default route to detect flowers (in this case, a straight line for 3 meters in total.) and then it also allows the user to manually input the location of the target(s) so that it will also fly to assigned locations.

The autodrone script just defines the default route and it is meant to be tested to see if the drone can fly in default routes and also fly towards assigned locations.

The search_flower script added in the connection to the grpc to allow the camera to detect the box which represent the flower in the frame and move to the center of the flower as well as do the operation (in this case, lower the height to 0.5 meter) before moving on to finish the rest of the route.


## Waypoint Goto Mission
This script provides a mission for a drone to follow a series of waypoints. The mission is defined by the WaypointGotoMission class, which is derived from the Mission class.

## Prerequisites
Before using this script, you will need to have the following installed:

Python 3


gRPC


The skyengine and skylog Python packages


## Usage
To use the script, you will need to specify the MAVProxy address of the flight controller and the name of the log file for location data. For example:

(` mission = WaypointGotoMission(fc_addr='127.0.0.1:14551', log_file='location.log')`)


Once you have created an instance of the WaypointGotoMission class, you can start the mission by calling the start_mission() method and passing it a list of waypoints and the desired hover duration at each waypoint. The waypoints should be specified as a list of dictionaries, each containing the latitude, longitude, and altitude of the waypoint. For example:

(`waypoints = [
    {'lat': 37.123456, 'lon': -122.123456, 'alt': 10.0},
    {'lat': 37.234567, 'lon': -122.234567, 'alt': 20.0},
    {'lat': 37.345678, 'lon': -122.345678, 'alt': 30.0},
]
mission.start_mission(waypoints, hover_duration=5.0)`)


This will start the drone on its mission to follow the waypoints in the order they were specified, hovering at each waypoint for the specified duration before moving on to the next waypoint. The drone's location will also be logged to the specified log file at regular intervals.

## Further Information
For more information on the classes and methods used in this script, please see the documentation for the skyengine and skylog packages.
