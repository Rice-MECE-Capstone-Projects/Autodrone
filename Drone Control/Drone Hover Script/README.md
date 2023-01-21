## Hover Mission

This script contains a HoverMission class that is a subclass of the Mission class and is used to take off, hover, and land a drone. The HoverMission class initializes a DroneController instance to control the drone and a DirectLogger instance to log the drone's location data.

The start_mission method is an example of a callback endpoint that is called by a client to begin the hover mission. It takes off, hovers for a specified amount of time at a specified altitude, and lands.

The panic method can be called as a panic endpoint to immediately abort the mission and land the drone.

## Usage

To use the HoverMission class, create an instance of the class and specify the MAVProxy address of the flight controller and the name of the log file for location data. The start_server method is called automatically to start the web server that listens for callbacks to start the mission.

(`mission = HoverMission(fc_addr='127.0.0.1:14550', log_file='location.log')`)


To start the hover mission, send a POST request to the /start-mission endpoint with the following JSON data:

(`{
  "alt": 10,
  "hover_time": 5
}`)


The alt field specifies the target altitude in meters and the hover_time field specifies the hover time in seconds.

To abort the mission, send a POST request to the /panic endpoint. This will immediately land the drone.
