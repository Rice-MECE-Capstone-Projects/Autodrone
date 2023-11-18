# Import necessary modules
from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
import time
import socket
import exceptions
import math
import argparse
 
def connectMyCopter():
    # Parse command line arguments to get connection string
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    args = parser.parse_args()
 
    connection_string = args.connect
     
    # If no connection string was provided, start the Sitl simulation
    if not connection_string:
        import dronekit_sitl
        sitl = dronekit_sitl.start_default()
        connection_string = sitl.connection_string()
 
    # Connect to the drone
    vehicle = connect(connection_string, wait_ready=True)
 
    return vehicle
 
def arm_and_takeoff(targetHeight):
    # Wait for the drone to become armable
    while vehicle.is_armable!=True:
        print("Waiting for vehicle to become aramable")
        time.sleep(1)
    print("Vehicle is now armable")
 
    # Set the drone's flight mode to "GUIDED"
    vehicle.mode = VehicleMode("GUIDED")
 
    # Wait for the drone to enter "GUIDED" mode
    while vehicle.mode!="GUIDED":
        print("Waiting for drone to become armed")
        time.sleep(1)
    print("Look out! Virtual props are spinning!")
 
    vehicle.simple_takeoff(targetHeight) ## in meters
 
    while True:
        print("Current Altitude:", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt>=.95*targetHeight:
            break
        time.sleep(1)
    print("Target altitude reached!!")
    return None
 
#  calculate the distance in meters and return
def get_distance_meters(targetLocation, currentLocation):
    dLat = targetLocation.lat - currentLocation.lat
    dLon = targetLocation.lon - currentLocation.lon
 
    return math.sqrt((dLon*dLon)+(dLat*dLat))*1.113195e5
 
#  fly the drone to the location specified
def goto(targetLocation):
    distanceToTargetLocation = get_distance_meters(targetLocation, vehicle.location.global_relative_frame)
 
    vehicle.simple_goto(targetLocation)
 
    while vehicle.mode.name == "GUIDED":
        currentDistance = get_distance_meters(targetLocation,vehicle.location.global_relative_frame)
        # when the drone is close to the location, change its state
        if currentDistance < distanceToTargetLocation*.01:
            print("Reached target waypoint")
            time.sleep(2)
            break
        time.sleep(1)
    return None
 
#  define the location
wp1 = LocationGlobalRelative(29.720273, -95.399829, 10)
 
vehicle = connectMyCopter()
arm_and_takeoff(10)
 
#  fly the drone
goto(wp1) 
 
vehicle.mode = VehicleMode("LAND")
while vehicle.mode != 'LAND'
    print("Waiting for drone to enter LAND mode")
    time.sleep(1)
print("Vehicle in LAND mode")
 
while True:
    time.sleep(1)
