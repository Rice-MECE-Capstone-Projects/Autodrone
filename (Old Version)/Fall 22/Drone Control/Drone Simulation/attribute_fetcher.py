# Import necessary modules
from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
import time
import socket
import exceptions
import math
import argparse

#####FUNCTIONS#####

# Function to connect to the drone
def connectMyCopter():

    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    args = parser.parse_args()
    connection_string = args.connect

    # If no connection string was provided, start the default SITL simulator
    if not connection_string:
        import dronekit_sitl
        sitl = dronekit_sitl.start_default()
        connection_string = sitl.connection_string()

    # Connect to the drone and return the vehicle object
    vehicle = connect(connection_string, wait_ready=True)
    return vehicle

#####MAIN EXECUTABLE#####

# Connect to the drone and retrieve the vehicle object
vehicle = connectMyCopter()

# Wait until the autopilot is ready and then print its version
vehicle.wait_ready('autopilot_version')
print('AutoPilot version: %s' % vehicle.version)

# Print the drone's current position, attitude, velocity, and last heartbeat
print('Position: %s' % vehicle.location.global_relative_frame)
print('Attitude: %s' % vehicle.attitude)
print('Velocity: %s' % vehicle.velocity)
print('Last heartbeat: %s' % vehicle.last_heartbeat)

# Print whether the vehicle is armable and its current flight mode
print('Is the vehicle good to arm: %s' % vehicle.is_armable)
 
#Flight mode
print('Flight mode: %s' % vehicle.mode.name)
 
#Is the vehicle armed
print('Armed: %s' % vehicle.armed)
 
vehicle.close()
