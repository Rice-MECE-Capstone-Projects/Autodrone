# Import the necessary modules from the dronekit library
from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException

# Import the time, socket, and math modules
import time
import socket
import math

# Import the argparse module for parsing command line arguments
import argparse


def connectMyCopter():
    """
    Connect to the drone using the specified connection string, or start the default SITL
    simulator if no connection string is provided.
    """

    # Create an ArgumentParser object for defining and parsing command line arguments
    parser = argparse.ArgumentParser(description='Connect to a drone')

    # Add an argument for the connection string
    parser.add_argument('--connect')

    # Parse the command line arguments
    args = parser.parse_args()

    # Get the connection string from the parsed arguments
    connection_string = args.connect

    # If no connection string was provided, start the default SITL simulator
    if not connection_string:
        import dronekit_sitl
        sitl = dronekit_sitl.start_default()
        connection_string = sitl.connection_string()

    # Connect to the drone using the connection string
    vehicle = connect(connection_string, wait_ready=True)

    # Return the vehicle object representing the connected drone
    return vehicle


# Connect to the drone using the connectMyCopter() function
vehicle = connectMyCopter()
