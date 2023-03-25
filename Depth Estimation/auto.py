from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
import time
import socket
# import exceptions
import math
import argparse

def connectMyCopter():
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    args = parser.parse_args()

    connection_string = args.connect
    baud_rate = 57600

    vehicle = connect(connection_string, baud=baud_rate, wait_ready=True)
    return vehicle

def arm(vehicle):
    while vehicle.is_armable == False:
        print("Waiting vehicle to become stable...")
        time.sleep(1)
    print("Vehicle armable.")
    print("")

    vehicle.armed = False
    while vehicle.armed == False:
        print("Waiting vehicle to becaom armed...")
        time.sleep(1)
    print("Vehicle armed.")
    print("Props will start spinning.")

    return None

vehicle = connectMyCopter()
arm(vehicle)
print("End")


