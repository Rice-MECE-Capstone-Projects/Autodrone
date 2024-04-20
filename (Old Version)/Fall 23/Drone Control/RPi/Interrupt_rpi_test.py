import time
from dronekit import connect, VehicleMode,LocationGlobal,LocationGlobalRelative
#############################

############DRONEKIT#################
vehicle = connect('/dev/ttyAMA0',wait_ready=True, baud=57600)

#Select /dev/ttyAMA0 for UART. /dev/ttyACM0 for USB

aTargetAltitude = 1

def arm_and_disarm():
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print ("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print (" Waiting for vehicle to initialise...")
        time.sleep(1)

    print ("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode    = VehicleMode("GUIDED")
    vehicle.armed   = True

    # Confirm vehicle armed before attempting to take off
    while not vehicle.armed:
        time.sleep(1)
        print (" Waiting for arming...")
        time.sleep(1)

    print (" Waiting 180 sec")
    time.sleep(180)
    
        
    while vehicle.armed:
        print(" Waiting for disarming...")
        time.sleep(1)

    # close the vehicle object
    vehicle.close()

try:
    # arm and take off to the target altitude
    arm_and_disarm(aTargetAltitude)

except KeyboardInterrupt:
    # if the user interrupts the program, land and disarm the vehicle
    print("User interrupt detected...")
