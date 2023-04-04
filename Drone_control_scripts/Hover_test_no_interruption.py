import time
from dronekit import connect, VehicleMode,LocationGlobal,LocationGlobalRelative
#############################

############DRONEKIT#################
vehicle = connect('/dev/ttyACM0',wait_ready=True, baud=57600)

#Select /dev/ttyAMA0 for UART. /dev/ttyACM0 for USB

def arm_and_takeoff(aTargetAltitude):
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
        print (" Waiting for arming...")
        time.sleep(1)

    print ("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print (" Altitude: ", vehicle.location.global_relative_frame.alt)
        #Break and return from function just below target altitude.
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.5:
            print ("Reached target altitude")
            break
        time.sleep(1)
        
    # Hover for 1 second
    print("Hovering for 1 second...")
    time.sleep(1)

    print("Setting LAND mode...")
    vehicle.mode = VehicleMode("LAND")
    time.sleep(5)
    vehicle.armed   = False
    vehicle.mode    = VehicleMode("STABILIZE")

    # Close vehicle object
    vehicle.close()

arm_and_takeoff(1)
