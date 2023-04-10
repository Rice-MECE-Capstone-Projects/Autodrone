import time
from dronekit import connect, VehicleMode,LocationGlobal,LocationGlobalRelative
#############################

############DRONEKIT#################
vehicle = connect('/dev/ttyACM0',wait_ready=True, baud=57600)

#Select /dev/ttyAMA0 for UART. /dev/ttyACM0 for USB

aTargetAltitude = 1

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
        
# land the vehicle and disarm
def land_and_disarm():
    print("Landing...")
    vehicle.mode = VehicleMode("LAND")

    # wait until the vehicle lands
    while not vehicle.mode == VehicleMode("LAND"):
        time.sleep(1)

    while vehicle.armed:
        print(" Waiting for disarming...")
        time.sleep(1)

    # close the vehicle object
    vehicle.close()

try:
    # arm and take off to the target altitude
    arm_and_takeoff(aTargetAltitude)

    # hover for 1 second at the target altitude
    print("Hovering...")
    time.sleep(1)

    # land and disarm the vehicle
    land_and_disarm()

except KeyboardInterrupt:
    # if the user interrupts the program, land and disarm the vehicle
    print("User interrupt detected...")
    land_and_disarm()
