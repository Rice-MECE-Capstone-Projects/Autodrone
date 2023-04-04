import time
from dronekit import connect, VehicleMode,LocationGlobal,LocationGlobalRelative
#############################

############DRONEKIT#################
# connect to the vehicle
vehicle = connect('/dev/ttyACM0',wait_ready=True, baud=57600)

#Select /dev/ttyAMA0 for UART. /dev/ttyACM0 for USB

# set the target altitude and duration for hovering
target_altitude = 1 # 1 foot
hover_duration = 1 # 1 second

# arm and takeoff the vehicle to the target altitude
def arm_and_takeoff(aTargetAltitude):
    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)

    print("Taking off to altitude: ", aTargetAltitude)
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        print("Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

# move forward for the specified duration at a fixed speed
def move_forward(duration):
    print("Moving forward for ", duration, " seconds")
    vehicle.airspeed = 1 # set airspeed to 1 m/s
    vehicle.simple_goto(LocationGlobalRelative \
    (vehicle.location.global_relative_frame.lat - 0.00001, vehicle.location.global_relative_frame.lon, vehicle.location.global_relative_frame.alt))
    time.sleep(duration)


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
    arm_and_takeoff(target_altitude)

    # hover for 1 second at the target altitude
    print("Hovering...")
    time.sleep(1)

    # move forward for 1 second at the specified velocity
    print("Moving forward...")
    move_forward(duration)

    # hover for 1 second at the target altitude
    print("Hovering...")
    time.sleep(1)

    # land and disarm the vehicle
    land_and_disarm()

except KeyboardInterrupt:
    # if the user interrupts the program, land and disarm the vehicle
    print("User interrupt detected...")
    land_and_disarm()