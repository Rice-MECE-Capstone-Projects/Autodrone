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

# hover for the specified duration at the current location
def hover(duration):
    print("Hovering in place for ", duration, " seconds")
    time.sleep(duration)

# move forward for the specified duration at a fixed speed
def move_forward(duration):
    print("Moving forward for ", duration, " seconds")
    vehicle.airspeed = 1 # set airspeed to 1 m/s
    vehicle.simple_goto(LocationGlobalRelative(vehicle.location.global_relative_frame.lat + 0.0001, vehicle.location.global_relative_frame.lon, vehicle.location.global_relative_frame.alt))
    time.sleep(duration)

# safely land the vehicle and disarm
def land():
    print("Landing vehicle")
    vehicle.mode = VehicleMode("LAND")
    time.sleep(5)
    vehicle.armed = False
    vehicle.mode    = VehicleMode("STABILIZE")

    # Close vehicle object
    vehicle.close()

arm_and_takeoff(target_altitude)
hover(hover_duration)
move_forward(hover_duration)
land()