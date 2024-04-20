import time
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command
from pymavlink import mavutil
#############################

############DRONEKIT#################
# connect to the vehicle
vehicle = connect('/dev/ttyACM0', baud=57600, wait_ready=True)

# define the target altitude in meters
target_altitude = 0.3  # 1 foot = 0.3048 meters

# arm and take off to the target altitude
def arm_and_takeoff(aTargetAltitude):
    # set the vehicle mode to GUIDED
    vehicle.mode = VehicleMode("GUIDED")

    # arm the vehicle
    vehicle.armed = True
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    # take off to the target altitude
    print("Taking off to a target altitude of %.1f meters..." % aTargetAltitude)
    vehicle.simple_takeoff(aTargetAltitude)

    # wait until the vehicle reaches the target altitude
    while True:
        altitude = vehicle.location.global_relative_frame.alt
        if altitude >= aTargetAltitude * 0.95:
            print("Reached target altitude of %.1f meters" % aTargetAltitude)
            break
        time.sleep(1)

# rotate the drone to the right by 90 degrees
def rotate_right():
    print("Rotating right by 90 degrees")
    yaw_degrees = 90
    yaw_rate = 90  # degrees per second
    msg = vehicle.message_factory.command_long_encode(
        0, 0, mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0,
        yaw_degrees, yaw_rate, 1, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)

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

    # rotate the drone to the left by 90 degrees
    print("Rotating...")
    rotate_right()

    # hover for 1 second at the target altitude
    print("Hovering...")
    time.sleep(1)

    # land and disarm the vehicle
    land_and_disarm()

except KeyboardInterrupt:
    # if the user interrupts the program, land and disarm the vehicle
    print("User interrupt detected...")
    land_and_disarm()
