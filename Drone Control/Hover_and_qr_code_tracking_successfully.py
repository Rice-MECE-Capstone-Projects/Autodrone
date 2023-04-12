# import the opencv and pyzbar library
import cv2
from pyzbar.pyzbar import decode
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command
from pymavlink import mavutil
#############################

############DRONEKIT#################
# connect to the vehicle
vehicle = connect('/dev/ttyACM0', baud=57600, wait_ready=True)

# define the target altitude in meters
target_altitude = 1  # 0.9 meters = 3 feet

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

# rotate the drone to the right
def rotate_right():
    # print("Rotating right...")
    yaw_degrees = 30
    yaw_rate = 30  # degrees per second
    msg = vehicle.message_factory.command_long_encode(
        0, 0, mavutil.mavlink.MAV_CMD_CONDITION_YAW, 0,
        yaw_degrees, yaw_rate, 1, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)

# land the vehicle and disarm
def land_and_disarm():
    print("Landing...")
    vehicle.mode = VehicleMode("LAND")
    time.sleep(5)

    # wait until the vehicle lands
    while not vehicle.mode == VehicleMode("LAND"):
        time.sleep(1)

    while vehicle.armed:
        print(" Waiting for disarming...")
        time.sleep(1)

    vehicle.armed   = False
    vehicle.mode    = VehicleMode("STABILIZE")

    # close the vehicle object
    vehicle.close()

window_title = "Drone's view"

def track_qr_code():
    # ASSIGN CAMERA ADDRESS HERE
    camera_id = "/dev/video0"

    video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
   
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(
                window_title, cv2.WINDOW_AUTOSIZE )
            # Window
            while True:
                ret_val, frame = video_capture.read()
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                # Decode the QR code in the frame
                decoded_objs = decode(frame)

                # Draw bounding box around the detected QR code
                for obj in decoded_objs:
                    cv2.rectangle(frame, obj.rect, (0, 255, 0), 2)

                    # Check if the center of the QR code is in the center of the frame
                    cx = obj.rect.left + obj.rect.width // 2
                    cy = obj.rect.top + obj.rect.height // 2
                    height, width, _ = frame.shape
                    if abs(cx - width//2) < 10 and abs(cy - height//2) < 10:
                        print("QR code in the Center!")
                    elif cx < frame.shape[1] // 2:
                        print("Rotating to the left side...")
                        # don't have a function for rotating left
                    else:
                        print("Rotating to the right side...")
                        rotate_right()
                
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break

        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    try:
        # arm and take off to the target altitude
        arm_and_takeoff(target_altitude)

        # Show the camera and track the QR code
        track_qr_code()

        # land and disarm the vehicle
        land_and_disarm()

    except KeyboardInterrupt:
        # if the user interrupts the program, land and disarm the vehicle
        print("User interrupt detected...")
        land_and_disarm()