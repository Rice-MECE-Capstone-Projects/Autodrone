import sys

import cv2

window_title = "USB Camera"

def show_camera():
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

    show_camera()
