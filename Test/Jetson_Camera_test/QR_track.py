# import the opencv and pyzbar library
import cv2
from pyzbar.pyzbar import decode

# ASSIGN CAMERA ADDRESS HERE
camera_id = "/dev/video0"

# define a video capture object
vid = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()

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
            print("Center")
        elif cx < frame.shape[1] // 2:
            print("Move to the left!")
        else:
            print("Move to the right!")

    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()