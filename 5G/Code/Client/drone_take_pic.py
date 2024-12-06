import cv2

def takePic(file_name):
    # Initialize the camera
    print("Opening camera")
    cap = cv2.VideoCapture(0)  # 0 usually refers to the default camera

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Capture a frame
    print("Capturing frame")
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Error reading frame")

    # Resize the frame to a lower resolution
    print("Scaling frame")
    new_width = 640  # Set your desired width
    new_height = 480  # Set your desired height
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Save the resized frame as an image
    print("Writing frame")
    cv2.imwrite(file_name, resized_frame)

    # Release the camera
    print("Closing camera")
    cap.release()

if __name__ == '__main__':
    takePic(test.jpg)