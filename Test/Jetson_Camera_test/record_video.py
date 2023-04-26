import cv2

def record_video():
    # ASSIGN CAMERA ADDRESS HERE
    camera_id = "/dev/video0"
    
    video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

    if not video_capture.isOpened():
        print("Unable to open camera")
        return
    
    # Set up the video writer
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("videos/output.avi", fourcc, fps, (frame_width, frame_height))
    
    recording = False
    
    while True:
        ret_val, frame = video_capture.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        if not ret_val:
            break
        
        # Display the frame
        cv2.imshow("Video", frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Start recording when "s" is pressed
        if key == ord("s") and not recording:
            recording = True
            out = cv2.VideoWriter("videos/output.avi", fourcc, fps, (frame_width, frame_height))
        
        # Stop recording when "p" is pressed
        elif key == ord("p") and recording:
            recording = False
            out.release()
        
        # Write the frame to the video file if recording
        if recording:
            out.write(frame)
        
        # Exit the loop if "q" is pressed
        elif key == ord("q"):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    record_video()
