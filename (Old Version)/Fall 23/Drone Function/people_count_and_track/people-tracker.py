import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(".../yolov8n-seg.pt")

# Open the video file
video_path = ".../pedestrian3.mp4"
cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)  # For Webcam

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, save = True, persist=True, classes=0)
       # results = model.track(frame, save=True, persist=True, classes=0, tracker = 'bytetrack.yaml')

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        #print(results[0].boxes.shape[0])
        cv2.putText(annotated_frame, f'count:{results[0].boxes.shape[0]}', (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        #print(results.__len__())
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
