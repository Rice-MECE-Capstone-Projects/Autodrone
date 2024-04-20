import cv2
from ultralytics import YOLO
import multiprocessing

def raw_stream(video_path):
    # Loop through the video frames
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if not success:
            # Break the loop if the end of the video is reached
            break
        # Display the annotated frame
        cv2.imshow("YOLOv8 real-time", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

def process_stream(video_path):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_path)
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if not success:
            # Break the loop if the end of the video is reached
            break
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Open the video file
    video_path = "http://192.168.4.1:8081/0/stream"

    p1 = multiprocessing.Process(target=raw_stream, args=(video_path,))
    p2 = multiprocessing.Process(target=process_stream, args=(video_path,))
    # starting process 1
    p1.start()
    # starting process 2
    p2.start()

    # wait until process 1 is finished
    p1.join()
    # wait until process 2 is finished
    p2.join()
