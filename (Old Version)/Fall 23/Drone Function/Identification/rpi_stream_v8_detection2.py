from ultralytics import YOLO
import multiprocessing
import cv2

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


def process_stream(model, video_path):
    results = model.predict(source=video_path, show=True)



if __name__ == '__main__':
    model = YOLO('../../YOLO_Models/yolov8/yolov8n.pt')
    # model = YOLO('../mask_detection/weights/best_weight.pt')
    video_path = "http://192.168.4.1:8081/0/stream"
    p1 = multiprocessing.Process(target=raw_stream, args=(video_path,))
    p2 = multiprocessing.Process(target=process_stream, args=(model, video_path)) # starting process 1
    p1.start()
    p2.start()

    p1.join()
    p2.join()