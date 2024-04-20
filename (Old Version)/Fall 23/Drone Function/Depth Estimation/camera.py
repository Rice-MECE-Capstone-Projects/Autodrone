import cv2
import depth_estimation as de
import time

cam_port = 0
cap = cv2.VideoCapture('C:/Users/MECE2/Desktop/Autodrone F23/Autodrone/Depth Estimation/video2.avi')

while cap.isOpened():
    ret, frame = cap.read()

    out_frame, depth_map = de.depth_estimation(frame)
    
    cv2.imshow('object detection & depth estimation', out_frame)
   
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.imshow("depth map", depth_map)

    c = cv2.waitKey(1)
    if c == 27:
        break

    if c == 32:
        while True:
            time.sleep(0.2)
            c = cv2.waitKey(1)
            if c == 32:
                break

cap.release()
cv2.destroyAllWindows()