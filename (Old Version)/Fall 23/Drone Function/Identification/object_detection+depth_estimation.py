import torch
import cv2
import numpy as np

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

midas_transforms = torch.hub.load("intel-isl/MiDaS",  "transforms")
transform = midas_transforms.dpt_transform

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')

device = torch.device('cuda')
midas.to(device)
yolo.to(device)

cam_port = 0
cam = cv2.VideoCapture(cam_port)

while True:
    ret, frame = cam.read()

    # uese YOLOv5 to detect object in the frame
    detections = yolo(frame)

    # midas
    midas_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    midas_frame = transform(midas_frame).to(device)
    with torch.no_grad():
        prediction = midas(midas_frame)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # iterate through the detections and estimate depth using MiDaS
    for det in detections.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det

        median_depth = np.median(depth[int(y1):int(y2), int(x1):int(x2)])
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, str(round(median_depth, 2)), (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('object detection & depth estimation', frame)

    depth = cv2.normalize(depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    cv2.imshow("depth map", depth)

    c = cv2.waitKey(1)
    if c == 27:
        break

cam.release()
cv2.destroyAllWindows()