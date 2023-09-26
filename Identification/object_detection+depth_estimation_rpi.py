import torch
import cv2
import numpy as np
import urllib.request

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

midas_transforms = torch.hub.load("intel-isl/MiDaS",  "transforms")
transform = midas_transforms.dpt_transform

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')

device = torch.device('cuda')
midas.to(device)
yolo.to(device)

stream = urllib.request.urlopen('http://192.168.4.1:8081/0/stream')
total_bytes = b''

print(torch.cuda.is_available())
# cam_port = 0
# cam = cv2.VideoCapture(cam_port)

while True:
    total_bytes += stream.read(1024)
    b = total_bytes.find(b'\xff\xd9') # JPEG end
    if not b == -1:
        a = total_bytes.find(b'\xff\xd8') # JPEG start
        jpg = total_bytes[a:b+2] # actual image
        total_bytes= total_bytes[b+2:] # other informations
        
        # decode to colored image ( another option is cv2.IMREAD_GRAYSCALE )
        frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR) 
        cv2.imshow('Window name',frame) # display image while receiving data
        if cv2.waitKey(1) == 27: # if user hit esc            
            break

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

# cam.release()
cv2.destroyAllWindows()