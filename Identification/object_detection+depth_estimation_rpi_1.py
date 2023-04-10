import torch
import cv2
import numpy as np
import urllib.request

def depth_detect(frame1, frame2):
    # Define the size of the object in real life (in meters)
    object_width = 0.5

    # Define the focal length of the camera (in pixels)
    focal_length = 1000


model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

midas_transforms = torch.hub.load("intel-isl/MiDaS",  "transforms")
transform = midas_transforms.dpt_transform

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')

device = torch.device('cuda')
midas.to(device)
yolo.to(device)

stream1 = urllib.request.urlopen('http://192.168.4.1:8081/stream')
stream2 = urllib.request.urlopen('http://192.168.4.1:8082/stream')
total_bytes1 = b''
total_bytes2 = b''

while True:
    total_bytes1 += stream1.read(1024)
    total_bytes2 += stream2.read(1024)
    b1 = total_bytes1.find(b'\xff\xd9') # JPEG end
    b2 = total_bytes2.find(b'\xff\xd9') # JPEG end
    if (not b1 == -1 and not b2 == -1):
        a1 = total_bytes1.find(b'\xff\xd8') # JPEG start
        a2 = total_bytes2.find(b'\xff\xd8') # JPEG start
        jpg1 = total_bytes1[a1:b1+2]        # actual image
        jpg2 = total_bytes2[a2:b2+2]        # actual image
        total_bytes1= total_bytes1[b1+2:]   # other informations
        total_bytes2= total_bytes2[b2+2:]   # other informations
        # decode to colored image ( another option is cv2.IMREAD_GRAYSCALE )
        frame1 = cv2.imdecode(np.frombuffer(jpg1, dtype=np.uint8), cv2.IMREAD_COLOR) 
        frame2 = cv2.imdecode(np.frombuffer(jpg2, dtype=np.uint8), cv2.IMREAD_COLOR)

        cv2.imshow('Window1 name',frame1) # display image while receiving data
        cv2.imshow('Window2 name',frame2) # display image while receiving data




        if cv2.waitKey(1) == 27: # if user hit esc            
            break

        # uese YOLOv5 to detect object in the frame
        detection1 = yolo(frame1)
        detection2 = yolo(frame2)

        # # midas
        # midas_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # midas_frame = transform(midas_frame).to(device)
        # with torch.no_grad():
        #     prediction = midas(midas_frame)

        #     prediction = torch.nn.functional.interpolate(
        #         prediction.unsqueeze(1),
        #         size=frame.shape[:2],
        #         mode="bicubic",
        #         align_corners=False,
        #     ).squeeze()
        # depth = prediction.cpu().numpy()

        # iterate through the detections and estimate depth using MiDaS
        # for det1, det2 in zip(detection1.xyxy[0], detection2.xyxy[0]):
        #     x11, y11, x12, y12, conf1, cls1 = det1
        #     x21, y21, x22, y22, conf2, cls2 = det2

        #     # median_depth = np.median(depth[int(y1):int(y2), int(x1):int(x2)])
            
        #     cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        #     # cv2.putText(frame, str(round(median_depth, 2)), (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 2)

        boxes1 = detection1.xyxy[0].cpu().detach().numpy()
        boxes2 = detection2.xyxy[0].cpu().detach().numpy()
        class_ids1 = detection1.pred[0].cpu().detach().numpy()[:, 5]
        class_ids2 = detection2.pred[0].cpu().detach().numpy()[:, 5]

        # Calculate distance between detected objects in both frames
        for i in range(len(boxes1)):
            for j in range(len(boxes2)):
                if class_ids1[i] == class_ids2[j]:
                    x1, y1, x2, y2, conf1, cls1 = boxes1[i]
                    x3, y3, x4, y4, conf2, cls2 = boxes2[j]
                    distance = ((x1 - x3)**2 + (y1 - y3)**2)**0.5
                    print(f"Distance between {yolo.names[int(class_ids1[i])]} in frame 1 and frame 2 is {distance}")

        # cv2.imshow('object detection & depth estimation', frame)

        # depth = cv2.normalize(depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        # cv2.imshow("depth map", depth)

        if cv2.waitKey(1) == 27:
            break

# cam.release()
cv2.destroyAllWindows()