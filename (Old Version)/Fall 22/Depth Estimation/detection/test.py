import cv2
import numpy as np
import os
import time

import depth.depth as depth
import model.model as model

#path
ROOT_PATH = os.path.dirname(__file__)
# camera input values
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 1280 # left right together
# calibration
cal = depth.Depth(os.path.join(ROOT_PATH, 'depth/calibrate.npz'))
# open camera
cap = cv2.VideoCapture(1)
# modify camera parameter
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
# warm up
_, frame = cap.read()
h, width2,_ = frame.shape
w = int(width2/2)

# set mouse click display distance
def onClick(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        print('depth = ' + str(depthmap[y][x]) + ' cm')

# stereo matching
cal.createStereoMatch()

# detection
detect = model.FlowerDetection(os.path.join(ROOT_PATH, 'trained/YOLOv5n/weights/best.pt'))

while(True):
    start_time = time.perf_counter()

    _, frame = cap.read()
    imgl = frame[0:h, 0:w]
    imgr = frame[0:h, w:width2]

    # calibrate left and right
    imgl = cal.calibrate(imgl, 'L')
    imgr = cal.calibrate(imgr, 'R')

    disparity = cal.disparityMap(imgl, imgr) # first get the disparity from stereo matching
    depthmap = cal.depthMap() # get actual depth map from disparity

    # cv2.imshow('left', imgl)
    # cv2.imshow('right', imgr)
    cv2.imshow('depth', disparity)

    # use left image for detecion
    results = detect.score_frame(imgl)
    img = detect.plot_boxes_depth(results, imgl, depthmap)
    # img = detect.plot_boxes(results, imgl, depthmap)

    end_time = time.perf_counter()
    fps = 1 / np.round(end_time - start_time, 3)
    cv2.putText(img, f'FPS: {int(fps)}', (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow('img', img)
    cv2.setMouseCallback('img', onClick) # click on image to get the depth
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()