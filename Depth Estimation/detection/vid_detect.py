import cv2
import numpy as np
import os
import time
import matplotlib as mpl
import matplotlib.cm as cm

from depth import depth
from model import model
#path
ROOT_PATH = os.path.dirname(__file__)

# camera setting for stereo
#   available
    # vcodec=mjpeg  min s=2560x960 fps=30 max s=2560x960 fps=60.0002
    # vcodec=mjpeg  min s=2560x720 fps=30 max s=2560x720 fps=60.0002
    # vcodec=mjpeg  min s=1280x480 fps=30 max s=1280x480 fps=60.0002
    # vcodec=mjpeg  min s=640x240 fps=30 max s=640x240 fps=60.0002

# start camera
vid = cv2.VideoCapture("../utils/cap_stereo.avi")

# warm up
_, frame = vid.read()
height, width2, n = frame.shape
width = int(width2/2)

# stereo matching
cal = depth.Depth(os.path.join(ROOT_PATH, 'depth/calibrate.npz'))
cal.createStereoMatch()

# detection
detect = model.FlowerDetection(os.path.join(ROOT_PATH, 'trained/YOLOv5n/weights/best.pt'))

# define output video
vid_left = cv2.VideoWriter(os.path.join(ROOT_PATH, 'vid_left.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 15, (width,height))
vid_disp = cv2.VideoWriter(os.path.join(ROOT_PATH, 'vid_disp.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 15, (width,height))
vid_detect = cv2.VideoWriter(os.path.join(ROOT_PATH, 'vid_detect.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 15, (width,height))

# helper for converting color map
normalizer = mpl.colors.Normalize(vmin=0, vmax=1)
mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')

# list of fps
fps = []

while(True):
    ret, frame = vid.read()

    if ret == False:
        break

    start_time = time.perf_counter()

    # split image to left and right
    imgl = frame[0:height, 0:width]
    imgr = frame[0:height, width:width2]

    # calibrate left and right
    imgl = cal.calibrate(imgl, 'L')
    imgr = cal.calibrate(imgr, 'R')

    disparity = cal.disparityMap(imgl, imgr) # first get the disparity from stereo matching
    # depthmap = cal.depthMap() # get actual depth map from disparity

    # convert to color scale image to display
    img_rgb = (mapper.to_rgba(disparity)[:, :, :3] * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
    # use left image for detection
    results = detect.score_frame(imgl)
    img = detect.plot_boxes_depth(results, imgl, disparity)

    # show image
    # cv2.imshow('left', imgl)
    # cv2.imshow('depth', disparity)
    # cv2.imshow('img', img)
    # cv2.imshow('depth', img_bgr)

    # start writing to video output
    vid_left.write(imgl)
    vid_disp.write(img_bgr)
    vid_detect.write(img)

    # fps
    end_time = time.perf_counter()
    fps_ = 1 / np.round(end_time - start_time, 3)
    fps.append(fps_)
    # print(fps_)

    k = cv2.waitKey(1)
    if k & 0xFF == 27: # 'esc' to quit
        break

# release and clean up
vid.release()
vid_left.release()
vid_disp.release()
vid_detect.release()

cv2.destroyAllWindows()

# print fps information
print('max: ' + str(np.max(fps)))
print('min: ' + str(np.min(fps)))
print('avg: ' + str(np.average(fps)))