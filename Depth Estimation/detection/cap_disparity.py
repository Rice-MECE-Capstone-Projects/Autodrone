import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from depth import depth

#path
ROOT_PATH = os.path.dirname(__file__)

# camera setting for stereo
#   available
    # vcodec=mjpeg  min s=2560x960 fps=30 max s=2560x960 fps=60.0002
    # vcodec=mjpeg  min s=2560x720 fps=30 max s=2560x720 fps=60.0002
    # vcodec=mjpeg  min s=1280x480 fps=30 max s=1280x480 fps=60.0002
    # vcodec=mjpeg  min s=640x240 fps=30 max s=640x240 fps=60.0002
# CAMERA_HEIGHT = 240
# CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 1280

# start camera
cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)

# warm up
_, frame = cam.read()
height, width2, n = frame.shape
width = int(width2/2)

# stereo matching
cal = depth.Depth(os.path.join(ROOT_PATH, 'depth/calibrate.npz'))
cal.createStereoMatch()

# define output video
# vidout = cv2.VideoWriter('cap_stereo.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (width,height))
vid_stereo = cv2.VideoWriter(os.path.join(ROOT_PATH, 'cap_stereo.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (width2,height))
vid_disp = cv2.VideoWriter(os.path.join(ROOT_PATH, 'cap_disp.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (width,height))

flag = False # flag for start saving video frame

while(True):
    ret, frame = cam.read()

    if ret == False:
        break

    else:
        cv2.imshow('img', frame) # show stereo
        
        imgl = frame[0:height, 0:width]
        imgr = frame[0:height, width:width2]

        # calibrate left and right
        imgl = cal.calibrate(imgl, 'L')
        imgr = cal.calibrate(imgr, 'R')

        disparity = cal.disparityMap(imgl, imgr) # first get the disparity from stereo matching
        depthmap = cal.depthMap() # get actual depth map from disparity

        # convert to color scale image to display
        normalizer = mpl.colors.Normalize(vmin=0, vmax=1)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        img_rgb = (mapper.to_rgba(disparity)[:, :, :3] * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow('mapper', img_bgr)
        # cv2.imshow('depth', disparity)

        # start writing to video output
        if flag:
            vid_stereo.write(frame)
            vid_disp.write(img_bgr)

        k = cv2.waitKey(1)
        if k & 0xFF == 27: # 'esc' to quit
            break
        elif k & 0xFF == ord('c'): # 'c' to start capturing
            flag = True
            print('start capturing...')

# release and clean up
cam.release()
vid_stereo.release()
vid_disp.release()

cv2.destroyAllWindows()

print('capture ends')
