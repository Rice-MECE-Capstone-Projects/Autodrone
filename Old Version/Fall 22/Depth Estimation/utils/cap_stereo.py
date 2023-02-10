import cv2

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

# height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
# width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# size = (frame_width, frame_height)

# define output video
vidout = cv2.VideoWriter('cap_stereo.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (width2,height))
# vidout = cv2.VideoWriter('cap_stereo.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (width,height))

flag = False # flag for start saving video frame

while(True):
    ret, frame = cam.read()

    if ret == False:
        break

    else:
        cv2.imshow('img', frame) # show stereo
        
        # imgl = frame[0:height, 0:width]
        # imgr = frame[0:height, width:width2]

        # start writing to video output
        if flag:
            vidout.write(frame)

        k = cv2.waitKey(1)
        if k & 0xFF == 27: # 'esc' to quit
            break
        elif k & 0xFF == ord('c'): # 'c' to start capturing
            flag = True

# release and clean up
cam.release()
vidout.release()

cv2.destroyAllWindows()
