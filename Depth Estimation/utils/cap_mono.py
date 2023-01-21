import cv2

# camera setting
    # vcodec=mjpeg  min s=1920x1080 fps=5 max s=1920x1080 fps=30
    # vcodec=mjpeg  min s=1280x1024 fps=5 max s=1280x1024 fps=30
    # vcodec=mjpeg  min s=1280x800 fps=5 max s=1280x800 fps=30
    # vcodec=mjpeg  min s=1280x720 fps=5 max s=1280x720 fps=30
    # vcodec=mjpeg  min s=800x600 fps=5 max s=800x600 fps=30
    # vcodec=mjpeg  min s=640x480 fps=5 max s=640x480 fps=30
    # vcodec=mjpeg  min s=320x240 fps=5 max s=320x240 fps=30
    # vcodec=mjpeg  min s=160x120 fps=5 max s=160x120 fps=30
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640

# start camera
cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
# cam.set(cv2.CAP_PROP_FPS, 30)

# warm up
size = (CAMERA_WIDTH, CAMERA_HEIGHT)

# height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
# width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# size = (frame_width, frame_height)

# define output video
vidout = cv2.VideoWriter('cap_mono.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size)

flag = False # flag for start saving video frame

while(True):
    ret, frame = cam.read()

    if ret == False:
        break

    else:
        cv2.imshow('img', frame) # show stereo

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
