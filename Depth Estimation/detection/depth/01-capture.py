import cv2 as cv
import os

# path for saving the images for calibration
ROOT_PATH = os.path.dirname(__file__)
IMAGE_PATH = os.path.join(ROOT_PATH, 'data/')
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

i = 0 # image index

# camera setting
CAMERA_HEIGHT = 720
CAMERA_WIDTH = 1920 # left right together

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)

ret, frame = cap.read()
height, width, n = frame.shape
width2 = int(width/2)

while (True):
    ret, frame = cap.read()

    im0 = frame[0:height, 0:width2]
    im1 = frame[0:height, width2:width]

    cv.imshow('left', im0)
    cv.imshow('right', im1)

    k = cv.waitKey(1)
    if k & 0xFF == ord('c'): # press c to capture image
        cv.imwrite(IMAGE_PATH + '/left' + str(i) + '.jpg', im0)
        cv.imwrite(IMAGE_PATH + '/right' + str(i) + '.jpg', im1)
        i += 1

    elif k & 0xFF == ord('q'): # press q to exit
        break


cap.release()
cv.destroyAllWindows()