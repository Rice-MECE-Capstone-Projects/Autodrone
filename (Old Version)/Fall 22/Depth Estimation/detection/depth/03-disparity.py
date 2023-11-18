import cv2 as cv
import numpy as np

# load calibration
CALIBRATION_FILE = 'calibrate.npz'
calibration = np.load(CALIBRATION_FILE, allow_pickle=False)
imgSize = tuple(calibration["imgSize"])
mapxL = calibration["mapxL"]
mapyL = calibration["mapyL"]
roiL = tuple(calibration["roiL"])
mapxR = calibration["mapxR"]
mapyR = calibration["mapyR"]
roiR = tuple(calibration["roiR"])

# camera input
CAMERA_HEIGHT = 720
CAMERA_WIDTH = 1920 # left right together

cap = cv.VideoCapture(1)

cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)

ret, frame = cap.read()
height, width, n = frame.shape
width2 = int(width/2)

# stereo matching
minDisparity = 4
numDisparities = 128
stereoMatcher = cv.StereoBM_create()
stereoMatcher.setROI1(roiL)
stereoMatcher.setROI2(roiR)
stereoMatcher.setMinDisparity(minDisparity)
stereoMatcher.setNumDisparities(numDisparities)
stereoMatcher.setBlockSize(19)
stereoMatcher.setUniquenessRatio(15)
stereoMatcher.setSpeckleRange(24)
stereoMatcher.setSpeckleWindowSize(45)

# Grab both frames first, then retrieve to minimize latency between cameras
while(True):
    ret, frame = cap.read()

    imgL = frame[0:height, 0:width2]
    imgR = frame[0:height, width2:width]

    imgL = cv.remap(imgL, mapxL, mapyL, cv.INTER_LINEAR)
    imgR = cv.remap(imgR, mapxR, mapyR, cv.INTER_LINEAR)

    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    disparity = stereoMatcher.compute(grayL, grayR) # disparity
    disparity = (disparity/16.0 - minDisparity)/numDisparities # scale down and normalize

    cv.imshow('left', imgL)
    cv.imshow('right', imgR)
    cv.imshow('disparity', disparity)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()