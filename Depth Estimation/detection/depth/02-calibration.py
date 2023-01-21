import numpy as np
import cv2 as cv
import os
import glob



CHESSBOARD_SIZE = (5, 8)
CHESSBOARD_OPTIONS = (cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_FAST_CHECK)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)


OPTIMIZE_ALPHA = 0.1

ROOT_PATH = os.path.dirname(__file__)
IMAGE_PATH = os.path.join(ROOT_PATH, 'data/')
imageL = glob.glob(IMAGE_PATH + 'left*.jpg')
imageR = glob.glob(IMAGE_PATH + 'right*.jpg')

def findChessboard(imageglob):
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in imageglob:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgSize = gray.shape[::-1]
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD_SIZE, cv.CALIB_CB_FAST_CHECK)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

    cv.destroyAllWindows()

    return objpoints, imgpoints, imgSize

objpointsL, imgpointsL, imgSize = findChessboard(imageL)
objpointsR, imgpointsR, imgSize = findChessboard(imageR)
# Calibration seperately
_, mtxL, distL, _, _ = cv.calibrateCamera(objpointsL, imgpointsL, imgSize, None, None)
_, mtxR, distR, _, _ = cv.calibrateCamera(objpointsR, imgpointsR, imgSize, None, None)

# Calibration together
_, _, _, _, _, rvecs, tvecs, _, _ = cv.stereoCalibrate(
    objpointsL, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR,
    imgSize, None, None, None, None, cv.CALIB_FIX_INTRINSIC, criteria)

# Rectify
recL, recR, projL, projR, dispartityToDepthMap, roiL, roiR = cv.stereoRectify(
    mtxL, distL, mtxR, distR, imgSize, rvecs, tvecs,
    None, None, None, None, None, cv.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)

# Save Calibration
mapxL, mapyL = cv.initUndistortRectifyMap(mtxL, distL, recL, projL, imgSize, cv.CV_32FC1)
mapxR, mapyR = cv.initUndistortRectifyMap(mtxR, distR, recR, projR, imgSize, cv.CV_32FC1)

outputFile = os.path.join(ROOT_PATH, 'calibrate')

np.savez_compressed(outputFile, imgSize=imgSize,
        mapxL=mapxL, mapyL=mapyL, roiL=roiL,
        mapxR=mapxR, mapyR=mapyR, roiR=roiR)


# draw for example
imgL = cv.imread(IMAGE_PATH + 'left0.jpg')
imgR = cv.imread(IMAGE_PATH + 'right0.jpg')

dstL = cv.remap(imgL, mapxL, mapyL, cv.INTER_LINEAR)
dstR = cv.remap(imgR, mapxR, mapyR, cv.INTER_LINEAR)

cv.imwrite(os.path.join(ROOT_PATH, 'calibrate_result_left.jpg'), dstL)
cv.imwrite(os.path.join(ROOT_PATH, 'calibrate_result_right.jpg'), dstR)

cv.destroyAllWindows()