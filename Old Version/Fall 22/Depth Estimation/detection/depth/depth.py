import numpy as np
import cv2


class Depth:
    def __init__(self, file='depth/calibrate.npz'):
        self.file = file
        # Load calibration parameters
        self.calibration = np.load(file, allow_pickle=False)
        self.imgSize = tuple(self.calibration["imgSize"])
        self.mapxL = self.calibration["mapxL"]
        self.mapyL = self.calibration["mapyL"]
        self.roiL = tuple(self.calibration["roiL"])
        self.mapxR = self.calibration["mapxR"]
        self.mapyR = self.calibration["mapyR"]
        self.roiR = tuple(self.calibration["roiR"])

    def calibrate(self, frame, side='L'):
        # Given frame and left or right camera, Return calibrated frame
        if side == 'L':
            return cv2.remap(frame, self.mapxL, self.mapyL, cv2.INTER_LINEAR)
        elif side == 'R':
            return cv2.remap(frame, self.mapxR, self.mapyR, cv2.INTER_LINEAR)
        else:
            return frame
    
    def createStereoMatch(self, numDisparities = 128, minDisparity = 4, blockSize = 17, speckleRange = 16, speckleWindowSize = 45):
        self.numDisparities = numDisparities
        self.minDisparity = minDisparity
        self.stereoMatcher = cv2.StereoBM_create()
        self.stereoMatcher.setROI1(self.roiL)
        self.stereoMatcher.setROI2(self.roiR)
        self.stereoMatcher.setMinDisparity(minDisparity)
        self.stereoMatcher.setNumDisparities(numDisparities)
        self.stereoMatcher.setBlockSize(blockSize)
        self.stereoMatcher.setSpeckleRange(speckleRange)
        self.stereoMatcher.setSpeckleWindowSize(speckleWindowSize)

    def disparityMap(self, imgl, imgr):
        grayl = cv2.cvtColor(imgl, cv2.COLOR_BGR2GRAY)
        grayr = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)

        self.disparity = self.stereoMatcher.compute(grayl, grayr) # disparity
        self.disparity = (self.disparity/16.0 - self.minDisparity + 1)/self.numDisparities # scale down and normalize to 0-1

        # return disparity
        return self.disparity

    def depthMap(self):
        return (1/(self.disparity+0.000000001))*25.268073 + 3.779730
