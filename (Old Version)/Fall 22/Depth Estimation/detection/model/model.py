# https://github.com/niconielsen32/ComputerVision/blob/master/DeployYOLOmodel.py
import torch
import numpy as np
import cv2

# util function to get average disparity
def avg(a, x1, y1, x2, y2):
    s = 0
    n = 0
    for i in range(y1,y2) :
        for j in range(x1,x2):
            if a[i][j] > 0: # only use available pixels values
                s += a[i][j]
                n += 1
    if n == 0 :
        return -1 # return for unvailable result
    else:
        return float(s/n) # return average

class FlowerDetection:   
    # initialize with parameters 
    def __init__(self, weights='yolov5s.pt'):
        self.weights = weights
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()
        self.classes = self.model.names

    # load model
    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weights)
        return model

    # get labels and box cordinate
    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        # get the coordinates from xyxyn
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    # from class index to label
    def class_to_label(self, x):
        return self.classes[int(x)]

    # plot box with labels and cordinates
    def plot_boxes(self, results, frame): 
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    # plot box with depth and cordinates
    def plot_boxes_depth(self, results, frame, depthmap): # input depthMap as disparity ranging [0,1]
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0] # get frame size
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.25:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (255, 255, 0)
                val = avg(depthmap, x1, y1, x2, y2)
                val = (1/val)*25.268073 + 3.779730 # convertion from disparity to depth (learn from depth)
                if val < 0: # ignore boxes out of range
                  continue
                # draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, (str("{:.2f}".format(val)) + " cm"), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
        return frame

    # def plot_boxes_depth(self, results, frame, depthmap):
    #     labels, cord = results
    #     n = len(labels)
    #     x_shape, y_shape = frame.shape[1], frame.shape[0]
    #     for i in range(n):
    #         row = cord[i]
    #         if row[4] >= 0.2:
    #             x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
    #             bgr = (255, 255, 0)
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
    #             cv2.putText(frame, (str("{:.2f}".format(avg(depthmap,x1,y1,x2,y2))) + " cm"), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
    #     return frame
