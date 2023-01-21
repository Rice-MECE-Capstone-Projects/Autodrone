#!/usr/bin/env python3

import os
import cv2
import glob
import torch
import numpy as np
import urllib.request
from PIL import Image, ImageOps
import torchvision.transforms as transforms

def avg(a, x1, y1, x2, y2):
  s = 0
  n = 0
  for i in range(x1,x2) :
    for j in range(y1,y2):
      if a[j][i] >= 0:
        s += a[j][i]
        n += 1
  if n == 0 :
    return -1
  else:
    return float(s/n)
    
def dist(a, x, y):
    distancei = (2 * 3.14 * 180) / (x + y * 360) * 1000 + 3
    return distancei
    

class FlowerDetection: 
  def __init__(self, weights='yolov5s.pt'):
    self.weights = weights
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model = self.load_model()
    self.classes = self.model.names

  def load_model(self):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weights)
    return model


  def score_frame(self, frame):
    self.model.to(self.device)
    frame = [frame]
    results = self.model(frame)
     
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord


  def class_to_label(self, x):
    return self.classes[int(x)]


  def plot_boxes(self, results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
      row = cord[i]
      if row[4] >= 0.5:
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        bgr = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame

  def plot_boxes_depth(self, results, frame, depthmap):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
      row = cord[i]
      if row[4] >= 0.2:
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        bgr = (255, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        #cv2.putText(frame, (str("{:.2f}".format(avg(depthmap,x1,y1,x2,y2))) + " cm"), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
        cv2.putText(frame, (str("{:.2f}".format(dist(depthmap,x_shape, y_shape))) + " inches"), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
    return frame


use_large_model = True
fps = []

if use_large_model:
	midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
else:
	midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

if use_large_model:
	transform = midas_transforms.dpt_transform
	print('Using large (slow) model.')
else:
	transform = midas_transforms.small_transform
	print('Using small (fast) model.')


for file in glob.glob('./rgb/*.jpg'):


	start_time = time.perf_counter()
	img = cv2.imread(file)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	input_batch = transform(img).to(device)
	
	with torch.no_grad():
		prediction = midas(input_batch)
	
		prediction = torch.nn.functional.interpolate(
			prediction.unsqueeze(1),
			size = img.shape[:2],
			mode = 'bicubic',
			align_corners = False,
		).squeeze()
	
	output = prediction.cpu().numpy()
	
	output_normalized = (output * 255 / np.max(output)).astype('uint8')
	#output_image = Image.fromarray(output_normalized)
	
	detect = FlowerDetection('/content/drive/MyDrive/depth/best.pt')
	results = detect.score_frame(img)
	#detect = FlowerDetection('/content/drive/MyDrive/depth/best.pt')

	opimg = detect.plot_boxes_depth(results, img, output_normalized)
	output_image = Image.fromarray(opimg)
	output_image_converted = output_image.convert('RGB').save(file.replace('rgb', 'depth'))
	end_time = time.perf_counter()
	fps_ = 1 / np.round(end_time - start_time, 3)
	fps.append(fps_)
	

	

	print('Converted: ' + file)

print('max: ' + str(np.max(fps)))
print('min: ' + str(np.min(fps)))
print('mean: ' + str(np.average(fps)))

print('Done.')