#!/usr/bin/env python3

import os
import glob
import numpy as np
from PIL import Image

items = len(glob.glob('./leresdepth/*.jpg')) - 2
first = './leresdepth/000001.jpg'
last = './leresdepth/' + str(items + 2).zfill(6) + '.jpg'
#print(len(last))
#print(last)
w, h = Image.open(first).size
Image.open(first).save(first.replace('leresdepth', 'averaged'))

for idx in range(items):
	current = idx + 2
	arr = np.zeros((h, w, 3), np.float) #h, w, 3
	
	prev = np.array(Image.open('./leresdepth/' + str(current - 1).zfill(6) + '.jpg'), dtype = np.float)
	curr = np.array(Image.open('./leresdepth/' + str(current).zfill(6) + '.jpg'), dtype = np.float)
	next = np.array(Image.open('./leresdepth/' + str(current + 1).zfill(6) + '.jpg'), dtype = np.float)
  
	arr = arr+prev/3
	arr = arr+curr/3
	arr = arr+next/3
	
	arr = np.array(np.round(arr), dtype = np.uint8)
	
	out = Image.fromarray(arr) #,mode = 'RGB'
	out.save('./averaged/' + str(current).zfill(6) + '.jpg')
	#print('Averaged: ' + str(current).zfill(6) + '.jpg')

Image.open(last).save(last.replace('leresdepth', 'averaged'))
print('Done.')