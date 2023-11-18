#
# MIT License
#
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modified file "webcam.py" from original repo: https://github.com/mattpoggi/pydnet

import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from utils_pydnet import *
from pydnet import *

# ------for detection
import detection

# forces tensorflow to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--model', dest='model', type=str, choices=['pydnet', 'pydnet2'], default='pydnet', help='choose model')
parser.add_argument('--resolution', dest='resolution', type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/IROS18/pydnet', help='checkpoint directory')

args = parser.parse_args()

def main(_):

    with tf.Graph().as_default():
        height = 256 if args.model == 'pydnet' else 384
        width = 512 if args.model == 'pydnet2' else 640
        # ---------------------------------------
        # get required ratio for model input to crop the camera input
        r = height/width
        crop = int(240 - height/2)
        # ---------------------------------------

        placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 3], name='im0')}

        with tf.variable_scope("model") as scope:
            if args.model == 'pydnet':
                model = pydnet(placeholders)
            elif args.model == 'pydnet2':
                model = pydnet2(placeholders)

        init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

        loader = tf.train.Saver()
        saver = tf.train.Saver()
        cam = cv2.VideoCapture(1)
        # ---------------------------------------
        #detection
        detect = detection.FlowerDetection('trained/YOLOv5n/weights/best.pt')
        # ---------------------------------------

        with tf.Session() as sess:
            sess.run(init)
            if args.model == 'pydnet2':
                args.checkpoint_dir = 'checkpoint/ITS/pydnet2'

            # ---------------------------------------
            # get frame size of camera
            _, img = cam.read()
            ch, cw, _ = img.shape
            # print(img.shape)
            # ---------------------------------------
            loader.restore(sess, args.checkpoint_dir)
            while True:
                for i in range(4):
                    cam.grab()
                _, img = cam.read() 
                # ---------------------------------------
                # crop image to correct input size
                img = img[int((ch-r*cw)/2):int((ch+r*cw)/2), :]
                # new image for detection only
                new = cv2.resize(img, (width, height)).astype(np.float32) 
                # ---------------------------------------
                img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
                img = np.expand_dims(img, 0)
                start = time.time()
                disp = sess.run(model.results[args.resolution-1], feed_dict={placeholders['im0']: img})
                # ---------------------------------------
                # detection
                results = detect.score_frame(new)
                new = detect.plot_boxes(results, new)
                # fps caculation
                end = time.time()
                fps = 1 / np.round(end - start, 3)
                # ---------------------------------------

                color_scaling = 20
                if args.model == 'pydnet2':
                    color_scaling = 1/64.

                disp_color = applyColorMap(disp[0,:,:,0]*color_scaling, 'viridis')
                toShow = (np.concatenate((new, disp_color*255.), 0)).astype(np.uint8)
                # toShow = (np.concatenate((img[0], disp_color), 0)*255.).astype(np.uint8)
                # toShow = cv2.resize(toShow, (width//2, height))

                cv2.imshow(args.model, toShow)
                k = cv2.waitKey(1)         
                if k == 1048603 or k == 27: 
                    break  # esc to quit
                if k == 1048688:
                    cv2.waitKey(0) # 'p' to pause
                # ---------------------------------------
                # print fps
                print("fps: " + str("{:.2f}".format(fps)) + "\tTime: " + str(end - start))
                # ---------------------------------------
                del img
                del disp
                del toShow
            
            cam.release()        

if __name__ == '__main__':
    tf.app.run()
