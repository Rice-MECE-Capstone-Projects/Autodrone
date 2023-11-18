## Flower Detection + Depth Esitmation with OpenCV StereoBM
Use this command to run:
```
python test.py
```

Limitation of stereo camera disparity: objects have to be at least around 30cm away from the camera to successful calculate distance


- ***cap_stereo.py***
  - start up the camera
  - press 'c' to start recording
  - press 'esc' to stop and escape
  - 2 videos will be saved
    - entire stereo video
    - disparity  
- ***vid_stereo.py***
  - use captured stereo video
  - populate 
    - calibrated left camera video
    - disparity

To do calibration for camera, use files listed in depth folder.
