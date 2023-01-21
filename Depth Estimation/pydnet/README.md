Modified code from original repo: https://github.com/mattpoggi/pydnet

- replace original ***pydnet.py*** with name ***utils_pydent.py***
    - original name conflicts with YOLOv5 *utils* file name
- copy ***test.py*** and ***video.py*** into the repo
    - modified from file 'webcam.py' from original repo
    - modified code is boxed by dash lines in comments
- copy ***detection.py*** from *autodrone/detection/* folder into the repo
- copy the .pt model from yolov5 into the repo

run with below to see camera stream (test) or video stream (video)
```
python test.py
python video.py
```