Modified code from original repo: https://github.com/jankais3r/Video-Depthify.git and https://github.com/isl-org/MiDaS.git 
Inlcuded detection section of the YoloV5 model trained for flower detection
Use the best.pt model in the path mentioned in depth_withDetection python file or adjust the path accordingly

!mkdir rgb
!mkdir depth
!mkdir merged

!pip3 install torch torchvision opencv-python timm Pillow numpy -q
!wget https://raw.githubusercontent.com/jankais3r/Video-Depthify/main/depth.py
!wget https://raw.githubusercontent.com/jankais3r/Video-Depthify/main/average.py
!wget https://raw.githubusercontent.com/jankais3r/Video-Depthify/main/merge.py
!echo "Done."

!pip3 install torch torchvision opencv-python timm Pillow numpy -q
!ffmpeg -i input.avi 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p"
!python3 depth_withDetection.py
!python3 average.py
!python3 merge.py
!ffmpeg -framerate 30 -i ./merged/%06d.jpg -vcodec libx264 -pix_fmt yuv420p outputMidasLargeDP.avi