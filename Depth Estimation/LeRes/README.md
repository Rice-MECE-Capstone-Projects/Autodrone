Modified code from original repo: https://github.com/jankais3r/Video-Depthify.git

Follow the commands either locally or open a colab 

!mkdir rgb
!mkdir depth
!mkdir merged
!git clone https://github.com/compphoto/BoostingMonocularDepth.git
!gdown https://drive.google.com/u/0/uc?id=1cU2y-kMbt0Sf00Ns4CN2oO9qPJ8BensP&export=download
!mkdir -p /content/BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/
!mv latest_net_G.pth /content/BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/
!gdown https://drive.google.com/uc?id=1nqW_Hwj86kslfsXR7EnXpEWdO2csz1cC
!mv model.pt /content/BoostingMonocularDepth/midas/
!echo "Done."

if necessary model weights are not available, please follow the steos mentioned in this github folder - https://github.com/aim-uofa/AdelaiDepth.git --> go to LeRes folder --> you can download the necessary weights for running the inference 
This link can also be taken inspiration to tune for custom dataset

!ffmpeg -i input.avi 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p"  -- FPS of source video
!ffmpeg -i input.avi -qmin 1 -qscale:v 1 ./rgb/%06d.jpg  -- split the frames and load them to rbg folder

use the convert.py script if needed for .png or .jpg conversion (one to another)

!python <path>/BoostingMonocularDepth/run.py --Final --data_dir <path>/rgb --output_dir  <output>/leresdepth --depthNet 2
    --depthNet 2 == Leres
    --depthNet 0 == Midas
    --depthNet 1 == srlnet <download the necessary weights/files from github folders mentioned above>

!python3 average.py
!python3 merge.py
!ffmpeg -framerate 30 -i ./merged/%06d.jpg -vcodec libx264 -pix_fmt yuv420p outputgarden.avi  --> stitch the frames together to make the video


The LICENSE file from the original repo has been included.