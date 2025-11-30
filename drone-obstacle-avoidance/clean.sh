#!/bin/bash

echo "🧹 Cleaning drone obstacle avoidance system data..."

# Clean model cache
rm -rf ~/.cache/depth_anything_v2/Depth-Anything-V2
rm -f ~/.cache/depth_anything_v2/depth_anything_v2_vits.pth

# Clean YOLO cache
rm -rf ~/.cache/torch/hub/ultralytics

# Clean output files
rm -f drone_avoidance_output.mp4
rm -f frame_*.jpg

echo "✅ Cleaning completed!"