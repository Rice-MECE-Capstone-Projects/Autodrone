#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"

echo "🚁 Initializing drone obstacle avoidance system..."

# Install dependencies
echo "📦 Installing Python dependencies..."
pip3 install -q "numpy<2.0" ultralytics opencv-python pymavlink torch torchvision

# Verify NumPy version
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
echo "✅ NumPy version: $NUMPY_VERSION"

# Create unified checkpoints directory
mkdir -p $CHECKPOINT_DIR

# Download YOLO11s model
YOLO_MODEL="$CHECKPOINT_DIR/yolo11s.pt"
if [ -f "$YOLO_MODEL" ]; then
    echo "✅ YOLO11s model already exists"
else
    echo "📥 Downloading YOLO11s model (~22MB)..."
    python3 << PYEOF
from ultralytics import YOLO
model = YOLO('yolo11s.pt')
# Move to checkpoints
import shutil
shutil.move('yolo11s.pt', '$YOLO_MODEL')
PYEOF
fi

# Download Depth Anything V2 Small model (backup)
DEPTH_MODEL_DIR="$HOME/.cache/depth_anything_v2"
mkdir -p $DEPTH_MODEL_DIR

if [ ! -f "$DEPTH_MODEL_DIR/depth_anything_v2_vits.pth" ]; then
    echo "📥 Downloading Depth Anything V2 Small model (~100MB)..."
    wget -q --show-progress -O $DEPTH_MODEL_DIR/depth_anything_v2_vits.pth \
        https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
else
    echo "✅ Depth Anything V2 model already exists"
fi

# Clone Depth Anything V2 code
if [ ! -d "$DEPTH_MODEL_DIR/Depth-Anything-V2" ]; then
    echo "📥 Cloning Depth Anything V2 code..."
    cd $DEPTH_MODEL_DIR
    git clone -q https://github.com/DepthAnything/Depth-Anything-V2.git
    cd -
fi

echo "✅ Initialization completed!"
echo "📊 Model locations:"
echo "  - YOLO11s: $CHECKPOINT_DIR/yolo11s.pt"
echo "  - Depth V2: $DEPTH_MODEL_DIR/"
ls -lh $CHECKPOINT_DIR/