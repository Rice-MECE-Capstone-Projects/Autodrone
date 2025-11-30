#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
MODEL_PATH="$CHECKPOINT_DIR/depth_pro.pt"
EXPECTED_SIZE=1904446787  # Official model size (bytes)

echo "📦 Installing Depth Pro dependencies..."

# 1. Install basic dependencies
pip3 install -q pillow timm matplotlib

# 2. Enter Depth Pro directory and install
cd "$PROJECT_ROOT/depth_pro"

# 3. Install Depth Pro (editable mode)
if ! pip3 show depth_pro &> /dev/null; then
    echo "📥 Installing Depth Pro..."
    pip3 install -e .
else
    echo "✅ Depth Pro already installed"
fi

# 4. Create unified checkpoints directory
mkdir -p $CHECKPOINT_DIR

# 5. Check model file
MODEL_VALID=false

if [ -f "$MODEL_PATH" ]; then
    ACTUAL_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null)
    
    if [ "$ACTUAL_SIZE" -eq "$EXPECTED_SIZE" ]; then
        # Verify file integrity
        if python3 -c "import torch; torch.load('$MODEL_PATH', map_location='cpu')" 2>/dev/null; then
            echo "✅ Depth Pro model already exists and is valid"
            MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
            echo "   Model size: $MODEL_SIZE"
            MODEL_VALID=true
        else
            echo "⚠️  Model file is corrupted, will re-download"
            rm -f "$MODEL_PATH"
        fi
    else
        echo "⚠️  Incorrect model size ($ACTUAL_SIZE bytes, expected $EXPECTED_SIZE bytes)"
        echo "   Deleting and re-downloading..."
        rm -f "$MODEL_PATH"
    fi
fi

# 6. Download model
if [ "$MODEL_VALID" = false ]; then
    echo "📥 Downloading Depth Pro pretrained model (~1.77GB)..."
    echo "   This may take a few minutes, please wait..."
    
    # Use -c to support resume
    wget -c --show-progress -O "$MODEL_PATH" \
        https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
    
    # Verify again
    ACTUAL_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null)
    if [ "$ACTUAL_SIZE" -ne "$EXPECTED_SIZE" ]; then
        echo "❌ Download failed: incorrect file size"
        exit 1
    fi
    
    if ! python3 -c "import torch; torch.load('$MODEL_PATH', map_location='cpu')" 2>/dev/null; then
        echo "❌ Downloaded file is corrupted"
        exit 1
    fi
    
    echo "✅ Model downloaded and verified successfully"
fi

echo ""
echo "✅ Depth Pro installation complete!"
echo "📊 Model location:"
ls -lh "$MODEL_PATH"
