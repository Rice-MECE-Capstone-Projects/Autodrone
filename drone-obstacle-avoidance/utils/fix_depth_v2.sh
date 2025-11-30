cat > ~/fix_depth_v2.sh << 'EOF'
#!/bin/bash

echo "🔧 Fixing depth-anything-v2 compatibility issues..."

RUN_FILE=~/.local/lib/python3.10/site-packages/reComputer/scripts/depth-anything-v2/run.sh

# Remove failed container
docker rm depth-anything-v2 2>/dev/null

# Backup
cp $RUN_FILE ${RUN_FILE}.bak
echo "✅ Backed up to ${RUN_FILE}.bak"

# Create fixed file
cat > $RUN_FILE << 'SCRIPT'
CONTAINER_NAME="depth-anything-v2"
IMAGE_NAME="yaohui1998/depthanything-v2-on-jetson-orin:latest"

# Pull the latest image
docker pull $IMAGE_NAME

# Check if the container with the specified name already exists
if [ $(docker ps -a -q -f name=^/${CONTAINER_NAME}$) ]; then
    echo "Container $CONTAINER_NAME already exists. Starting and attaching..."
    docker start $CONTAINER_NAME
else
    echo "Container $CONTAINER_NAME does not exist. Creating and starting..."
    docker run -it \
        --name $CONTAINER_NAME \
        --privileged \
        --network host \
        -v /usr/lib/aarch64-linux-gnu:/host/lib:ro \
        -e LD_LIBRARY_PATH=/host/lib:$LD_LIBRARY_PATH \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /dev/*:/dev/* \
        -v /etc/localtime:/etc/localtime:ro \
        --runtime nvidia \
        $IMAGE_NAME
fi
SCRIPT

echo "✅ Fix completed!"
echo ""
echo "Changes:"
diff ${RUN_FILE}.bak $RUN_FILE || true
echo ""
echo "🚀 Now run:"
echo "   reComputer run depth-anything-v2"
EOF

chmod +x ~/fix_depth_v2.sh
~/fix_depth_v2.sh