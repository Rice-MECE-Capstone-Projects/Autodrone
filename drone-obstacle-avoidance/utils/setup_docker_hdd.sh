#!/bin/bash

echo "🔧 Configure Docker to use 1TB disk..."

# 1. Find the 1TB disk mount point (e.g., /media/penguin/... or /mnt/...)
echo "Current disk status:"
df -h | grep -E "Filesystem|/media|/mnt|/$"

echo ""
read -p "Please enter the 1TB disk mount point (e.g. /mnt/hdd): " DISK_PATH

if [ ! -d "$DISK_PATH" ]; then
    echo "❌ Path does not exist: $DISK_PATH"
    exit 1
fi

# 2. Create Docker directory
sudo mkdir -p $DISK_PATH/docker

# 3. Stop Docker
echo "Stopping Docker..."
sudo systemctl stop docker

# 4. Copy existing data (optional)
read -p "Copy existing Docker data? (y/n): " COPY_DATA
if [[ $COPY_DATA == "y" ]]; then
    echo "Copying data..."
    sudo rsync -aP /var/lib/docker/ $DISK_PATH/docker/
fi

# 5. Modify Docker config
echo "Modifying Docker configuration..."
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "data-root": "$DISK_PATH/docker",
  "storage-driver": "overlay2"
}
EOF

# 6. Start Docker
echo "Starting Docker..."
sudo systemctl start docker

# 7. Verify
echo ""
echo "✅ Configuration complete!"
echo "New Docker data directory:"
docker info | grep "Docker Root Dir"

echo ""
echo "💾 Disk usage:"
df -h | grep -E "Filesystem|$DISK_PATH|/$"

# 8. Optional: clean old data
echo ""
read -p "Delete old Docker data to free SSD space? (y/n): " CLEAN_OLD
if [[ $CLEAN_OLD == "y" ]]; then
    echo "⚠️  Deleting old data..."
    sudo rm -rf /var/lib/docker
    echo "✅ Old data deleted"
fi

echo ""
echo "🎉 Done! Docker images will now be stored in $DISK_PATH/docker"