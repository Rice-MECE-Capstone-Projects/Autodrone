#!/bin/bash

echo "🔍 Pre-run checklist"
echo ""

# 1. Disk mount
echo "1️⃣ Check disk mount:"
if mountpoint -q /mnt/hdd; then
    echo "   ✅ /mnt/hdd is mounted"
    df -h | grep hdd
else
    echo "   ❌ /mnt/hdd is NOT mounted"
    echo "   Run: sudo mount /dev/nvme0n1p1 /mnt/hdd"
fi

echo ""

# 2. Docker configuration
echo "2️⃣ Check Docker configuration:"
DOCKER_ROOT=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $NF}')
if [[ "$DOCKER_ROOT" == "/mnt/hdd/docker" ]]; then
    echo "   ✅ Docker storage is configured on 1TB disk"
else
    echo "   ⚠️  Docker storage: $DOCKER_ROOT"
    echo "   Run: sudo systemctl restart docker"
fi

echo ""

# 3. Disk space
echo "3️⃣ Check disk space:"
df -h | grep -E "Filesystem|hdd|mmcblk0p1"

echo ""

# 4. Docker service status
echo "4️⃣ Check Docker service:"
if systemctl is-active --quiet docker; then
    echo "   ✅ Docker service is running"
else
    echo "   ❌ Docker service is NOT running"
    echo "   Run: sudo systemctl start docker"
fi

echo ""
echo "✨ Check complete! If all are ✅, you can run YOLO"