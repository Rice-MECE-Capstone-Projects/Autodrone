#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚁 Starting drone obstacle avoidance system..."

# Run Python script
python3 $SCRIPT_DIR/avoidance.py "$@"