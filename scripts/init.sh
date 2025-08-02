#!/bin/bash
set -e

# Run this script from the root of the project.

# 1. Require Python ≥3.8
if ! command -v python3 &> /dev/null; then
  echo "Python 3.8+ is required. Please install it."
  exit 1
fi

# Check Python version using Python itself for portability
PYTHON_VERSION=$(python3 -V 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
  echo "Python 3.8+ is required. Found $PYTHON_VERSION."
  exit 1
fi

# 2. Create & enter virtualenv
echo "Creating virtualenv..."
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip & install deps
echo "Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Prepare directories
mkdir -p weights images_uploaded

# 5. Download YOLOv3-tiny (if missing)
if [ ! -f weights/yolov3-tiny.weights ]; then
  echo "Downloading yolov3-tiny.weights from reliable mirror (original pjreddie.com is unavailable)..."
  curl -L -o weights/yolov3-tiny.weights https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov3-tiny.weights
  if [ ! -s weights/yolov3-tiny.weights ] || [ $(stat -f%z weights/yolov3-tiny.weights) -lt 30000000 ]; then
    echo "Error: Failed to download valid yolov3-tiny.weights (file too small or empty). Trying alternative mirror..."
    curl -L -o weights/yolov3-tiny.weights https://sourceforge.net/projects/yolov3.mirror/files/v8/yolov3-tiny.weights/download
    if [ ! -s weights/yolov3-tiny.weights ] || [ $(stat -f%z weights/yolov3-tiny.weights) -lt 30000000 ]; then
      echo "Error: All downloads failed. Please manually download yolov3-tiny.weights from https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov3-tiny.weights"
      exit 1
    fi
  fi
fi
if [ ! -f weights/yolov3-tiny.cfg ]; then
  echo "Downloading yolov3-tiny.cfg..."
  curl -L -o weights/yolov3-tiny.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
fi

# 6. Download full YOLOv3

if [ ! -f weights/yolov3.weights ]; then
  echo "Downloading yolov3.weights from reliable mirror (original pjreddie.com is unavailable)..."
  curl -L -o weights/yolov3.weights https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov3.weights
  if [ ! -s weights/yolov3.weights ] || [ $(stat -f%z weights/yolov3.weights) -lt 200000000 ]; then
    echo "Error: Failed to download valid yolov3.weights (file too small or empty). Trying alternative mirror..."
    curl -L -o weights/yolov3.weights https://sourceforge.net/projects/yolov3.mirror/files/v8/yolov3.weights/download
    if [ ! -s weights/yolov3.weights ] || [ $(stat -f%z weights/yolov3.weights) -lt 200000000 ]; then
      echo "Error: All downloads failed. Please manually download yolov3.weights from https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov3.weights"
      exit 1
    fi
  fi
fi
if [ ! -f weights/yolov3.cfg ]; then
  echo "Downloading yolov3.cfg..."
  curl -L -o weights/yolov3.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
fi


# 7. Download COCO names
if [ ! -f coco.names ]; then
  echo "Downloading coco.names..."
  curl -L -o coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
fi

echo "Setup complete!"
echo "  • Activate venv:   source venv/bin/activate"
echo "  • Start server:    uvicorn api.main:app --host 0.0.0.0 --port 8000"