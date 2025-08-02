#!/bin/bash
set -e
# Run this script from the root of the project.
# 1. Require Python ≥3.8
if ! command -v python3 &> /dev/null; then
  echo "Python 3.8+ is required. Please install it."
  exit 1
fi
PYTHON_VERSION=$(python3 -V 2>&1 | awk '{print $2}')
if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
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
  echo "Downloading yolov3-tiny.weights..."
  curl -L -o weights/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights
fi
if [ ! -f weights/yolov3-tiny.cfg ]; then
  echo "Downloading yolov3-tiny.cfg..."
  curl -L -o weights/yolov3-tiny.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
fi

# 6. (Optional) full YOLOv3
# Uncomment these lines if you want the full model
# if [ ! -f weights/yolov3.weights ]; then
#   echo "Downloading yolov3.weights..."
#   curl -L -o weights/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
# fi
# if [ ! -f weights/yolov3.cfg ]; then
#   echo "Downloading yolov3.cfg..."
#   curl -L -o weights/yolov3.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
# fi

# 7. Download COCO names
if [ ! -f coco.names ]; then
  echo "Downloading coco.names..."
  curl -L -o coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
fi

echo "Setup complete!"
echo "  • Activate venv:   source venv/bin/activate"
echo "  • Start server:    uvicorn api.main:app --host 0.0.0.0 --port 8000"
