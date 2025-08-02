# YOLOv3 Detection API
<img width="822" alt="Demo" src="https://github.com/user-attachments/assets/599989e7-2179-4ab7-a2bd-39ead18b6803" />

A FastAPI-based object detection service using YOLOv3/YOLOv3-Tiny (Darknet) models.

## Features

* **Multiple Routes**

  * `/detect/bboxes` — Returns JSON bounding boxes & confidence.
  * `/detect/metadata` — Returns detailed detections (coords, class names, confidence).
  * `/detect/image` — Returns annotated image as PNG.
  * `/detect/detect_and_draw_box` — Returns Base64-encoded PNG + metadata.
  * `/predict` — Unified endpoint streaming annotated JPEG.
* **Model Selection**

  * Choose between `tiny` (YOLOv3-Tiny) or `full` (YOLOv3) via `?model=tiny|full`.
* **Zero dependencies on CVLib** — Pure OpenCV DNN + FastAPI.
* **Docker-friendly** — Easy containerization.

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/cosminmemetea/mmx-detector.git
   cd mmx-detector
   ```

2. **Initialize environment & download weights**

   ```bash
   chmod +x scripts/init.sh
   scripts/init.sh
   source venv/bin/activate
   ```

3. **Start the server**

   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Explore API docs**

   * Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   * ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## Usage Examples

```bash
# 1. Get bounding boxes
curl -F "file=@image.jpg" "http://localhost:8000/detect/bboxes?model=tiny"

# 2. Get full metadata
curl -F "file=@image.jpg" "http://localhost:8000/detect/metadata?model=full"

# 3. Get annotated PNG
curl -F "file=@image.jpg" "http://localhost:8000/detect/image?model=tiny" --output out.png

# 4. Detect & draw with metadata
curl -F "file=@image.jpg" "http://localhost:8000/detect/detect_and_draw_box?model=tiny" | jq .detections

# 5. Unified predict (JPEG stream)
curl -F "file=@image.jpg" "http://localhost:8000/predict?model=full" --output out.jpg
```

## Project Structure

```
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── events.py
│   └── utils.py
├── weights/
│   ├── yolov3-tiny.cfg    # YOLOv3-Tiny config
│   ├── yolov3-tiny.weights
│   ├── yolov3.cfg         # YOLOv3 full config
│   └── yolov3.weights
├── coco.names             # COCO class names
├── images_uploaded/       # Annotated output images
├── scripts/
│   └── init.sh            # Environment initialization script
├── requirements.txt
└── README.md
```
Docker Builds:

docker run --rm -p 8000:8000 yolov3-ros2-api

docker run --rm -p 8000:8000 mmx-detector

## Usage
docker build -t xmmx-detector:latest .
docker run --rm -p 8000:8000 xmmx-detector:latest


## Links

- Original Darknet (archived): https://github.com/pjreddie/darknet
- AlexeyAB's fork (widely used for YOLOv3/v4): https://github.com/AlexeyAB/darknet
- Hank.ai's fork (current active maintenance): https://github.com/hank-ai/darknet
- Official Darknet website (original, by Joseph Redmon): https://pjreddie.com/darknet/
- Hank.ai announcements: https://hank.ai/darknet-welcomes-hank-ai-as-official-sponsor-and-commercial-entity/ (sponsorship details)
- For weights downloads (use mirrors for reliability, as pjreddie.com is unstable):

- YOLOv3-Tiny: https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov3-tiny.weights
- YOLOv3 Full: https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov3.weights


- COCO classes: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names (still reliable)

## Credits

- **YOLOv3 & YOLOv3-Tiny models** developed by **Joseph Redmon** and the Darknet community (original repository: https://github.com/pjreddie/darknet).
- Fork and enhancements by **Alexey Bochkovskiy (AlexeyAB)** (repository: https://github.com/AlexeyAB/darknet).
- Current maintenance sponsored by **Hank.ai** and developed by **Stéphane Charette** (repository: https://github.com/hank-ai/darknet).
- **COCO dataset classes** by **COCO Consortium**.
- **FastAPI** by **Sebastián Ramírez**.


## License

MIT – Permissive open-source license allowing free use, modification, and distribution for any purpose, with minimal restrictions.

## Contributing
Contributions make this repo better! Whether fixing bugs, adding features, or improving docs, you're welcome.
Let's have fun!
