# YOLOv3 Detection API

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
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Initialize environment & download weights**

   ```bash
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

## Credits

* **YOLOv3 & YOLOv3-Tiny models** developed by **Joseph Redmon** and the Darknet community.
* **COCO dataset classes** by **COCO Consortium**.
* **FastAPI** by **Sebastian Ramirez**.

## License
