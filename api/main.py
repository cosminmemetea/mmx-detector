import io
import logging
import base64
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

from .events import EventBus

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model files
MODELS = {
    "tiny": {
        "cfg": BASE_DIR / "weights" / "yolov3-tiny.cfg",
        "weights": BASE_DIR / "weights" / "yolov3-tiny.weights",
    },
    "full": {
        "cfg": BASE_DIR / "weights" / "yolov3.cfg",
        "weights": BASE_DIR / "weights" / "yolov3.weights",
    },
}
COCO_NAMES_PATH = BASE_DIR / "coco.names"

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI(
    title="YOLOv3 Detection API",
    version="1.0.0",
    description="Routes: bboxes, metadata, image, detect_and_draw_box, predict",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

event_bus = EventBus()

# Load class names
with open(COCO_NAMES_PATH) as f:
    CLASS_NAMES = [c.strip() for c in f]

# Pydantic models
class BBox(BaseModel):
    x1: float; y1: float; x2: float; y2: float; confidence: float

class Detection(BaseModel):
    x1: float; y1: float; x2: float; y2: float; confidence: float; class_name: str

class DetectionResponse(BaseModel):
    detections: List[Detection]

class DetectAndDrawResponse(BaseModel):
    image_base64: str
    detections: List[Detection]

# Thresholds
CONF_THRESH = 0.5
NMS_THRESH = 0.4

# Load & cache networks
networks: Dict[str, cv2.dnn_Net] = {}
output_layers_map: Dict[str, List[str]] = {}
for name, paths in MODELS.items():
    net = cv2.dnn.readNetFromDarknet(str(paths["cfg"]), str(paths["weights"]))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    ln = net.getLayerNames()
    out_idxs = net.getUnconnectedOutLayers().flatten()
    output_layers_map[name] = [ln[i - 1] for i in out_idxs]
    networks[name] = net

def run_yolo(image: np.ndarray, model: str) -> Tuple[List[List[int]], List[float], List[int], List[int]]:
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net = networks[model]
    net.setInput(blob)
    outputs = net.forward(output_layers_map[model])
    boxes, confs, cls_ids = [], [], []
    for out in outputs:
        for det in out:
            scores = det[5:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if conf > CONF_THRESH:
                cx, cy, bw, bh = det[0]*w, det[1]*h, det[2]*w, det[3]*h
                x, y = int(cx - bw/2), int(cy - bh/2)
                boxes.append([x, y, int(bw), int(bh)])
                confs.append(conf)
                cls_ids.append(cid)
    idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
    # normalize to flat list
    if hasattr(idxs, "flatten"):
        idx_list = idxs.flatten().tolist()
    elif isinstance(idxs, (list, tuple)) and idxs:
        # cv2 returns [(n1,),(n2,)...]
        idx_list = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in idxs]
    else:
        idx_list = []
    return boxes, confs, cls_ids, idx_list

def _read_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image.")
    return img

def draw_boxes(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw professionally styled boxes + labels with percent confidences."""
    h, w = image.shape[:2]
    thick = max(2, int(0.003 * max(h, w)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, 0.001 * max(h, w))
    for det in detections:
        x1, y1 = int(det.x1), int(det.y1)
        x2, y2 = int(det.x2), int(det.y2)
        label = f"{det.class_name}: {det.confidence*100:.1f}%"
        # box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thick)
        # text background
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thick)
        cv2.rectangle(
            image,
            (x1, y1 - th - baseline - 5),
            (x1 + tw + 5, y1),
            (0, 255, 0),
            cv2.FILLED,
        )
        # put text
        cv2.putText(
            image,
            label,
            (x1 + 2, y1 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness=max(1, thick // 2),
            lineType=cv2.LINE_AA,
        )
    return image

# ————————— Routes ————————— #

from fastapi.responses import HTMLResponse

@app.get("/", include_in_schema=False)
async def root() -> HTMLResponse:
    html_content = """
    <html>
      <head>
        <title>🚀 YOLOv3 Detection API 🚀</title>
        <style>
          body {
            background: #1e1e1e;
            color: #c5c6c7;
            font-family: 'Segoe UI', Tahoma, sans-serif;
            text-align: center;
            padding: 40px;
          }
          h1 {
            color: #66fcf1;
            font-size: 3em;
            margin-bottom: 0.2em;
          }
          .card {
            background: #0b0c10;
            border-radius: 8px;
            display: inline-block;
            margin: 10px;
            padding: 20px;
            min-width: 200px;
          }
          .card h2 {
            margin-top: 0;
            color: #45a29e;
          }
          a {
            color: #45a29e;
            text-decoration: none;
            font-weight: bold;
          }
          a:hover {
            text-decoration: underline;
          }
          footer {
            margin-top: 50px;
            font-size: 0.9em;
            color: #808080;
          }
        </style>
      </head>
      <body>
        <h1>🤖 YOLOv3 Detection API 🤖</h1>
        <div class="card">
          <h2>📖 Documentation</h2>
          <p>Interactive docs at <a href="/docs">/docs</a> or <a href="/redoc">/redoc</a></p>
        </div>
        <div class="card">
          <h2>✨ Endpoints</h2>
          <ul style="list-style:none; padding:0;">
            <li>POST <code>/detect/bboxes</code></li>
            <li>POST <code>/detect/metadata</code></li>
            <li>POST <code>/detect/image</code></li>
            <li>POST <code>/detect/detect_and_draw_box</code></li>
            <li>POST <code>/predict</code></li>
          </ul>
        </div>
        <footer>Made with ❤️ using FastAPI & YOLOv3 • Happy detecting! 🚀</footer>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)



@app.post("/detect/bboxes", response_model=List[BBox])
async def detect_bboxes(
    file: UploadFile = File(...),
    model: str = Query("tiny", enum=list(MODELS.keys())),
):
    data = await file.read()
    img = _read_image(data)
    boxes, confs, _, idxs = run_yolo(img, model)
    return [
        BBox(x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3], confidence=confs[i])
        for i, b in zip(idxs, [boxes[j] for j in idxs])
    ]

@app.post("/detect/metadata", response_model=DetectionResponse)
async def detect_metadata(
    file: UploadFile = File(...),
    model: str = Query("tiny", enum=list(MODELS.keys())),
):
    data = await file.read()
    img = _read_image(data)
    boxes, confs, cls_ids, idxs = run_yolo(img, model)
    dets = [
        Detection(
            x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3],
            confidence=confs[i], class_name=CLASS_NAMES[cls_ids[i]]
        )
        for i, b in zip(idxs, [boxes[j] for j in idxs])
    ]
    return DetectionResponse(detections=dets)

@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    model: str = Query("tiny", enum=list(MODELS.keys())),
):
    data = await file.read()
    img = _read_image(data)
    boxes, confs, cls_ids, idxs = run_yolo(img, model)
    # build Detection objects
    dets = [
        Detection(
            x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3],
            confidence=confs[i], class_name=CLASS_NAMES[cls_ids[i]]
        )
        for i, b in zip(idxs, [boxes[j] for j in idxs])
    ]
    annotated = draw_boxes(img.copy(), dets)
    _, buf = cv2.imencode(".png", annotated)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

@app.post("/detect/detect_and_draw_box", response_model=DetectAndDrawResponse)
async def detect_and_draw_box(
    file: UploadFile = File(...),
    model: str = Query("tiny", enum=list(MODELS.keys())),
):
    data = await file.read()
    img = _read_image(data)
    boxes, confs, cls_ids, idxs = run_yolo(img, model)
    dets = [
        Detection(
            x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3],
            confidence=confs[i], class_name=CLASS_NAMES[cls_ids[i]]
        )
        for i, b in zip(idxs, [boxes[j] for j in idxs])
    ]
    annotated = draw_boxes(img.copy(), dets)
    _, buf = cv2.imencode(".png", annotated)
    img64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return DetectAndDrawResponse(image_base64=img64, detections=dets)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("tiny", enum=list(MODELS.keys())),
):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("jpg", "jpeg", "png"):
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    data = await file.read()
    img = _read_image(data)
    boxes, confs, cls_ids, idxs = run_yolo(img, model)
    dets = [
        Detection(
            x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3],
            confidence=confs[i], class_name=CLASS_NAMES[cls_ids[i]]
        )
        for i, b in zip(idxs, [boxes[j] for j in idxs])
    ]
    annotated = draw_boxes(img.copy(), dets)
    # save & stream
    save_dir = BASE_DIR / "images_uploaded"
    save_dir.mkdir(exist_ok=True)
    out_path = save_dir / file.filename
    cv2.imwrite(str(out_path), annotated)
    with open(out_path, "rb") as f:
        return StreamingResponse(f, media_type="image/jpeg")
