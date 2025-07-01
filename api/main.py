import io
import logging
import base64
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .events import EventBus

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent
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
    description="Routes: bboxes, metadata, annotated image, detect_and_draw_box, unified predict without cvlib",
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

# Pydantic models for responses
class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

class Detection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str

class DetectionResponse(BaseModel):
    detections: List[Detection]

class DetectAndDrawResponse(BaseModel):
    image_base64: str
    detections: List[Detection]

# Shared thresholds
CONF_THRESH = 0.5
NMS_THRESH = 0.4

# Cached YOLO networks
networks: Dict[str, cv2.dnn_Net] = {}
output_layers_map: Dict[str, List[str]] = {}
for name, paths in MODELS.items():
    net = cv2.dnn.readNetFromDarknet(str(paths['cfg']), str(paths['weights']))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    networks[name] = net
    output_layers_map[name] = out_layers

# Helper: inference
def run_yolo(image: np.ndarray, model: str) -> Tuple[List[List[int]], List[float], List[int], List[int]]:
    net = networks[model]
    output_layers = output_layers_map[model]
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
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
    if hasattr(idxs, 'flatten'):
        idx_list = idxs.flatten().tolist()
    elif isinstance(idxs, (list, tuple)) and idxs:
        idx_list = idxs[0] if isinstance(idxs[0], (list, np.ndarray)) else idxs
    else:
        idx_list = []
    return boxes, confs, cls_ids, idx_list

# Helper: read image bytes
def _read_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image.")
    return img

# 1. Bounding boxes only
@app.post("/detect/bboxes", response_model=List[BBox])
async def detect_bboxes(
    file: UploadFile = File(...),
    model: str = Query("tiny", enum=list(MODELS.keys())),
):
    data = await file.read()
    img = _read_image(data)
    boxes, confs, _, idxs = run_yolo(img, model)
    return [BBox(x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3], confidence=confs[i]) for i in idxs]

# 2. Metadata with class names
@app.post("/detect/metadata", response_model=DetectionResponse)
async def detect_metadata(
    file: UploadFile = File(...),
    model: str = Query("tiny", enum=list(MODELS.keys())),
):
    data = await file.read()
    img = _read_image(data)
    boxes, confs, cls_ids, idxs = run_yolo(img, model)
    detections = [
        Detection(
            x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3],
            confidence=confs[i], class_name=CLASS_NAMES[cls_ids[i]]
        ) for i, b in zip(idxs, [boxes[j] for j in idxs])
    ]
    return DetectionResponse(detections=detections)

# 3. Annotated image
@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    model: str = Query("tiny", enum=list(MODELS.keys())),
):
    data = await file.read()
    img = _read_image(data)
    boxes, confs, cls_ids, idxs = run_yolo(img, model)
    for i in idxs:
        x, y, bw, bh = boxes[i]
        cv2.rectangle(img, (x, y), (x+bw, y+bh), (0,255,0), 2)
        label = f"{CLASS_NAMES[cls_ids[i]]}: {confs[i]:.2f}"
        cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    _, buf = cv2.imencode('.png', img)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type='image/png')

# 4. Detect and draw with metadata
@app.post("/detect/detect_and_draw_box", response_model=DetectAndDrawResponse)
async def detect_and_draw_box(
    file: UploadFile = File(...),
    model: str = Query("tiny", enum=list(MODELS.keys())),
):
    data = await file.read()
    img = _read_image(data)
    boxes, confs, cls_ids, idxs = run_yolo(img, model)
    detections: List[Detection] = []
    for i in idxs:
        b = boxes[i]
        detections.append(
            Detection(
                x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3],
                confidence=confs[i], class_name=CLASS_NAMES[cls_ids[i]]
            )
        )
        cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (255,0,0), 2)
        cv2.putText(img, f"{CLASS_NAMES[cls_ids[i]]}", (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    _, buf = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    return DetectAndDrawResponse(image_base64=img_base64, detections=detections)

# 5. Unified predict endpoint returning image stream
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("tiny", enum=list(MODELS.keys())),
):
    # 1. Validate extension
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ("jpg", "jpeg", "png"):
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    # 2. Read and decode image
    data = await file.read()
    img = _read_image(data)

    # 3. Run YOLO inference and draw boxes
    boxes, confs, cls_ids, idxs = run_yolo(img, model)
    for i in idxs:
        b = boxes[i]
        # draw rectangle and label
        cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0,0,255), 2)
        label = f"{CLASS_NAMES[cls_ids[i]]}: {confs[i]:.2f}"
        cv2.putText(img, label, (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # 4. Save annotated image to server
    save_dir = BASE_DIR / 'images_uploaded'
    save_dir.mkdir(exist_ok=True)
    out_path = save_dir / file.filename
    cv2.imwrite(str(out_path), img)

    # 5. Stream the response back to the client
    file_image = open(str(out_path), mode='rb')
    return StreamingResponse(file_image, media_type="image/jpeg")
