import io
import logging
import base64
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

# ——————————————————————————————————————————————————————————————————————————— #
#   PROJECT STRUCTURE:
#
#   .
#   ├── api/
#   │   └── main.py
#   ├── coco.names
#   ├── weights/
#   │   ├── yolov3-tiny.cfg
#   │   ├── yolov3-tiny.weights
#   │   ├── yolov3.cfg
#   │   └── yolov3.weights
#   └── requirements.txt
# ——————————————————————————————————————————————————————————————————————————— #

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

# ————————— Logger —————————————————————————————————————————————————————— #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detector")

# ————————— FastAPI Setup ——————————————————————————————————————————— #
app = FastAPI(
    title="YOLOv3 Detection API",
    version="1.0.0",
    description="Endpoints: bboxes, metadata, image, detect_and_draw_box, predict",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ————————— Load Class Names ——————————————————————————————————————— #
with open(COCO_NAMES_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

# ————————— Pydantic Models ——————————————————————————————————————— #
class BBox(BaseModel):
    x1: float; y1: float; x2: float; y2: float; confidence: float

class Detection(BaseModel):
    x1: float; y1: float; x2: float; y2: float
    confidence: float; class_name: str

class DetectionResponse(BaseModel):
    detections: List[Detection]

class DetectAndDrawResponse(BaseModel):
    image_base64: str
    detections: List[Detection]

# ————————— Thresholds & Caches ————————————————————————————————————— #
CONF_THRESH = 0.5
NMS_THRESH = 0.4

networks: Dict[str, cv2.dnn_Net] = {}
output_layers_map: Dict[str, List[str]] = {}
for name, paths in MODELS.items():
    net = cv2.dnn.readNetFromDarknet(str(paths["cfg"]), str(paths["weights"]))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    ln = net.getLayerNames()
    out_ids = net.getUnconnectedOutLayers().flatten()
    output_layers_map[name] = [ln[i - 1] for i in out_ids]
    networks[name] = net

# ————————— Helpers ————————————————————————————————————————————————— #
def _read_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image.")
    return img

def run_yolo(
    image: np.ndarray, model: str
) -> Tuple[List[List[int]], List[float], List[int], List[int]]:
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
                boxes .append([x, y, int(bw), int(bh)])
                confs .append(conf)
                cls_ids.append(cid)

    idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
    if hasattr(idxs, "flatten"):
        idxs = idxs.flatten().tolist()
    else:
        # older OpenCV sometimes returns list of tuples
        idxs = [i[0] if isinstance(i, (list,tuple,np.ndarray)) else i for i in idxs]  
    return boxes, confs, cls_ids, idxs

def draw_boxes(
    image: np.ndarray,
    dets: List[Detection]
) -> np.ndarray:
    h, w = image.shape[:2]
    thickness = max(2, int(0.003 * max(h, w)))
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = max(0.5, 0.001 * max(h, w))

    for det in dets:
        x1,y1 = int(det.x1), int(det.y1)
        x2,y2 = int(det.x2), int(det.y2)
        label = f"{det.class_name}: {det.confidence*100:.1f}%"

        # box
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), thickness)

        # text bg
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        cv2.rectangle(
            image,
            (x1, y1-th-baseline-4),
            (x1+tw+4, y1),
            (0,255,0),
            cv2.FILLED,
        )
        # text
        cv2.putText(
            image, label,
            (x1+2, y1-baseline-2),
            font, scale, (0,0,0),
            thickness=max(1, thickness//2),
            lineType=cv2.LINE_AA,
        )
    return image

# ————————— Routes ————————————————————————————————————————————————— #from fastapi.responses import HTMLResponse

@app.get("/", include_in_schema=False)
async def root() -> HTMLResponse:
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>🚀 YOLOv3 Detection API</title>
  <style>
    /* full-screen gradient background */
    body {
      margin: 0;
      height: 100vh;
      overflow: hidden;
      background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
      font-family: 'Segoe UI', sans-serif;
      color: #f0f0f0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    h1 {
      font-size: 3.5em;
      margin: 0.2em 0;
      text-shadow: 0 0 10px #66fcf1;
    }
    .links a {
      margin: 0 1em;
      color: #66fcf1;
      font-size: 1.2em;
      text-decoration: none;
      transition: color 0.3s;
    }
    .links a:hover {
      color: #45a29e;
    }
    footer {
      position: absolute;
      bottom: 20px;
      font-size: 0.9em;
      color: #888;
    }
    /* falling stars */
    .star {
      position: absolute;
      top: -10px;
      width: 2px;
      height: 2px;
      background: white;
      opacity: 0.8;
      border-radius: 50%;
      animation: fall linear infinite;
    }
    @keyframes fall {
      to {
        transform: translateY(110vh) translateX(var(--drift));
        opacity: 0;
      }
    }
  </style>
</head>
<body>

  <h1>🤖 YOLOv3 Detection API 🤖</h1>
  <div class="links">
    <a href="/docs">Interactive Docs</a>
    <a href="/redoc">ReDoc</a>
    <a href="/health">Health Check</a>
  </div>

  <footer>Made with ❤️ using FastAPI & YOLOv3 • Happy detecting!</footer>

  <script>
    // create a bunch of falling stars
    const stars = 80;
    for (let i = 0; i < stars; i++) {
      const star = document.createElement('div');
      star.classList.add('star');
      // random horizontal start
      star.style.left = Math.random() * 100 + 'vw';
      // random drift
      star.style.setProperty('--drift', (Math.random() * 200 - 100) + 'px');
      // random size and speed
      const scale = Math.random() * 1 + 0.3;
      star.style.width = star.style.height = (scale * 2) + 'px';
      const duration = Math.random() * 3 + 2;
      star.style.animationDuration = duration + 's';
      star.style.animationDelay = '-' + Math.random() * duration + 's';
      document.body.appendChild(star);
    }
  </script>

</body>
</html>
        """,
        status_code=200,
        media_type="text/html"
    )


@app.get("/health", tags=["health"])
async def health():
    return {"status":"ok"}

# 1) Bounding‐boxes only
@app.post("/detect/bboxes", response_model=List[BBox])
async def detect_bboxes(
    file: UploadFile = File(...),
    model: str        = Query("tiny", enum=list(MODELS.keys())),
):
    data = await file.read()
    img  = _read_image(data)
    boxes, confs, _, idxs = run_yolo(img, model)
    return [
        BBox(x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3], confidence=confs[i])
        for i,b in zip(idxs, [boxes[j] for j in idxs])
    ]

# 2) Metadata + class names
@app.post("/detect/metadata", response_model=DetectionResponse)
async def detect_metadata(
    file: UploadFile = File(...),
    model: str        = Query("tiny", enum=list(MODELS.keys())),
):
    data = await file.read()
    img  = _read_image(data)
    boxes, confs, cls_ids, idxs = run_yolo(img, model)
    dets = [
        Detection(
          x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3],
          confidence=confs[i], class_name=CLASS_NAMES[cls_ids[i]]
        )
        for i,b in zip(idxs, [boxes[j] for j in idxs])
    ]
    return DetectionResponse(detections=dets)

# 3) Annotated image only
@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    model: str        = Query("tiny", enum=list(MODELS.keys())),
):
    data = await file.read()
    img  = _read_image(data)
    boxes, confs, cls_ids, idxs = run_yolo(img, model)
    dets = [
        Detection( x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3],
                   confidence=confs[i], class_name=CLASS_NAMES[cls_ids[i]] )
        for i,b in zip(idxs, [boxes[j] for j in idxs])
    ]
    annotated = draw_boxes(img.copy(), dets)
    _, buf    = cv2.imencode(".png", annotated)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

# 4) Annotated + metadata
@app.post("/detect/detect_and_draw_box", response_model=DetectAndDrawResponse)
async def detect_and_draw_box(
    file: UploadFile = File(...),
    model: str        = Query("tiny", enum=list(MODELS.keys())),
):
    data = await file.read()
    img  = _read_image(data)
    boxes, confs, cls_ids, idxs = run_yolo(img, model)
    dets = [
        Detection( x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3],
                   confidence=confs[i], class_name=CLASS_NAMES[cls_ids[i]] )
        for i,b in zip(idxs, [boxes[j] for j in idxs])
    ]
    annotated = draw_boxes(img.copy(), dets)
    _, buf    = cv2.imencode(".png", annotated)
    return DetectAndDrawResponse(
        image_base64=base64.b64encode(buf.tobytes()).decode("utf-8"),
        detections=dets
    )

# 5) “Predict” → streams back a JPEG
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str        = Query("tiny", enum=list(MODELS.keys())),
):
    ext = file.filename.rsplit(".",1)[-1].lower()
    if ext not in ("jpg","jpeg","png"):
        raise HTTPException(415, "Unsupported file type.")
    data = await file.read()
    img  = _read_image(data)
    boxes, confs, cls_ids, idxs = run_yolo(img, model)
    dets = [
        Detection( x1=b[0], y1=b[1], x2=b[0]+b[2], y2=b[1]+b[3],
                   confidence=confs[i], class_name=CLASS_NAMES[cls_ids[i]] )
        for i,b in zip(idxs, [boxes[j] for j in idxs])
    ]
    annotated = draw_boxes(img.copy(), dets)
    # stream back as JPEG
    ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return StreamingResponse(io.BytesIO(jpg.tobytes()), media_type="image/jpeg")
