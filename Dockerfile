FROM python:3.11-slim

# 1) system deps for headless OpenCV wheel
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy code + assets
COPY api/      ./api/
COPY coco.names .
COPY weights/  ./weights/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

# 4) Final entry
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
