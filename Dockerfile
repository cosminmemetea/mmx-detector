# ---------- Builder Stage ----------
FROM ros:humble-ros-core AS builder

# Install Python venv, pip & OpenCV build deps (if needed)
RUN apt-get update && apt-get install -y \
      python3-venv python3-pip python3-opencv curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create & activate a venv
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy and install Python dependencies, including gunicorn & uvicorn
COPY requirements.txt .
# Ensure gunicorn and uvicorn are installed
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn uvicorn

# ---------- Runtime Stage ----------
FROM ros:humble-ros-core

# Runtime-only system deps
RUN apt-get update && apt-get install -y \
      curl python3-opencv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the venv from the builder
COPY --from=builder /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy app code, labels, weights
COPY api/ api/
COPY coco.names .
COPY weights/ weights/

# Prepare output dir and drop to non-root user
RUN mkdir -p images_uploaded \
    && useradd -m appuser \
    && chown -R appuser:appuser /app

USER appuser

# Expose HTTP port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch under Gunicorn+Uvicorn workers
CMD ["gunicorn", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000", \
     "--workers", "4", \
     "api.main:app"]
