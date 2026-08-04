# syntax=docker/dockerfile:1

# ────────────────────────────────────────────────────────────
# Base image: CUDA 13 runtime + cuDNN, matching torch==2.12.0
# (nvidia-cudnn-cu13, nvidia-cublas etc. in requirements.txt are the
# *Python* CUDA wheels; the container still needs the matching CUDA
# driver-facing runtime libs from an nvidia/cuda base image to actually
# talk to the GPU via nvidia-container-toolkit).
# ────────────────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

# ────────────────────────────────────────────────────────────
# System dependencies
#   - python3.14 (matches the dev venv exactly — not in Ubuntu 22.04's
#     default apt repos, so pulled from the deadsnakes PPA)
#   - tesseract-ocr: required by pytesseract (image_parser.py, pdf_parser.py OCR fallback)
#   - poppler-utils: required by pdfplumber/pypdfium2 for some PDF operations
#   - libgl1 / libglib2.0-0: required by opencv-python (image_parser.py)
#   - libmagic1: used by python-magic-style filetype detection (docling/filetype)
#   - build-essential: some packages (e.g. rtree, shapely) may need to compile
# ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.14 \
    python3.14-venv \
    python3.14-dev \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# python3.14 has no pip module preinstalled via apt on some deadsnakes
# builds — bootstrap it with ensurepip rather than relying on python3-pip
# (which would pull in the system's default python, not 3.14).
RUN python3.14 -m ensurepip --upgrade

# Make python3.14 the default `python`/`pip`
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.14 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3.14 1

WORKDIR /app

# ────────────────────────────────────────────────────────────
# Python dependencies — copied separately from source so Docker
# caches this (slow, ~heavy torch/cuda install) as its own layer,
# only rebuilding it when requirements.txt actually changes.
# ────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ────────────────────────────────────────────────────────────
# Application source
# ────────────────────────────────────────────────────────────
COPY app/ ./app/

# Directories your app writes to at runtime (uploads, chroma persistence,
# audit logs) — these get bind-mounted/volumed in compose, but declaring
# them here means `docker build` won't fail if code expects them to exist.
RUN mkdir -p /app/data/uploads /app/data/chroma /app/data/logs

EXPOSE 8000

# Basic healthcheck hitting your own /health route
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]