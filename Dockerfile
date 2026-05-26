# ===========================================================================
# CIDeconvolve — BIAFLOWS-compatible GPU-enabled Docker image
# ===========================================================================
# Base: Python slim. GPU support comes from the CUDA-enabled PyTorch wheel
# plus the host NVIDIA driver mounted by NVIDIA Container Toolkit.
#
# BIAFLOWS convention: images in /data/in, results in /data/out,
# ground truth in /data/gt.  The entrypoint is wrapper.py which
# parses --infolder / --outfolder / --gtfolder and descriptor.json
# parameters, then delegates to deconvolve.py.
# ===========================================================================

FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# --- System packages ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        fonts-dejavu-core \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ---
COPY requirements_docker.txt /app/requirements_docker.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-compile -r requirements_docker.txt

# --- Application code ---
COPY core/ /app/core/
COPY bilayers_local.py /app/bilayers_local.py
COPY wrapper_bl.py /app/wrapper_bl.py
COPY bilayers_config.yaml /app/bilayers_config.yaml

# --- BIAFLOWS data directories ---
RUN mkdir -p /data/in /data/out /data/gt

# Expose NVIDIA GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["python", "/app/wrapper_bl.py"]
