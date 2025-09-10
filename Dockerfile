# Use a base image with Python 3.9 or higher
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire repository into the container
COPY . /app

# Create a non-root user for safety
RUN groupadd -r tensorpack && useradd -r -g tensorpack tensorpack

# Install system dependencies needed by common scientific packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       g++ \
       git \
       curl \
       libgl1 \
       libglib2.0-0 \
       libsm6 \
       libxrender1 \
       libxext6 \
       ffmpeg \
       libsndfile1 \
       libsndfile1-dev \
       libopenblas-dev \
       liblapack-dev \
       pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is up to date and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt || true

# Some heavy packages (torch) may need a separate install or a specific wheel; attempt a common CPU install
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu || echo "torch install skipped"

# Create logs and data directories and switch to non-root user
RUN mkdir -p /app/exports /app/.tensorpack && chown -R tensorpack:tensorpack /app
USER tensorpack

# Default command
CMD ["python", "tensorpack.py"]