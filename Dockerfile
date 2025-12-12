# syntax=docker/dockerfile:1.7

FROM --platform=linux/amd64 python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential pkg-config \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        liblapack-dev libblas-dev gfortran \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data /root/.deepface/weights

COPY . /app

ENV PORT=8080

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} app:app --workers 2 --threads 8 --timeout 120"]