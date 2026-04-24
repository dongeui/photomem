FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-kor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install PyTorch CPU-only first (avoids pulling CUDA, saves ~2GB)
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

RUN mkdir -p /app/cache /data/photos

ENV PHOTOMEM_DB=/app/cache/photomem.db
ENV PHOTOMEM_MODELS=/app/cache/models
ENV PHOTOMEM_CACHE=/app/cache
ENV PHOTOMEM_PHOTOS=/data/photos
ENV PHOTOMEM_INDEX_WORKERS=1
ENV PHOTOMEM_TORCH_THREADS=
ENV HF_HOME=/app/cache/huggingface
ENV TORCH_HOME=/app/cache/torch
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
