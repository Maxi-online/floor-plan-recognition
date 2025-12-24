# Floor Plan Recognition Service - Docker Image
# Base image: Python 3.8 on Debian
FROM python:3.8-slim

# Metadata
LABEL maintainer="Maksim Strekolovsky"
LABEL description="Floor Plan Recognition Service with SAM2 + Hough Transform"
LABEL version="1.0"

# Working directory
WORKDIR /app

# Install system dependencies for OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY services/ ./services/
COPY tg_bot.py .
COPY settings.py .
COPY settings_secret.json .

# Create directory for model weights
RUN mkdir -p /app/models

# Expose ports for services
EXPOSE 8001 8002 8003

# Healthcheck for service availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8003/health')" || exit 1

# Create script to start all services
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🚀 Starting Floor Plan Recognition Services..."\n\
echo ""\n\
# Start services in background\n\
uvicorn services.cleanup_service:app --host 0.0.0.0 --port 8001 &\n\
echo "✅ Cleanup Service started on port 8001"\n\
\n\
uvicorn services.ocr_service:app --host 0.0.0.0 --port 8002 &\n\
echo "✅ OCR Service started on port 8002"\n\
\n\
uvicorn services.hybrid_service:app --host 0.0.0.0 --port 8003 &\n\
echo "✅ Hybrid Service started on port 8003"\n\
\n\
# Telegram Bot in foreground (so container doesn't exit)\n\
echo "✅ Starting Telegram Bot..."\n\
echo ""\n\
python tg_bot.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Start all services
CMD ["/bin/bash", "/app/start.sh"]

