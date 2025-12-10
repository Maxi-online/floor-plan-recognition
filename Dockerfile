# Floor Plan Recognition Service - Docker Image
# Базовый образ: Python 3.8 на Debian
FROM python:3.8-slim

# Метаданные
LABEL maintainer="Стреколовский Максим Владимирович"
LABEL description="Floor Plan Recognition Service with SAM2 + Hough Transform"
LABEL version="1.0"
LABEL client="ООО Refloor"

# Рабочая директория
WORKDIR /app

# Установка системных зависимостей для OpenCV и EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements.txt и установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY services/ ./services/
COPY tg_bot.py .
COPY settings.py .
COPY settings_secret.json .

# Создание директории для весов моделей
RUN mkdir -p /app/models

# Копирование весов SAM2 если есть (опционально, иначе скачается автоматически)
COPY sam2.1_l.pt ./sam2.1_l.pt 2>/dev/null || true

# Expose портов для сервисов
EXPOSE 8001 8002 8003

# Healthcheck для проверки работоспособности
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8003/health')" || exit 1

# Создание скрипта запуска всех сервисов
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🚀 Starting Floor Plan Recognition Services..."\n\
echo ""\n\
# Запуск сервисов в фоне\n\
uvicorn services.cleanup_service:app --host 0.0.0.0 --port 8001 &\n\
echo "✅ Cleanup Service started on port 8001"\n\
\n\
uvicorn services.ocr_service:app --host 0.0.0.0 --port 8002 &\n\
echo "✅ OCR Service started on port 8002"\n\
\n\
uvicorn services.hybrid_service:app --host 0.0.0.0 --port 8003 &\n\
echo "✅ Hybrid Service started on port 8003"\n\
\n\
# Telegram Bot в foreground (чтобы контейнер не завершился)\n\
echo "✅ Starting Telegram Bot..."\n\
echo ""\n\
python tg_bot.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Запуск всех сервисов
CMD ["/bin/bash", "/app/start.sh"]

