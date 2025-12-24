# Docker Deployment Guide

Quick guide for running the project in a Docker container.

## Requirements

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Docker Compose** v2.0+
- **8GB RAM** minimum (16GB recommended)
- **10GB** free disk space

## Quick Start

### Windows:

```bash
# Start
docker-run.bat

# Stop
docker-stop.bat
```

### Linux/Mac:

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# Stop
docker-compose down
```

## Project Structure

```
room-detector/
├── Dockerfile              # Image with Python + dependencies
├── docker-compose.yml      # Service orchestration
├── .dockerignore          # Build exclusions
├── docker-run.bat         # Quick start (Windows)
├── docker-stop.bat        # Quick stop (Windows)
└── settings_secret.json   # Configuration (Telegram bot token)
```

## Configuration

### 1. Telegram Bot Token

Create/update `settings_secret.json`:

```json
{
  "telegram_bot_token": "YOUR_BOT_TOKEN",
  "hybrid_url": "http://localhost:8003/detect"
}
```

### 2. Ports

Default ports:
- `8001` - Cleanup Service
- `8002` - OCR Service  
- `8003` - Hybrid Service (main)

Can be changed in `docker-compose.yml`:

```yaml
ports:
  - "8001:8001"  # host:container
```

## What Happens During Build

1. **Base image**: Python 3.8-slim
2. **System dependencies**: OpenCV, GL libraries
3. **Python packages**: SAM2, EasyOCR, FastAPI, etc.
4. **Model weights**: SAM 2.1 Large (224MB) copied or downloaded
5. **Auto-start**: All 4 services start automatically

## Useful Commands

```bash
# View logs in real-time
docker-compose logs -f

# View logs for specific service
docker-compose logs -f floor-plan-service

# Restart services
docker-compose restart

# Enter container for debugging
docker-compose exec floor-plan-service bash

# Check status
docker-compose ps

# Clean up old images
docker system prune -a
```

## Health Check

After starting the container, check:

```bash
# Health check Hybrid Service
curl http://localhost:8003/health

# List services
docker-compose ps
```

Expected response from `/health`:
```json
{
  "status": "ok",
  "sam2_large": true
}
```

## Troubleshooting

### Container won't start

```bash
# View logs
docker-compose logs

# Rebuild image from scratch
docker-compose build --no-cache
```

### Out of memory

Increase limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 16G
```

### Ports in use

Change ports in `docker-compose.yml` to available ones.

## Updating

```bash
# Stop
docker-compose down

# Rebuild with latest changes
docker-compose build

# Start
docker-compose up -d
```

## Data Persistence

Model weights are cached in Docker volumes:
- `model-cache` - SAM2 weights
- `easyocr-cache` - EasyOCR models

This speeds up subsequent container starts.

## Production Deployment

For production, it's recommended to:

1. **Use GPU** (if available):
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

2. **Configure reverse proxy** (nginx/traefik)
3. **Add HTTPS** via Let's Encrypt
4. **Monitoring** via Prometheus/Grafana
5. **Backups** of configuration and volumes
