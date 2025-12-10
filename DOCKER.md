# Docker Deployment Guide

Краткое руководство по запуску проекта в Docker контейнере.

## Требования

- **Docker Desktop** (Windows/Mac) или **Docker Engine** (Linux)
- **Docker Compose** v2.0+
- **8GB RAM** минимум (рекомендуется 16GB)
- **10GB** свободного места на диске

## Быстрый старт

### Windows:

```bash
# Запуск
docker-run.bat

# Остановка
docker-stop.bat
```

### Linux/Mac:

```bash
# Сборка образа
docker-compose build

# Запуск сервисов
docker-compose up -d

# Остановка
docker-compose down
```

## Структура проекта

```
room-detector/
├── Dockerfile              # Образ с Python + зависимости
├── docker-compose.yml      # Оркестрация сервисов
├── .dockerignore          # Исключения для сборки
├── docker-run.bat         # Быстрый запуск (Windows)
├── docker-stop.bat        # Быстрая остановка (Windows)
└── settings_secret.json   # Конфигурация (токен TG бота)
```

## Конфигурация

### 1. Telegram Bot Token

Создайте/обновите `settings_secret.json`:

```json
{
  "telegram_bot_token": "YOUR_BOT_TOKEN",
  "hybrid_url": "http://localhost:8003/detect"
}
```

### 2. Порты

По умолчанию используются порты:
- `8001` - Cleanup Service
- `8002` - OCR Service  
- `8003` - Hybrid Service (main)

Изменить можно в `docker-compose.yml`:

```yaml
ports:
  - "8001:8001"  # host:container
```

## Что происходит при сборке

1. **Базовый образ**: Python 3.8-slim
2. **Системные зависимости**: OpenCV, GL libraries
3. **Python пакеты**: SAM2, EasyOCR, FastAPI, и др.
4. **Веса моделей**: SAM 2.1 Large (224MB) копируется или скачивается
5. **Автозапуск**: Все 4 сервиса стартуют автоматически

## Полезные команды

```bash
# Просмотр логов в реальном времени
docker-compose logs -f

# Просмотр логов конкретного сервиса
docker-compose logs -f floor-plan-service

# Перезапуск сервисов
docker-compose restart

# Вход в контейнер для отладки
docker-compose exec floor-plan-service bash

# Проверка статуса
docker-compose ps

# Очистка старых образов
docker system prune -a
```

## Проверка работоспособности

После запуска контейнера проверьте:

```bash
# Health check Hybrid Service
curl http://localhost:8003/health

# Список сервисов
docker-compose ps
```

Ожидаемый ответ от `/health`:
```json
{
  "status": "ok",
  "sam2_large": true
}
```

## Troubleshooting

### Контейнер не запускается

```bash
# Просмотр логов
docker-compose logs

# Пересборка образа с нуля
docker-compose build --no-cache
```

### Нехватка памяти

Увеличьте лимиты в `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 16G
```

### Порты заняты

Измените порты в `docker-compose.yml` на свободные.

## Обновление

```bash
# Остановка
docker-compose down

# Пересборка с последними изменениями
docker-compose build

# Запуск
docker-compose up -d
```

## Персистентность данных

Веса моделей кэшируются в Docker volumes:
- `model-cache` - SAM2 weights
- `easyocr-cache` - EasyOCR models

Это ускоряет повторные запуски контейнера.

## Production Deployment

Для production рекомендуется:

1. **Использовать GPU** (если доступен):
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

2. **Настроить reverse proxy** (nginx/traefik)
3. **Добавить HTTPS** через Let's Encrypt
4. **Мониторинг** через Prometheus/Grafana
5. **Backups** конфигурации и volumes
