# Quick Start Guide

Запуск за 3 минуты!

## Что нужно

- Docker Desktop (Windows/Mac) или Docker Engine (Linux)
- Telegram Bot Token ([получить от @BotFather](https://t.me/BotFather))

## Шаг за шагом

### 1. Получите Telegram Bot Token

```
1. Откройте @BotFather в Telegram
2. Отправьте /newbot
3. Следуйте инструкциям
4. Скопируйте token (например: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz)
```

### 2. Склонируйте проект

```bash
git clone <repository-url>
cd room-detector
```

### 3. Создайте конфигурацию

Создайте файл `settings_secret.json`:

```json
{
  "telegram_bot_token": "ВАШ_ТОКЕН_СЮДА",
  "hybrid_url": "http://localhost:8003/detect"
}
```

### 4. Запустите Docker

**Windows:**
```bash
docker-run.bat
```

**Linux/Mac:**
```bash
docker-compose up -d
```

### 5. Готово!

Откройте вашего бота в Telegram и отправьте /start

## Использование

1. Отправьте изображение плана в бот
2. Подождите 10-20 секунд
3. Получите визуализацию + JSON

## Проверка работы

```bash
# Проверить что сервисы запущены
docker-compose ps

# Посмотреть логи
docker-compose logs -f

# Проверить health
curl http://localhost:8003/health
```

Должно вернуть:
```json
{"status": "ok", "sam2_large": true}
```

## Если что-то не работает

### Docker не запускается?
```bash
# Убедитесь что Docker Desktop запущен
docker info

# Если видите ошибку - перезапустите Docker Desktop
```

### Бот не отвечает?
```bash
# Проверьте логи
docker-compose logs floor-plan-service

# Проверьте что токен правильный
cat settings_secret.json
```

### Порты заняты?
Измените порты в `docker-compose.yml`:
```yaml
ports:
  - "9001:8001"  # вместо 8001:8001
  - "9002:8002"  # вместо 8002:8002
  - "9003:8003"  # вместо 8003:8003
```

## Больше информации

- [Полная документация](README.md)
- [Docker гайд](DOCKER.md)

---

**Нужна помощь?** Проверьте логи: `docker-compose logs -f`
