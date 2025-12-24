# Quick Start Guide

Get started in 3 minutes!

## Requirements

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Telegram Bot Token ([get from @BotFather](https://t.me/BotFather))

## Step by Step

### 1. Get Telegram Bot Token

```
1. Open @BotFather in Telegram
2. Send /newbot
3. Follow the instructions
4. Copy the token (e.g.: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz)
```

### 2. Clone the Project

```bash
git clone <repository-url>
cd room-detector
```

### 3. Create Configuration

Create `settings_secret.json` file:

```json
{
  "telegram_bot_token": "YOUR_TOKEN_HERE",
  "hybrid_url": "http://localhost:8003/detect"
}
```

### 4. Start Docker

**Windows:**
```bash
docker-run.bat
```

**Linux/Mac:**
```bash
docker-compose up -d
```

### 5. Ready!

Open your bot in Telegram and send /start

## Usage

1. Send a floor plan image to the bot
2. Wait 10-20 seconds
3. Get visualization + JSON

## Health Check

```bash
# Check that services are running
docker-compose ps

# View logs
docker-compose logs -f

# Check health
curl http://localhost:8003/health
```

Should return:
```json
{"status": "ok", "sam2_large": true}
```

## Troubleshooting

### Docker won't start?

```bash
# Make sure Docker Desktop is running
docker info

# If you see an error - restart Docker Desktop
```

### Bot not responding?

```bash
# Check logs
docker-compose logs floor-plan-service

# Check that token is correct
cat settings_secret.json
```

### Ports in use?

Change ports in `docker-compose.yml`:
```yaml
ports:
  - "9001:8001"  # instead of 8001:8001
  - "9002:8002"  # instead of 8002:8002
  - "9003:8003"  # instead of 8003:8003
```

## More Information

- [Full documentation](README.md)
- [Docker guide](DOCKER.md)

---

**Need help?** Check logs: `docker-compose logs -f`
