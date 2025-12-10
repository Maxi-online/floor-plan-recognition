@echo off
REM Quick launcher for Docker container (Windows)

echo ========================================
echo  Floor Plan Recognition Service
echo  Docker Container Launcher
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check if settings_secret.json exists
if not exist settings_secret.json (
    echo [ERROR] settings_secret.json not found!
    echo Please create it with your Telegram bot token.
    pause
    exit /b 1
)

echo [1/3] Building Docker image...
docker-compose build

if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Starting services...
docker-compose up -d

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start services!
    pause
    exit /b 1
)

echo.
echo [3/3] Services started successfully!
echo.
echo ========================================
echo  Services running on:
echo    - Cleanup:  http://localhost:8001
echo    - OCR:      http://localhost:8002
echo    - Hybrid:   http://localhost:8003
echo    - TG Bot:   Active
echo ========================================
echo.
echo Commands:
echo   - View logs:   docker-compose logs -f
echo   - Stop:        docker-compose down
echo   - Restart:     docker-compose restart
echo.

pause

