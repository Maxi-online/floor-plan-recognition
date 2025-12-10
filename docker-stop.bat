@echo off
REM Quick stop script for Docker container (Windows)

echo ========================================
echo  Stopping Floor Plan Recognition Service
echo ========================================
echo.

docker-compose down

if %errorlevel% neq 0 (
    echo [ERROR] Failed to stop services!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] All services stopped!
echo.

pause

