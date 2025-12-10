"""Settings Management Module.

Модуль для загрузки конфигурационных параметров из settings_secret.json.
Содержит токен Telegram бота и URL сервисов.

Структура settings_secret.json:
    {
        "telegram_bot_token": "YOUR_BOT_TOKEN",
        "hybrid_url": "http://localhost:8003/detect"
    }

Автор: Стреколовский Максим Владимирович
Заказчик: ООО Refloor
Дата: 10.12.2025
Версия: 1.0
"""
import json
from pathlib import Path
from typing import Any, Dict, Tuple


CONFIG_PATH = Path("settings_secret.json")


def load_settings() -> Tuple[str, str]:
    """Загрузка настроек из settings_secret.json.
    
    Читает конфигурационный файл и извлекает:
    - Telegram bot token (обязательный)
    - Hybrid service URL (по умолчанию http://localhost:8003/detect)
    
    Returns:
        Tuple[str, str]: Кортеж (telegram_bot_token, hybrid_url)
        
    Raises:
        RuntimeError: Если файл не найден или отсутствует telegram_bot_token
    """
    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"Файл {CONFIG_PATH} не найден. Создайте JSON вида "
            '{"telegram_bot_token":"<TOKEN>","hybrid_url":"http://localhost:8003/detect"}'
        )
    data: Dict[str, Any] = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    token = data.get("telegram_bot_token")
    if not token:
        raise RuntimeError("В settings_secret.json отсутствует поле telegram_bot_token.")
    hybrid_url = data.get("hybrid_url", "http://localhost:8003/detect")
    return str(token), str(hybrid_url)

