"""Settings Management Module.

Module for loading configuration parameters from settings_secret.json.
Contains Telegram bot token and service URLs.

settings_secret.json structure:
    {
        "telegram_bot_token": "YOUR_BOT_TOKEN",
        "hybrid_url": "http://localhost:8003/detect"
    }

Author: Maksim Strekolovsky
Date: 10.12.2025
Version: 1.0
"""
import json
from pathlib import Path
from typing import Any, Dict, Tuple


CONFIG_PATH = Path("settings_secret.json")


def load_settings() -> Tuple[str, str]:
    """Load settings from settings_secret.json.
    
    Reads configuration file and extracts:
    - Telegram bot token (required)
    - Hybrid service URL (default http://localhost:8003/detect)
    
    Returns:
        Tuple[str, str]: Tuple (telegram_bot_token, hybrid_url)
        
    Raises:
        RuntimeError: If file not found or telegram_bot_token missing
    """
    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"File {CONFIG_PATH} not found. Create JSON like "
            '{"telegram_bot_token":"<TOKEN>","hybrid_url":"http://localhost:8003/detect"}'
        )
    data: Dict[str, Any] = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    token = data.get("telegram_bot_token")
    if not token:
        raise RuntimeError("telegram_bot_token field missing in settings_secret.json.")
    hybrid_url = data.get("hybrid_url", "http://localhost:8003/detect")
    return str(token), str(hybrid_url)

