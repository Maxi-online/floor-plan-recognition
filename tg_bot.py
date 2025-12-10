"""Telegram Bot для распознавания планов помещений.

Telegram бот предоставляет удобный интерфейс для работы с системой
распознавания планов помещений. Принимает изображения планов и возвращает
визуализацию + JSON с координатами стен, комнат и размеров.

Основные возможности:
    - Приём изображений планов (JPG/PNG)
    - Параллельная обработка (Hybrid Service + OCR Service)
    - Визуализация результатов (стены, комнаты, размеры)
    - Отправка JSON файла с координатами

Технологический стек:
    - python-telegram-bot для Telegram API
    - httpx для асинхронных HTTP запросов
    - OpenCV для визуализации результатов

Автор: Стреколовский Максим Владимирович
Заказчик: ООО Refloor
Дата: 10.12.2025
Версия: 1.0
"""
import asyncio
import io
import json
from pathlib import Path
from typing import Any, Dict

import cv2
import httpx
import numpy as np
from telegram import Update
from telegram.error import Conflict
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from settings import load_settings


async def send_to_hybrid(image_bytes: bytes, url: str) -> Dict[str, Any]:
    """Отправка изображения в Hybrid Service для детекции стен и комнат.
    
    Использует SAM 2.1 Large + Hough Transform для распознавания структуры плана.
    
    Args:
        image_bytes: Байты изображения плана
        url: URL Hybrid Service
        
    Returns:
        Dict[str, Any]: JSON с результатами распознавания:
            - "meta": метаданные
            - "walls": список стен
            - "rooms": список комнат
            
    Raises:
        httpx.HTTPStatusError: При ошибке HTTP запроса
    """
    async with httpx.AsyncClient(timeout=1200) as client:  # 20 минут для SAM2
        files = {"file": ("plan.png", image_bytes, "image/png")}
        resp = await client.post(url, files=files)
        resp.raise_for_status()
        return resp.json()


async def send_to_ocr(image_bytes: bytes, base_url: str) -> Dict[str, Any]:
    """Отправка изображения в OCR Service для распознавания текста.
    
    Использует EasyOCR для распознавания размеров и площадей комнат на плане.
    
    Args:
        image_bytes: Байты изображения плана
        base_url: Базовый URL (не используется, OCR всегда localhost:8002)
        
    Returns:
        Dict[str, Any]: JSON с результатами OCR:
            - "items": список распознанных элементов
            - "model": название модели ("EasyOCR")
            
    Note:
        При ошибке возвращает пустой список items
    """
    ocr_url = "http://localhost:8002/ocr"  # Фиксированный URL для OCR
    async with httpx.AsyncClient(timeout=120) as client:
        files = {"file": ("plan.png", image_bytes, "image/png")}
        resp = await client.post(ocr_url, files=files)
        if resp.status_code == 200:
            return resp.json()
        return {"items": []}


def visualize_result(image_bytes: bytes, result_json: Dict[str, Any]) -> bytes:
    """Визуализация результатов распознавания на изображении плана.
    
    Рисует поверх оригинального плана:
    - Стены (красные линии с ID)
    - Комнаты (зелёные полигоны с ID)
    - OCR размеры (синие bbox с текстом)
    
    Args:
        image_bytes: Байты оригинального изображения
        result_json: JSON с результатами распознавания
        
    Returns:
        bytes: Изображение с визуализацией в PNG формате
    """
    # Декодируем изображение
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Рисуем стены (красный)
    for wall in result_json.get("walls", []):
        points = wall["points"]
        if len(points) >= 2:
            for i in range(len(points) - 1):
                pt1 = tuple(points[i])
                pt2 = tuple(points[i + 1])
                cv2.line(img, pt1, pt2, (0, 0, 255), 3)
            # ID стены
            mid_x = int(np.mean([p[0] for p in points]))
            mid_y = int(np.mean([p[1] for p in points]))
            cv2.putText(
                img, wall["id"], (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
            )
    
    # Кодируем обратно в bytes
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start.
    
    Отправляет приветственное сообщение с инструкцией по использованию бота.
    
    Args:
        update: Telegram Update объект
        context: Контекст приложения
    """
    text = (
        "🏠 Бот распознавания планов помещений\n\n"
        "Отправьте фото плана (JPG/PNG) — я верну:\n"
        "• Визуализацию с найденными стенами\n"
        "• JSON файл с координатами стен\n\n"
        "🤖 Использую Hough Transform\n"
        "⏱️ Обработка занимает 10-20 секунд"
    )
    await update.message.reply_text(text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик фотографий/изображений.
    
    Основная логика бота:
    1. Получение изображения от пользователя
    2. Параллельная отправка в Hybrid Service и OCR Service
    3. Визуализация результатов
    4. Отправка пользователю визуализации + JSON
    
    Args:
        update: Telegram Update объект с фото
        context: Контекст приложения с bot_data
    """
    if not update.message or not update.message.photo:
        return
    token, hybrid_url = context.bot_data["token"], context.bot_data["hybrid_url"]
    photo = update.message.photo[-1]
    file = await photo.get_file()
    image_bytes = bytes(await file.download_as_bytearray())

    try:
        # Уведомляем пользователя о начале обработки
        await update.message.reply_text(
            "🔄 Обработка началась...\n"
            "⏱️ Займёт 10-20 секунд (Hough Transform)"
        )
        
        # Отправляем в Hybrid Service
        payload = await send_to_hybrid(image_bytes, hybrid_url)
        
        # Визуализация результата
        viz_image = visualize_result(image_bytes, payload)
        
        # Отправляем статистику
        stats = (
            f"📊 Распознано:\n"
            f"🔴 Стен: {len(payload.get('walls', []))}\n\n"
            f"🤖 Hough Transform"
        )
        
        # Отправляем визуализацию + JSON
        await update.message.reply_photo(
            photo=io.BytesIO(viz_image),
            caption=stats,
        )
        
        json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        await update.message.reply_document(
            document=io.BytesIO(json_bytes),
            filename="result.json",
            caption="📄 JSON с координатами",
        )
    except Exception as exc:  # noqa: BLE001
        await update.message.reply_text(f"❌ Ошибка: {exc}")


def build_app() -> Application:
    """Создание и конфигурация Telegram бота.
    
    Загружает настройки из settings_secret.json,
    создаёт Application с обработчиками команд и сообщений.
    
    Returns:
        Application: Сконфигурированный Telegram бот
    """
    token, hybrid_url = load_settings()
    app = ApplicationBuilder().token(token).build()
    app.bot_data["token"] = token
    app.bot_data["hybrid_url"] = hybrid_url
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    return app


def main() -> None:
    """Точка входа в приложение.
    
    Создаёт и запускает Telegram бота в режиме polling.
    Обрабатывает Conflict исключение (если бот уже запущен).
    """
    app = build_app()
    try:
        app.run_polling()
    except Conflict:
        print(
            "⚠️ Telegram bot conflict: "
            "another instance is running. Stop it before starting a new one."
        )
        raise SystemExit(0)


if __name__ == "__main__":
    main()

