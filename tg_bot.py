"""Telegram Bot for Floor Plan Recognition.

Telegram bot provides a convenient interface for working with the floor plan
recognition system. Accepts floor plan images and returns visualization + JSON
with coordinates of walls, rooms, and dimensions.

Main features:
    - Accept floor plan images (JPG/PNG)
    - Parallel processing (Hybrid Service + OCR Service)
    - Result visualization (walls, rooms, dimensions)
    - Send JSON file with coordinates

Technology stack:
    - python-telegram-bot for Telegram API
    - httpx for async HTTP requests
    - OpenCV for result visualization

Author: Maksim Strekolovsky
Date: 10.12.2025
Version: 1.0
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
    """Send image to Hybrid Service for wall and room detection.
    
    Uses SAM 2.1 Large + Hough Transform for plan structure recognition.
    
    Args:
        image_bytes: Floor plan image bytes
        url: Hybrid Service URL
        
    Returns:
        Dict[str, Any]: JSON with recognition results:
            - "meta": metadata
            - "walls": list of walls
            - "rooms": list of rooms
            
    Raises:
        httpx.HTTPStatusError: On HTTP request error
    """
    async with httpx.AsyncClient(timeout=1200) as client:  # 20 minutes for SAM2
        files = {"file": ("plan.png", image_bytes, "image/png")}
        resp = await client.post(url, files=files)
        resp.raise_for_status()
        return resp.json()


async def send_to_ocr(image_bytes: bytes, base_url: str) -> Dict[str, Any]:
    """Send image to OCR Service for text recognition.
    
    Uses EasyOCR to recognize dimensions and room areas on the plan.
    
    Args:
        image_bytes: Floor plan image bytes
        base_url: Base URL (not used, OCR always localhost:8002)
        
    Returns:
        Dict[str, Any]: JSON with OCR results:
            - "items": list of recognized elements
            - "model": model name ("EasyOCR")
            
    Note:
        Returns empty items list on error
    """
    ocr_url = "http://localhost:8002/ocr"  # Fixed URL for OCR
    async with httpx.AsyncClient(timeout=120) as client:
        files = {"file": ("plan.png", image_bytes, "image/png")}
        resp = await client.post(ocr_url, files=files)
        if resp.status_code == 200:
            return resp.json()
        return {"items": []}


def visualize_result(image_bytes: bytes, result_json: Dict[str, Any]) -> bytes:
    """Visualize recognition results on the floor plan image.
    
    Draws on top of the original plan:
    - Walls (red lines with ID)
    - Rooms (green polygons with ID)
    - OCR dimensions (blue bbox with text)
    
    Args:
        image_bytes: Original image bytes
        result_json: JSON with recognition results
        
    Returns:
        bytes: Image with visualization in PNG format
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Draw walls (red)
    for wall in result_json.get("walls", []):
        points = wall["points"]
        if len(points) >= 2:
            for i in range(len(points) - 1):
                pt1 = tuple(points[i])
                pt2 = tuple(points[i + 1])
                cv2.line(img, pt1, pt2, (0, 0, 255), 3)
            # Wall ID
            mid_x = int(np.mean([p[0] for p in points]))
            mid_y = int(np.mean([p[1] for p in points]))
            cv2.putText(
                img, wall["id"], (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
            )
    
    # Encode back to bytes
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /start command.
    
    Sends welcome message with bot usage instructions.
    
    Args:
        update: Telegram Update object
        context: Application context
    """
    text = (
        "🏠 Floor Plan Recognition Bot\n\n"
        "Send a floor plan photo (JPG/PNG) — I'll return:\n"
        "• Visualization with detected walls\n"
        "• JSON file with wall coordinates\n\n"
        "🤖 Using Hough Transform\n"
        "⏱️ Processing takes 10-20 seconds"
    )
    await update.message.reply_text(text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for photos/images.
    
    Main bot logic:
    1. Get image from user
    2. Parallel sending to Hybrid Service and OCR Service
    3. Result visualization
    4. Send visualization + JSON to user
    
    Args:
        update: Telegram Update object with photo
        context: Application context with bot_data
    """
    if not update.message or not update.message.photo:
        return
    token, hybrid_url = context.bot_data["token"], context.bot_data["hybrid_url"]
    photo = update.message.photo[-1]
    file = await photo.get_file()
    image_bytes = bytes(await file.download_as_bytearray())

    try:
        # Notify user that processing started
        await update.message.reply_text(
            "🔄 Processing started...\n"
            "⏱️ Will take 10-20 seconds (Hough Transform)"
        )
        
        # Send to Hybrid Service
        payload = await send_to_hybrid(image_bytes, hybrid_url)
        
        # Visualize result
        viz_image = visualize_result(image_bytes, payload)
        
        # Send statistics
        stats = (
            f"📊 Recognized:\n"
            f"🔴 Walls: {len(payload.get('walls', []))}\n\n"
            f"🤖 Hough Transform"
        )
        
        # Send visualization + JSON
        await update.message.reply_photo(
            photo=io.BytesIO(viz_image),
            caption=stats,
        )
        
        json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        await update.message.reply_document(
            document=io.BytesIO(json_bytes),
            filename="result.json",
            caption="📄 JSON with coordinates",
        )
    except Exception as exc:  # noqa: BLE001
        await update.message.reply_text(f"❌ Error: {exc}")


def build_app() -> Application:
    """Create and configure Telegram bot.
    
    Loads settings from settings_secret.json,
    creates Application with command and message handlers.
    
    Returns:
        Application: Configured Telegram bot
    """
    token, hybrid_url = load_settings()
    app = ApplicationBuilder().token(token).build()
    app.bot_data["token"] = token
    app.bot_data["hybrid_url"] = hybrid_url
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    return app


def main() -> None:
    """Application entry point.
    
    Creates and runs Telegram bot in polling mode.
    Handles Conflict exception (if bot is already running).
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

