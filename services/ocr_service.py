"""OCR Service with EasyOCR.

Сервис оптического распознавания текста на планах помещений.
Использует современный EasyOCR (deep learning) для точного распознавания
размеров, площадей и технических обозначений на архитектурных планах.

Основные возможности:
    - Распознавание цифр и размерных обозначений
    - Автоматическая коррекция типичных ошибок OCR (O→0, l→1, S→5)
    - Продвинутый preprocessing (CLAHE, bilateral filter, Otsu)
    - Фильтрация нерелевантных результатов

Автор: Стреколовский Максим Владимирович
Заказчик: ООО Refloor
Дата: 10.12.2025
Версия: 2.0.0
"""
import os
from typing import Dict, List
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

# EasyOCR - современный OCR на базе нейросетей
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("❌ EasyOCR not available. Install: pip install easyocr")

app = FastAPI(title="Floorplan OCR Service (EasyOCR)", version="2.0.0")

# Global EasyOCR reader
reader = None


def init_easyocr():
    """Инициализация EasyOCR reader.
    
    Загружает предобученную модель EasyOCR для английского языка и цифр.
    Веса модели (~50MB) скачиваются автоматически при первом запуске
    и кэшируются локально для последующих использований.
    
    Returns:
        easyocr.Reader: Инициализированный reader или None при ошибке
        
    Note:
        GPU отключен (gpu=False) для максимальной совместимости
    """
    global reader
    
    if not EASYOCR_AVAILABLE:
        return None
    
    try:
        # Только английский и цифры для скорости
        # Веса скачаются автоматически при первом запуске (~50MB)
        reader = easyocr.Reader(['en'], gpu=False)  # GPU=False для совместимости
        print("✅ EasyOCR initialized (deep learning OCR)")
        return reader
    except Exception as e:
        print(f"❌ EasyOCR init failed: {e}")
        return None


def run_ocr(image_bytes: bytes) -> List[Dict]:
    """Распознавание текста на изображении плана с постобработкой.
    
    Применяет продвинутый preprocessing для улучшения качества OCR
    и постобработку для коррекции типичных ошибок распознавания.
    
    Pipeline:
    1. Декодирование изображения из bytes
    2. Preprocessing:
        - CLAHE (адаптивная эквализация гистограммы)
        - Bilateral filter (сглаживание с сохранением границ)
        - Otsu thresholding (бинаризация)
    3. EasyOCR распознавание с фильтром символов
    4. Постобработка:
        - Фильтрация по длине и наличию цифр
        - Коррекция типичных ошибок (O→0, l→1, S→5)
        - Извлечение bbox координат
    
    Args:
        image_bytes: Байты изображения (JPG/PNG)
        
    Returns:
        List[Dict]: Список распознанных элементов:
            - "text": str, распознанный текст
            - "bbox": [x_min, y_min, x_max, y_max], координаты bbox
            - "confidence": float, уверенность модели (0-100)
            
    Raises:
        ValueError: Если EasyOCR не инициализирован или изображение невалидно
    """
    if reader is None:
        raise ValueError("EasyOCR not initialized")
    
    data = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Невозможно прочитать изображение")
    
    # Улучшенный preprocessing для технических чертежей
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter для сохранения краёв
    bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Otsu thresholding
    _, binary = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # EasyOCR работает с RGB
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    # EasyOCR распознавание
    results = reader.readtext(
        rgb,
        allowlist='0123456789.,xмMм',  # Только цифры и размерные символы
        paragraph=False,
        min_size=8,  # Уменьшен для мелкого текста
        text_threshold=0.5,  # Снижен порог для захвата большего
    )
    
    out: List[Dict] = []
    for bbox, text, conf in results:
        text = text.strip()
        
        # Post-processing: фильтруем мусор и исправляем ошибки
        if not text or len(text) > 10:  # Размеры обычно короткие
            continue
        
        # Паттерн для размеров: X.XX или X,XX или просто цифры
        import re
        if not re.search(r'\d', text):  # Должна быть хотя бы одна цифра
            continue
        
        # Исправление типичных ошибок OCR
        text = text.replace('O', '0').replace('o', '0')  # O → 0
        text = text.replace('l', '1').replace('I', '1')  # l,I → 1
        text = text.replace('S', '5').replace('s', '5')  # S → 5
        
        # bbox = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        out.append({
            "text": text,
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": float(conf) * 100
        })
    
    return out


@app.on_event("startup")
async def startup():
    init_easyocr()
    print("🚀 OCR Service ready (EasyOCR)")


@app.post("/ocr", summary="OCR размеров и надписей на плане (EasyOCR)")
async def ocr_endpoint(file: UploadFile = File(...)):
    if reader is None:
        raise HTTPException(status_code=503, detail="EasyOCR not initialized")
    
    try:
        content = await file.read()
        detections = run_ocr(content)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"items": detections, "model": "EasyOCR"}


@app.get("/health")
async def health():
    return {"status": "ok", "easyocr": reader is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8002")))
