"""Floorplan Cleanup Service.

Сервис предобработки изображений планов помещений.
Выполняет коррекцию перспективы, шумоподавление и бинаризацию
для улучшения качества последующего распознавания.

Основные возможности:
    - Автоматическая коррекция перспективы (perspective transform)
    - Детекция и выравнивание четырёхугольников (планы, сфотографированные под углом)
    - Шумоподавление (fastNlMeansDenoising)
    - Адаптивная бинаризация для удаления артефактов

Автор: Стреколовский Максим Владимирович
Заказчик: ООО Refloor
Дата: 10.12.2025
Версия: 0.1.0
"""
import io
import os
from typing import Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse


app = FastAPI(title="Floorplan Cleanup Service", version="0.1.0")


def largest_quadrilateral(mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """Поиск самого большого четырёхугольника на изображении.
    
    Находит все контуры на бинарной маске, аппроксимирует их
    и возвращает самый большой четырёхугольник (план, сфотографированный под углом).
    
    Args:
        mask: Бинарная маска (255 = объект, 0 = фон)
        
    Returns:
        Tuple[np.ndarray, float]: Кортеж из:
            - Массив 4 точек четырёхугольника или None
            - Площадь четырёхугольника
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area
                best = approx
    return best, best_area


def warp_perspective(image: np.ndarray) -> np.ndarray:
    """Коррекция перспективы плана (выравнивание четырёхугольника).
    
    Автоматически находит самый большой четырёхугольник на изображении
    (предполагается что это план) и применяет perspective transform
    для выравнивания в прямоугольник.
    
    Алгоритм:
    1. Конвертация в grayscale и бинаризация (Otsu)
    2. Поиск самого большого четырёхугольника
    3. Упорядочивание углов (top-left, top-right, bottom-right, bottom-left)
    4. Вычисление размеров выходного прямоугольника
    5. Применение perspective transform
    
    Args:
        image: Входное изображение в BGR формате
        
    Returns:
        np.ndarray: Выровненное изображение или оригинал если четырёхугольник не найден
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    quad, area = largest_quadrilateral(th)
    if quad is None or area < 0.1 * image.shape[0] * image.shape[1]:
        return image
    quad = quad.reshape(4, 2).astype(np.float32)
    s = quad.sum(axis=1)
    diff = np.diff(quad, axis=1)
    ordered = np.array(
        [
            quad[np.argmin(s)],
            quad[np.argmin(diff)],
            quad[np.argmax(s)],
            quad[np.argmax(diff)],
        ],
        dtype=np.float32,
    )
    (tl, tr, br, bl) = ordered
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_w = int(max(width_a, width_b))
    max_h = int(max(height_a, height_b))
    dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (max_w, max_h))
    return warped


def despeckle(image: np.ndarray) -> np.ndarray:
    """Удаление шума и артефактов с изображения.
    
    Применяет адаптивную бинаризацию и морфологические операции
    для удаления мелких точек, пятен и других артефактов.
    
    Pipeline:
    1. Gaussian blur для сглаживания
    2. Adaptive thresholding для бинаризации
    3. Morphological opening для удаления мелких объектов
    4. Morphological closing для заполнения разрывов
    
    Args:
        image: Входное изображение в BGR формате
        
    Returns:
        np.ndarray: Бинарное изображение без шума (255 = объект, 0 = фон)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_img = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def clean_image(file_bytes: bytes) -> bytes:
    """Полный pipeline очистки изображения плана.
    
    Выполняет все этапы предобработки:
    1. Декодирование изображения
    2. Коррекция перспективы
    3. Шумоподавление (Non-Local Means Denoising)
    4. Бинаризация и удаление артефактов
    5. Кодирование в PNG
    
    Args:
        file_bytes: Байты изображения (JPG/PNG)
        
    Returns:
        bytes: Обработанное бинарное изображение в PNG формате
        
    Raises:
        ValueError: Если изображение невалидно или ошибка кодирования
    """
    data = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Невозможно прочитать изображение")
    warped = warp_perspective(bgr)
    denoised = cv2.fastNlMeansDenoisingColored(warped, None, 10, 10, 7, 21)
    bin_mask = despeckle(denoised)
    # Возвращаем бинарное изображение, пригодное для последующего пайплайна
    ok, buf = cv2.imencode(".png", bin_mask)
    if not ok:
        raise ValueError("Ошибка кодирования PNG")
    return buf.tobytes()


@app.post("/clean", summary="Нормализация перспективы и шумоподавление")
async def clean_endpoint(file: UploadFile = File(...)):
    try:
        content = await file.read()
        cleaned = clean_image(content)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StreamingResponse(io.BytesIO(cleaned), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))

