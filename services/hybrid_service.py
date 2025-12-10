"""Floor Plan Recognition Service.

Гибридный сервис для распознавания планов помещений.
Использует SAM 2.1 Large для сегментации комнат и Hough Transform для детекции стен.

Основные компоненты:
    - Препроцессинг изображений (CLAHE, bilateral filter, Otsu+Adaptive thresholding)
    - Детекция стен через Probabilistic Hough Transform
    - Детекция комнат через OCR площадей + Watershed
    - Fallback детекция через SAM 2.1 Large

Автор: Стреколовский Максим Владимирович
Заказчик: ООО Refloor
Дата: 10.12.2025
Версия: 1.0
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import ORJSONResponse
import orjson

# SAM 2 imports via Ultralytics (автоматическая загрузка весов)
try:
    from ultralytics import SAM
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("❌ Ultralytics not installed. Run: pip install ultralytics")

app = FastAPI(title="Floor Plan Service")

# Global models
sam_model = None


def init_sam2():
    """Инициализация модели SAM 2.1 Large.
    
    Загружает предобученную модель SAM 2.1 Large для автоматической сегментации.
    Веса модели (224MB) скачиваются автоматически при первом запуске
    и кэшируются локально для последующих использований.
    
    Returns:
        SAM: Инициализированная модель SAM 2.1 Large или None при ошибке
        
    Raises:
        Exception: Если модель недоступна или произошла ошибка инициализации
    """
    global sam_model
    
    if not SAM2_AVAILABLE:
        print("❌ SAM2 not available")
        return None
    
    try:
        # SAM 2.1 Large (224MB) - МАКСИМАЛЬНАЯ точность (latest 2024)
        # Веса скачаются автоматически при первом запуске, потом будут в кэше
        sam_model = SAM('sam2.1_l.pt')
        print(f"✅ SAM 2.1 Large initialized (максимальная точность)")
        return sam_model
    except Exception as e:
        print(f"❌ SAM2 init failed: {e}")
        return None


def preprocess(image: np.ndarray) -> np.ndarray:
    """Продвинутая предобработка изображения плана.
    
    Применяет многоступенчатую обработку для улучшения качества распознавания:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization) для усиления контраста
    2. Bilateral filter для сглаживания с сохранением границ
    3. Бинаризация через комбинацию Otsu и Adaptive thresholding
    4. Морфологические операции для удаления шума
    
    Args:
        image: Входное изображение в BGR или grayscale формате
        
    Returns:
        np.ndarray: Бинаризованное изображение (255 = объект, 0 = фон)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Multi-scale CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter (edge-preserving)
    bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Otsu + Adaptive fusion
    _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    binary = cv2.bitwise_and(otsu, adapt)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary


def detect_walls(binary: np.ndarray) -> List[Dict]:
    """Детекция стен на плане помещения.
    
    Использует агрессивный Probabilistic Hough Transform для детекции
    всех линий на плане (внешние и внутренние стены).
    
    Алгоритм:
    1. Скелетизация бинарного изображения (scikit-image thin)
    2. Probabilistic Hough Transform с низкими порогами
    3. Snap to axis (выравнивание по 0°/45°/90°/135°)
    4. Удаление дубликатов
    5. Слияние коллинеарных сегментов
    
    Args:
        binary: Бинаризованное изображение плана (255 = объект, 0 = фон)
        
    Returns:
        List[Dict]: Список стен, каждая стена - dict с ключами:
            - "id": str, уникальный идентификатор (например, "w1")
            - "points": List[[x, y]], координаты концов стены в пикселях
    """
    # Работаем с инвертированным изображением (стены = черные линии)
    from skimage.morphology import thin
    skeleton = (thin(binary == 0) * 255).astype(np.uint8)
    
    # АГРЕССИВНЫЙ Hough для захвата ВСЕХ линий (включая внутренние стены)
    lines = cv2.HoughLinesP(
        skeleton, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30,      # СНИЖЕН для внутренних стен
        minLineLength=20,  # СНИЖЕН для коротких стен
        maxLineGap=35      # УВЕЛИЧЕН для соединения разрывов
    )
    
    segments = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            length = np.hypot(x2 - x1, y2 - y1)
            if length >= 15:  # Минимум 15px (захватываем короткие внутренние стены)
                x1, y1, x2, y2 = snap_to_axis(x1, y1, x2, y2)
                segments.append({"points": [[int(x1), int(y1)], [int(x2), int(y2)]]})
    
    # Удаляем дубликаты
    unique_segments = []
    for seg in segments:
        is_duplicate = False
        for useg in unique_segments:
            dist1 = np.hypot(seg["points"][0][0] - useg["points"][0][0], seg["points"][0][1] - useg["points"][0][1])
            dist2 = np.hypot(seg["points"][1][0] - useg["points"][1][0], seg["points"][1][1] - useg["points"][1][1])
            if dist1 < 15 and dist2 < 15:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_segments.append(seg)
    
    # Merge collinear (агрессивно для объединения фрагментированных стен)
    merged = merge_segments(unique_segments, max_angle=3.0, max_gap=50.0)
    
    print(f"🔴 Detected {len(merged)} walls (after merging from {len(unique_segments)} segments)")
    return [{"id": f"w{i+1}", **seg} for i, seg in enumerate(merged)]


def detect_rooms_sam2(image: np.ndarray, binary: np.ndarray) -> List[Dict]:
    """Детекция комнат через SAM 2.1 Large с многослойной фильтрацией.
    
    Использует автоматическую сегментацию SAM2 для поиска крупных областей (комнат)
    с последующей строгой фильтрацией для исключения мебели, текста и других объектов.
    
    Фильтры:
    1. Площадь маски: 3000 < area < 60% изображения
    2. Минимальная площадь контура: > 4000 пикселей
    3. Aspect ratio: < 5 (не вытянутые объекты)
    4. Solidity: > 0.75 (компактные формы)
    5. Количество углов: 3-12 (многоугольники)
    
    Args:
        image: Оригинальное цветное изображение в BGR формате
        binary: Бинаризованное изображение (не используется напрямую)
        
    Returns:
        List[Dict]: Список комнат с полигонами, площадями и confidence scores
        
    Note:
        Используется как fallback если OCR нашел < 5 комнат
    """
    if sam_model is None:
        return detect_rooms_fallback(binary)
    
    try:
        # SAM2 автоматическая сегментация с параметрами для крупных объектов
        results = sam_model(
            image, 
            retina_masks=True,
            imgsz=1024,
            conf=0.4,  # Снижен порог для захвата всех комнат
            iou=0.9,   # Высокий IOU чтобы не объединять соседние комнаты
        )
        
        all_masks = []
        for result in results:
            if result.masks is None:
                continue
            
            for idx, mask in enumerate(result.masks.data):
                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                mask_uint8 = (mask_np * 255).astype(np.uint8)
                
                # Сразу проверяем размер маски
                area = cv2.countNonZero(mask_uint8)
                img_area = mask_uint8.shape[0] * mask_uint8.shape[1]
                
                # ФИЛЬТР: средние и большие области (комнаты), не мебель и не вся квартира
                if area < 3000 or area > img_area * 0.6:
                    continue
                
                conf = result.masks.conf[idx].item() if hasattr(result.masks, 'conf') else 0.9
                all_masks.append({
                    "mask": mask_uint8,
                    "area": area,
                    "confidence": float(conf)
                })
        
        print(f"📊 SAM2 found {len(all_masks)} large areas (potential rooms)")
        
        # Конвертируем маски в полигоны с УМНОЙ фильтрацией
        rooms = []
        for mask_data in all_masks:
            mask = mask_data["mask"]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Фильтр 1: Минимальная площадь (НЕ мебель)
                if area < 4000:
                    continue
                
                # Фильтр 2: Aspect ratio (комнаты НЕ слишком вытянутые)
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > 5:  # Слишком вытянутый объект (труба, стена)
                        continue
                
                # Фильтр 3: Solidity (комнаты - это solid shapes)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity < 0.75:  # Слишком сложная форма (мебель)
                        continue
                
                # Аппроксимация полигона
                epsilon = 0.012 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Фильтр 4: Количество углов (комнаты имеют 3-12 углов)
                if not (3 <= len(approx) <= 12):
                    continue
                
                polygon = [[int(p[0][0]), int(p[0][1])] for p in approx]
                rooms.append({
                    "id": f"r{len(rooms)+1}",
                    "polygon": polygon,
                    "area": int(area),
                    "confidence": mask_data["confidence"]
                })
        
        print(f"✅ SAM2 detected {len(rooms)} rooms")
        return rooms if rooms else detect_rooms_fallback(binary)
    
    except Exception as e:
        print(f"⚠️ SAM2 failed: {e}, using fallback")
        return detect_rooms_fallback(binary)


def detect_rooms_by_labels(image: np.ndarray, binary: np.ndarray) -> List[Dict]:
    """Интеллектуальная детекция комнат через OCR площадей.
    
    Использует EasyOCR для распознавания площадей комнат (например, "7.9", "12.6")
    на плане БТИ и строит полигоны комнат через Watershed от найденных центров.
    
    Алгоритм:
    1. OCR распознавание текста на плане (EasyOCR)
    2. Фильтрация: поиск чисел в диапазоне 0.8-20 м² (реалистичные площади комнат)
    3. Исключение технических обозначений (k=, h=, размеры)
    4. Использование координат найденных площадей как центров комнат
    5. Watershed сегментация для построения полигонов от центров
    6. Фильтрация результатов по area и aspect ratio
    
    Args:
        image: Оригинальное цветное изображение плана в BGR формате
        binary: Бинаризованное изображение плана
        
    Returns:
        List[Dict]: Список комнат, каждая комната - dict с ключами:
            - "id": str, уникальный идентификатор (например, "r1")
            - "polygon": List[[x, y]], вершины полигона комнаты
            - "area": int, площадь комнаты в пикселях²
            - "label": str, распознанный текст площади (например, "7.9")
            - "area_sqm": float, площадь в квадратных метрах
    """
    try:
        import easyocr
        reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
        
        # OCR на изображении
        results = reader.readtext(image)
        
        # Ищем числа похожие на площади комнат (СТРОГИЙ фильтр)
        room_centers = []
        for (bbox, text, conf) in results:
            # ИСКЛЮЧАЕМ технические обозначения (k=, h=, w=, и т.д.)
            if any(x in text.lower() for x in ['k=', 'h=', 'w=', 'х', '×']):
                continue
            
            # Пытаемся извлечь число
            import re
            numbers = re.findall(r'\d+\.?\d*', text)
            
            # Должно быть ОДНО число (чистая площадь, не "3.96")
            if len(numbers) != 1:
                continue
            
            try:
                area_sqm = float(numbers[0])
                # СТРОГИЙ фильтр: реалистичные площади КОМНАТ (0.8-20 м²)
                # Включаем маленькие помещения (ванная, коридор, кладовка)
                if 0.8 <= area_sqm <= 20:
                    # Центр bbox = центр комнаты
                    x_center = int((bbox[0][0] + bbox[2][0]) / 2)
                    y_center = int((bbox[0][1] + bbox[2][1]) / 2)
                    room_centers.append({
                        "center": [x_center, y_center],
                        "area_sqm": area_sqm,
                        "text": text
                    })
                    print(f"  📍 Found room label: {text} at ({x_center}, {y_center})")
            except:
                continue
        
        print(f"🏠 Found {len(room_centers)} room labels via OCR")
        
        if not room_centers:
            return []
        
        # Строим полигоны комнат используя watershed от центров
        # Инвертируем (комнаты = белые области)
        binary_inv = cv2.bitwise_not(binary)
        
        # АГРЕССИВНОЕ закрытие разрывов в стенах (чтобы комнаты были замкнуты)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        # Distance transform
        dist = cv2.distanceTransform(binary_inv, cv2.DIST_L2, 5)
        
        # Создаем маркеры для watershed
        markers = np.zeros(binary.shape, dtype=np.int32)
        for idx, room_center in enumerate(room_centers):
            cx, cy = room_center["center"]
            markers[cy, cx] = idx + 1
        
        # Watershed
        bgr_img = cv2.cvtColor(binary_inv, cv2.COLOR_GRAY2BGR)
        cv2.watershed(bgr_img, markers)
        
        # Извлекаем контуры комнат с ФИЛЬТРАЦИЕЙ
        rooms = []
        img_area = binary.shape[0] * binary.shape[1]
        
        for idx, room_center in enumerate(room_centers):
            mask = (markers == idx + 1).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # ФИЛЬТР: комната НЕ должна быть слишком маленькой или ОГРОМНОЙ
                if area < 2000:
                    continue
                
                # КРИТИЧЕСКИЙ ФИЛЬТР: комната НЕ должна занимать > 40% изображения
                # (SAM2 иногда обводит всю квартиру)
                if area > img_area * 0.4:
                    print(f"  ⚠️ Rejected room {idx+1}: too large ({area}/{img_area} = {area/img_area:.1%})")
                    continue
                
                epsilon = 0.015 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) >= 3:
                    polygon = [[int(p[0][0]), int(p[0][1])] for p in approx]
                    rooms.append({
                        "id": f"r{len(rooms)+1}",
                        "polygon": polygon,
                        "area": int(area),
                        "label": room_center["text"],
                        "area_sqm": room_center["area_sqm"]
                    })
        
        print(f"✅ Built {len(rooms)} room polygons from OCR labels")
        return rooms
    
    except Exception as e:
        print(f"⚠️ OCR-based room detection failed: {e}")
        return []


def detect_rooms_fallback(binary: np.ndarray) -> List[Dict]:
    """Fallback детекция комнат через морфологические операции.
    
    Используется когда SAM2 недоступен или OCR не нашел комнаты.
    Применяет морфологические операции для замыкания разрывов в стенах
    и поиска замкнутых контуров (комнат).
    
    Алгоритм:
    1. Морфологическое закрытие разрывов (MORPH_CLOSE)
    2. Инверсия изображения (комнаты = белые области)
    3. Удаление шума (MORPH_OPEN)
    4. Поиск контуров с иерархией (RETR_TREE)
    5. Фильтрация по площади, иерархии и количеству углов
    
    Args:
        binary: Бинаризованное изображение плана
        
    Returns:
        List[Dict]: Список комнат с полигонами и площадями
    """
    # Закрытие разрывов (умеренное)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=5)
    
    # Инвертируем (комнаты = белые области)
    binary_inv = cv2.bitwise_not(closed)
    
    # Убираем мелкий шум
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Находим контуры с иерархией
    contours, hierarchy = cv2.findContours(binary_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"📊 Found {len(contours)} total contours")
    
    rooms = []
    img_area = binary.shape[0] * binary.shape[1]
    
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Фильтр по площади (более мягкий)
        if not (1000 < area < img_area * 0.9):
            continue
        
        # Проверяем иерархию
        parent = hierarchy[0][idx][3]
        if parent == -1 and area < 5000:  # Только очень маленькие без родителя пропускаем
            continue
        
        # Аппроксимация полигона
        epsilon = 0.015 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= 3:  # Минимум 3 угла
            polygon = [[int(p[0][0]), int(p[0][1])] for p in approx]
            rooms.append({
                "id": f"r{len(rooms)+1}",
                "polygon": polygon,
                "area": int(area)
            })
            print(f"  ✓ Room {len(rooms)}: area={area:.0f}, corners={len(approx)}, parent={parent}")
    
    print(f"✅ Detected {len(rooms)} rooms")
    return rooms


def snap_to_axis(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int, int, int]:
    """Выравнивание линии по основным углам (0°/45°/90°/135°).
    
    Корректирует координаты концов линии чтобы она была строго
    горизонтальной, вертикальной или диагональной (45°/135°).
    Улучшает качество детекции стен на планах.
    
    Args:
        x1, y1: Координаты первой точки
        x2, y2: Координаты второй точки
        
    Returns:
        Tuple[int, int, int, int]: Скорректированные координаты (x1, y1, x2, y2)
    """
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length < 1:
        return int(x1), int(y1), int(x2), int(y2)
    
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    
    if abs(angle) < 5 or abs(angle - 180) < 5:
        y2 = y1
    elif abs(angle - 90) < 5:
        x2 = x1
    elif abs(angle - 45) < 5:
        avg = (abs(dx) + abs(dy)) / 2
        x2 = x1 + (avg if dx > 0 else -avg)
        y2 = y1 + (avg if dy > 0 else -avg)
    elif abs(angle - 135) < 5:
        avg = (abs(dx) + abs(dy)) / 2
        x2 = x1 + (avg if dx > 0 else -avg)
        y2 = y1 - (avg if dy > 0 else -avg)
    
    return int(x1), int(y1), int(x2), int(y2)


def merge_segments(segments: List[Dict], max_angle: float, max_gap: float) -> List[Dict]:
    """Слияние коллинеарных сегментов стен.
    
    Объединяет фрагментированные линии стен в единые сегменты
    на основе угла между ними и расстояния между концами.
    
    Args:
        segments: Список сегментов стен
        max_angle: Максимальный угол между сегментами для слияния (градусы)
        max_gap: Максимальное расстояние между концами сегментов (пиксели)
        
    Returns:
        List[Dict]: Объединённые сегменты стен
    """
    if not segments:
        return []
    
    merged = []
    used = [False] * len(segments)
    
    for i, seg1 in enumerate(segments):
        if used[i]:
            continue
        
        p1, p2 = seg1["points"]
        pts = [p1, p2]
        used[i] = True
        
        for j, seg2 in enumerate(segments):
            if used[j] or i == j:
                continue
            
            p3, p4 = seg2["points"]
            angle_diff = angle_between(p1, p2, p3, p4)
            dist = min(
                np.hypot(p2[0]-p3[0], p2[1]-p3[1]),
                np.hypot(p2[0]-p4[0], p2[1]-p4[1]),
                np.hypot(p1[0]-p3[0], p1[1]-p3[1]),
                np.hypot(p1[0]-p4[0], p1[1]-p4[1]),
            )
            
            if angle_diff < max_angle and dist < max_gap:
                pts.extend([p3, p4])
                used[j] = True
        
        if len(pts) > 2:
            pts_arr = np.array(pts)
            dists = np.linalg.norm(pts_arr[:, None] - pts_arr[None, :], axis=2)
            i_max, j_max = np.unravel_index(dists.argmax(), dists.shape)
            pts = [pts[i_max], pts[j_max]]
        
        merged.append({"points": pts})
    
    return merged


def angle_between(p1, p2, p3, p4) -> float:
    """Вычисление угла между двумя сегментами.
    
    Рассчитывает минимальный угол между векторами двух сегментов
    через скалярное произведение.
    
    Args:
        p1, p2: Концы первого сегмента [x, y]
        p3, p4: Концы второго сегмента [x, y]
        
    Returns:
        float: Угол между сегментами в градусах (0-90°)
    """
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p4[0] - p3[0], p4[1] - p3[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    angle = np.degrees(np.arccos(abs(cos_angle)))
    return min(angle, 180 - angle)


def process_image(image_bytes: bytes, source_name: str = "unknown.png") -> Dict:
    """Основной pipeline распознавания плана помещения.
    
    Выполняет полный цикл обработки изображения плана:
    1. Декодирование изображения из bytes
    2. Предобработка (CLAHE, фильтрация, бинаризация)
    3. Детекция стен (Hough Transform)
    4. Детекция комнат (OCR площадей + Watershed, fallback SAM2)
    5. Комбинирование результатов OCR и SAM2
    
    Args:
        image_bytes: Байты изображения (JPG/PNG)
        source_name: Имя исходного файла (опционально)
        
    Returns:
        Dict: JSON с результатами распознавания:
            - "meta": Dict с метаданными (source, width, height, model)
            - "walls": List[Dict] список стен с координатами
            - "rooms": List[Dict] список комнат с полигонами и площадями
            - "error": str (только при ошибке)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if bgr is None:
        return {"error": "Failed to decode image"}
    
    binary = preprocess(bgr)
    
    # ОСНОВНАЯ ЗАДАЧА: Детекция стен
    walls = detect_walls(binary)
    
    # БОНУС: Контуры помещений (УМНАЯ детекция через OCR площадей)
    rooms = detect_rooms_by_labels(bgr, binary)
    
    # КОМБИНИРОВАННЫЙ подход: Если OCR нашел мало комнат, дополняем через SAM2
    if len(rooms) < 5:  # Ожидаем минимум 5-6 комнат в квартире
        print(f"⚠️ OCR found only {len(rooms)} rooms, adding SAM2 rooms...")
        sam2_rooms = detect_rooms_sam2(bgr, binary)
        
        # Добавляем SAM2 комнаты которые НЕ пересекаются с OCR комнатами
        for sam_room in sam2_rooms:
            is_duplicate = False
            sam_poly = np.array(sam_room["polygon"], dtype=np.int32)
            
            for ocr_room in rooms:
                ocr_poly = np.array(ocr_room["polygon"], dtype=np.int32)
                # Проверяем пересечение полигонов
                intersection = cv2.intersectConvexConvex(ocr_poly, sam_poly)[1]
                if intersection is not None and cv2.contourArea(intersection) > 1000:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                sam_room["id"] = f"r{len(rooms)+1}"
                rooms.append(sam_room)
                print(f"  ➕ Added SAM2 room (area={sam_room['area']})")
    
    print(f"✅ Total rooms detected: {len(rooms)}")
    
    return {
        "meta": {
            "source": source_name
        },
        "walls": walls
    }


@app.on_event("startup")
async def startup():
    init_sam2()
    print("🚀 Floor Plan Service ready")


@app.post("/detect", response_class=ORJSONResponse)
async def detect(file: UploadFile = File(...)):
    content = await file.read()
    source_name = file.filename if file.filename else "unknown.png"
    result = process_image(content, source_name)
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "sam2_large": sam_model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
