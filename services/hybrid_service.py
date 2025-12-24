"""Floor Plan Recognition Service.

Hybrid service for floor plan recognition.
Uses SAM 2.1 Large for room segmentation and Hough Transform for wall detection.

Main components:
    - Image preprocessing (CLAHE, bilateral filter, Otsu+Adaptive thresholding)
    - Wall detection via Probabilistic Hough Transform
    - Room detection via OCR areas + Watershed
    - Fallback detection via SAM 2.1 Large

Author: Maksim Strekolovsky
Date: 10.12.2025
Version: 1.0
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import ORJSONResponse
import orjson

# SAM 2 imports via Ultralytics (automatic weight loading)
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
    """Initialize SAM 2.1 Large model.
    
    Loads pretrained SAM 2.1 Large model for automatic segmentation.
    Model weights (224MB) are downloaded automatically on first run
    and cached locally for subsequent uses.
    
    Returns:
        SAM: Initialized SAM 2.1 Large model or None on error
        
    Raises:
        Exception: If model unavailable or initialization error
    """
    global sam_model
    
    if not SAM2_AVAILABLE:
        print("❌ SAM2 not available")
        return None
    
    try:
        # SAM 2.1 Large (224MB) - MAXIMUM accuracy (latest 2024)
        # Weights will be downloaded automatically on first run, then cached
        sam_model = SAM('sam2.1_l.pt')
        print(f"✅ SAM 2.1 Large initialized (maximum accuracy)")
        return sam_model
    except Exception as e:
        print(f"❌ SAM2 init failed: {e}")
        return None


def preprocess(image: np.ndarray) -> np.ndarray:
    """Advanced floor plan image preprocessing.
    
    Applies multi-stage processing to improve recognition quality:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
    2. Bilateral filter for edge-preserving smoothing
    3. Binarization via combination of Otsu and Adaptive thresholding
    4. Morphological operations for noise removal
    
    Args:
        image: Input image in BGR or grayscale format
        
    Returns:
        np.ndarray: Binary image (255 = object, 0 = background)
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
    """Wall detection on floor plan.
    
    Uses aggressive Probabilistic Hough Transform to detect
    all lines on the plan (exterior and interior walls).
    
    Algorithm:
    1. Binary image skeletonization (scikit-image thin)
    2. Probabilistic Hough Transform with low thresholds
    3. Snap to axis (alignment to 0°/45°/90°/135°)
    4. Duplicate removal
    5. Collinear segment merging
    
    Args:
        binary: Binary floor plan image (255 = object, 0 = background)
        
    Returns:
        List[Dict]: List of walls, each wall is a dict with keys:
            - "id": str, unique identifier (e.g., "w1")
            - "points": List[[x, y]], wall endpoint coordinates in pixels
    """
    # Work with inverted image (walls = black lines)
    from skimage.morphology import thin
    skeleton = (thin(binary == 0) * 255).astype(np.uint8)
    
    # AGGRESSIVE Hough to capture ALL lines (including interior walls)
    lines = cv2.HoughLinesP(
        skeleton, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30,      # LOWERED for interior walls
        minLineLength=20,  # LOWERED for short walls
        maxLineGap=35      # INCREASED to connect breaks
    )
    
    segments = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            length = np.hypot(x2 - x1, y2 - y1)
            if length >= 15:  # Minimum 15px (capture short interior walls)
                x1, y1, x2, y2 = snap_to_axis(x1, y1, x2, y2)
                segments.append({"points": [[int(x1), int(y1)], [int(x2), int(y2)]]})
    
    # Remove duplicates
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
    
    # Merge collinear (aggressively to merge fragmented walls)
    merged = merge_segments(unique_segments, max_angle=3.0, max_gap=50.0)
    
    print(f"🔴 Detected {len(merged)} walls (after merging from {len(unique_segments)} segments)")
    return [{"id": f"w{i+1}", **seg} for i, seg in enumerate(merged)]


def detect_rooms_sam2(image: np.ndarray, binary: np.ndarray) -> List[Dict]:
    """Room detection via SAM 2.1 Large with multi-layer filtering.
    
    Uses SAM2 automatic segmentation to find large areas (rooms)
    with strict filtering to exclude furniture, text, and other objects.
    
    Filters:
    1. Mask area: 3000 < area < 60% of image
    2. Minimum contour area: > 4000 pixels
    3. Aspect ratio: < 5 (not elongated objects)
    4. Solidity: > 0.75 (compact shapes)
    5. Number of corners: 3-12 (polygons)
    
    Args:
        image: Original color image in BGR format
        binary: Binary image (not used directly)
        
    Returns:
        List[Dict]: List of rooms with polygons, areas, and confidence scores
        
    Note:
        Used as fallback if OCR found < 5 rooms
    """
    if sam_model is None:
        return detect_rooms_fallback(binary)
    
    try:
        # SAM2 automatic segmentation with parameters for large objects
        results = sam_model(
            image, 
            retina_masks=True,
            imgsz=1024,
            conf=0.4,  # Lowered threshold to capture all rooms
            iou=0.9,   # High IOU to avoid merging adjacent rooms
        )
        
        all_masks = []
        for result in results:
            if result.masks is None:
                continue
            
            for idx, mask in enumerate(result.masks.data):
                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                mask_uint8 = (mask_np * 255).astype(np.uint8)
                
                # Check mask size immediately
                area = cv2.countNonZero(mask_uint8)
                img_area = mask_uint8.shape[0] * mask_uint8.shape[1]
                
                # FILTER: medium and large areas (rooms), not furniture and not entire apartment
                if area < 3000 or area > img_area * 0.6:
                    continue
                
                conf = result.masks.conf[idx].item() if hasattr(result.masks, 'conf') else 0.9
                all_masks.append({
                    "mask": mask_uint8,
                    "area": area,
                    "confidence": float(conf)
                })
        
        print(f"📊 SAM2 found {len(all_masks)} large areas (potential rooms)")
        
        # Convert masks to polygons with SMART filtering
        rooms = []
        for mask_data in all_masks:
            mask = mask_data["mask"]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter 1: Minimum area (NOT furniture)
                if area < 4000:
                    continue
                
                # Filter 2: Aspect ratio (rooms NOT too elongated)
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > 5:  # Too elongated object (pipe, wall)
                        continue
                
                # Filter 3: Solidity (rooms are solid shapes)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity < 0.75:  # Too complex shape (furniture)
                        continue
                
                # Polygon approximation
                epsilon = 0.012 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Filter 4: Number of corners (rooms have 3-12 corners)
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
    """Intelligent room detection via OCR areas.
    
    Uses EasyOCR to recognize room areas (e.g., "7.9", "12.6")
    on technical drawings and builds room polygons via Watershed from found centers.
    
    Algorithm:
    1. OCR text recognition on plan (EasyOCR)
    2. Filtering: search for numbers in range 0.8-20 m² (realistic room areas)
    3. Exclude technical notations (k=, h=, dimensions)
    4. Use coordinates of found areas as room centers
    5. Watershed segmentation to build polygons from centers
    6. Filter results by area and aspect ratio
    
    Args:
        image: Original color floor plan image in BGR format
        binary: Binary floor plan image
        
    Returns:
        List[Dict]: List of rooms, each room is a dict with keys:
            - "id": str, unique identifier (e.g., "r1")
            - "polygon": List[[x, y]], room polygon vertices
            - "area": int, room area in pixels²
            - "label": str, recognized area text (e.g., "7.9")
            - "area_sqm": float, area in square meters
    """
    try:
        import easyocr
        reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
        
        # OCR on image
        results = reader.readtext(image)
        
        # Search for numbers similar to room areas (STRICT filter)
        room_centers = []
        for (bbox, text, conf) in results:
            # EXCLUDE technical notations (k=, h=, w=, etc.)
            if any(x in text.lower() for x in ['k=', 'h=', 'w=', 'х', '×']):
                continue
            
            # Try to extract number
            import re
            numbers = re.findall(r'\d+\.?\d*', text)
            
            # Must be ONE number (pure area, not "3.96")
            if len(numbers) != 1:
                continue
            
            try:
                area_sqm = float(numbers[0])
                # STRICT filter: realistic ROOM areas (0.8-20 m²)
                # Include small spaces (bathroom, corridor, storage)
                if 0.8 <= area_sqm <= 20:
                    # bbox center = room center
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
        
        # Build room polygons using watershed from centers
        # Invert (rooms = white areas)
        binary_inv = cv2.bitwise_not(binary)
        
        # AGGRESSIVE closing of wall breaks (so rooms are closed)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        # Distance transform
        dist = cv2.distanceTransform(binary_inv, cv2.DIST_L2, 5)
        
        # Create markers for watershed
        markers = np.zeros(binary.shape, dtype=np.int32)
        for idx, room_center in enumerate(room_centers):
            cx, cy = room_center["center"]
            markers[cy, cx] = idx + 1
        
        # Watershed
        bgr_img = cv2.cvtColor(binary_inv, cv2.COLOR_GRAY2BGR)
        cv2.watershed(bgr_img, markers)
        
        # Extract room contours with FILTERING
        rooms = []
        img_area = binary.shape[0] * binary.shape[1]
        
        for idx, room_center in enumerate(room_centers):
            mask = (markers == idx + 1).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # FILTER: room should NOT be too small or HUGE
                if area < 2000:
                    continue
                
                # CRITICAL FILTER: room should NOT occupy > 40% of image
                # (SAM2 sometimes outlines entire apartment)
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
    """Fallback room detection via morphological operations.
    
    Used when SAM2 is unavailable or OCR didn't find rooms.
    Applies morphological operations to close wall breaks
    and find closed contours (rooms).
    
    Algorithm:
    1. Morphological closing of breaks (MORPH_CLOSE)
    2. Image inversion (rooms = white areas)
    3. Noise removal (MORPH_OPEN)
    4. Contour search with hierarchy (RETR_TREE)
    5. Filtering by area, hierarchy, and number of corners
    
    Args:
        binary: Binary floor plan image
        
    Returns:
        List[Dict]: List of rooms with polygons and areas
    """
    # Close breaks (moderate)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=5)
    
    # Invert (rooms = white areas)
    binary_inv = cv2.bitwise_not(closed)
    
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(binary_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"📊 Found {len(contours)} total contours")
    
    rooms = []
    img_area = binary.shape[0] * binary.shape[1]
    
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Area filter (more lenient)
        if not (1000 < area < img_area * 0.9):
            continue
        
        # Check hierarchy
        parent = hierarchy[0][idx][3]
        if parent == -1 and area < 5000:  # Only very small ones without parent are skipped
            continue
        
        # Polygon approximation
        epsilon = 0.015 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= 3:  # Minimum 3 corners
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
    """Align line to main angles (0°/45°/90°/135°).
    
    Corrects line endpoint coordinates so it is strictly
    horizontal, vertical, or diagonal (45°/135°).
    Improves wall detection quality on plans.
    
    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates
        
    Returns:
        Tuple[int, int, int, int]: Corrected coordinates (x1, y1, x2, y2)
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
    """Merge collinear wall segments.
    
    Combines fragmented wall lines into single segments
    based on angle between them and distance between endpoints.
    
    Args:
        segments: List of wall segments
        max_angle: Maximum angle between segments for merging (degrees)
        max_gap: Maximum distance between segment endpoints (pixels)
        
    Returns:
        List[Dict]: Merged wall segments
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
    """Calculate angle between two segments.
    
    Computes minimum angle between vectors of two segments
    via dot product.
    
    Args:
        p1, p2: First segment endpoints [x, y]
        p3, p4: Second segment endpoints [x, y]
        
    Returns:
        float: Angle between segments in degrees (0-90°)
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
    """Main floor plan recognition pipeline.
    
    Performs full cycle of floor plan image processing:
    1. Image decoding from bytes
    2. Preprocessing (CLAHE, filtering, binarization)
    3. Wall detection (Hough Transform)
    4. Room detection (OCR areas + Watershed, fallback SAM2)
    5. Combining OCR and SAM2 results
    
    Args:
        image_bytes: Image bytes (JPG/PNG)
        source_name: Source filename (optional)
        
    Returns:
        Dict: JSON with recognition results:
            - "meta": Dict with metadata (source, width, height, model)
            - "walls": List[Dict] list of walls with coordinates
            - "rooms": List[Dict] list of rooms with polygons and areas
            - "error": str (only on error)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if bgr is None:
        return {"error": "Failed to decode image"}
    
    binary = preprocess(bgr)
    
    # MAIN TASK: Wall detection
    walls = detect_walls(binary)
    
    # BONUS: Room contours (SMART detection via OCR areas)
    rooms = detect_rooms_by_labels(bgr, binary)
    
    # COMBINED approach: If OCR found few rooms, supplement with SAM2
    if len(rooms) < 5:  # Expect minimum 5-6 rooms in apartment
        print(f"⚠️ OCR found only {len(rooms)} rooms, adding SAM2 rooms...")
        sam2_rooms = detect_rooms_sam2(bgr, binary)
        
        # Add SAM2 rooms that do NOT intersect with OCR rooms
        for sam_room in sam2_rooms:
            is_duplicate = False
            sam_poly = np.array(sam_room["polygon"], dtype=np.int32)
            
            for ocr_room in rooms:
                ocr_poly = np.array(ocr_room["polygon"], dtype=np.int32)
                # Check polygon intersection
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
