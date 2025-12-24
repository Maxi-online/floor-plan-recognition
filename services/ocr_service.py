"""OCR Service with EasyOCR.

Optical character recognition service for floor plans.
Uses modern EasyOCR (deep learning) for accurate recognition
of dimensions, areas, and technical notations on architectural plans.

Main features:
    - Recognition of digits and dimension notations
    - Automatic correction of typical OCR errors (O→0, l→1, S→5)
    - Advanced preprocessing (CLAHE, bilateral filter, Otsu)
    - Filtering of irrelevant results

Author: Maksim Strekolovsky
Date: 10.12.2025
Version: 2.0.0
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
    """Initialize EasyOCR reader.
    
    Loads pretrained EasyOCR model for English and digits.
    Model weights (~50MB) are downloaded automatically on first run
    and cached locally for subsequent uses.
    
    Returns:
        easyocr.Reader: Initialized reader or None on error
        
    Note:
        GPU disabled (gpu=False) for maximum compatibility
    """
    global reader
    
    if not EASYOCR_AVAILABLE:
        return None
    
    try:
        # English and digits only for speed
        # Weights will be downloaded automatically on first run (~50MB)
        reader = easyocr.Reader(['en'], gpu=False)  # GPU=False for compatibility
        print("✅ EasyOCR initialized (deep learning OCR)")
        return reader
    except Exception as e:
        print(f"❌ EasyOCR init failed: {e}")
        return None


def run_ocr(image_bytes: bytes) -> List[Dict]:
    """Text recognition on floor plan image with post-processing.
    
    Applies advanced preprocessing to improve OCR quality
    and post-processing to correct typical recognition errors.
    
    Pipeline:
    1. Decode image from bytes
    2. Preprocessing:
        - CLAHE (adaptive histogram equalization)
        - Bilateral filter (edge-preserving smoothing)
        - Otsu thresholding (binarization)
    3. EasyOCR recognition with character filter
    4. Post-processing:
        - Filter by length and presence of digits
        - Correct typical errors (O→0, l→1, S→5)
        - Extract bbox coordinates
    
    Args:
        image_bytes: Image bytes (JPG/PNG)
        
    Returns:
        List[Dict]: List of recognized elements:
            - "text": str, recognized text
            - "bbox": [x_min, y_min, x_max, y_max], bbox coordinates
            - "confidence": float, model confidence (0-100)
            
    Raises:
        ValueError: If EasyOCR not initialized or image invalid
    """
    if reader is None:
        raise ValueError("EasyOCR not initialized")
    
    data = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot read image")
    
    # Improved preprocessing for technical drawings
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter for edge preservation
    bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Otsu thresholding
    _, binary = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # EasyOCR works with RGB
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    # EasyOCR recognition
    results = reader.readtext(
        rgb,
        allowlist='0123456789.,xмMм',  # Only digits and dimension symbols
        paragraph=False,
        min_size=8,  # Reduced for small text
        text_threshold=0.5,  # Lowered threshold to capture more
    )
    
    out: List[Dict] = []
    for bbox, text, conf in results:
        text = text.strip()
        
        # Post-processing: filter junk and fix errors
        if not text or len(text) > 10:  # Dimensions are usually short
            continue
        
        # Pattern for dimensions: X.XX or X,XX or just digits
        import re
        if not re.search(r'\d', text):  # Must have at least one digit
            continue
        
        # Fix typical OCR errors
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


@app.post("/ocr", summary="OCR dimensions and labels on plan (EasyOCR)")
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
