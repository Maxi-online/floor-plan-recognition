"""Floorplan Cleanup Service.

Preprocessing service for floor plan images.
Performs perspective correction, denoising, and binarization
to improve subsequent recognition quality.

Main features:
    - Automatic perspective correction (perspective transform)
    - Detection and alignment of quadrilaterals (plans photographed at an angle)
    - Denoising (fastNlMeansDenoising)
    - Adaptive binarization for artifact removal

Author: Maksim Strekolovsky
Date: 10.12.2025
Version: 0.1.0
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
    """Find the largest quadrilateral in the image.
    
    Finds all contours on binary mask, approximates them
    and returns the largest quadrilateral (plan photographed at an angle).
    
    Args:
        mask: Binary mask (255 = object, 0 = background)
        
    Returns:
        Tuple[np.ndarray, float]: Tuple of:
            - Array of 4 quadrilateral points or None
            - Quadrilateral area
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
    """Plan perspective correction (quadrilateral alignment).
    
    Automatically finds the largest quadrilateral in the image
    (assumed to be the plan) and applies perspective transform
    to align it into a rectangle.
    
    Algorithm:
    1. Convert to grayscale and binarize (Otsu)
    2. Find largest quadrilateral
    3. Order corners (top-left, top-right, bottom-right, bottom-left)
    4. Calculate output rectangle dimensions
    5. Apply perspective transform
    
    Args:
        image: Input image in BGR format
        
    Returns:
        np.ndarray: Aligned image or original if quadrilateral not found
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
    """Remove noise and artifacts from image.
    
    Applies adaptive binarization and morphological operations
    to remove small dots, spots, and other artifacts.
    
    Pipeline:
    1. Gaussian blur for smoothing
    2. Adaptive thresholding for binarization
    3. Morphological opening to remove small objects
    4. Morphological closing to fill gaps
    
    Args:
        image: Input image in BGR format
        
    Returns:
        np.ndarray: Binary image without noise (255 = object, 0 = background)
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
    """Full floor plan image cleanup pipeline.
    
    Performs all preprocessing stages:
    1. Image decoding
    2. Perspective correction
    3. Denoising (Non-Local Means Denoising)
    4. Binarization and artifact removal
    5. PNG encoding
    
    Args:
        file_bytes: Image bytes (JPG/PNG)
        
    Returns:
        bytes: Processed binary image in PNG format
        
    Raises:
        ValueError: If image is invalid or encoding error
    """
    data = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot read image")
    warped = warp_perspective(bgr)
    denoised = cv2.fastNlMeansDenoisingColored(warped, None, 10, 10, 7, 21)
    bin_mask = despeckle(denoised)
    # Return binary image suitable for subsequent pipeline
    ok, buf = cv2.imencode(".png", bin_mask)
    if not ok:
        raise ValueError("PNG encoding error")
    return buf.tobytes()


@app.post("/clean", summary="Perspective normalization and denoising")
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

