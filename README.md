# Floor Plan Recognition Service

A computer vision system for automatic extraction of structural elements from architectural floor plan images. This service processes scanned or photographed floor plans and outputs structured JSON data containing wall coordinates, room boundaries, and spatial relationships.

## Overview

The Floor Plan Recognition Service is a prototype system designed to automate the digitization of architectural drawings. It employs a hybrid approach combining classical computer vision techniques (Hough Transform) with state-of-the-art deep learning models (SAM 2.1) to detect and extract structural elements from floor plan images.

### Key Features

- **Wall Detection**: Automatic extraction of wall lines using Probabilistic Hough Transform with post-processing
- **Image Preprocessing**: Advanced pipeline with CLAHE, bilateral filtering, and adaptive thresholding
- **Room Segmentation**: Experimental support for room boundary detection via SAM 2.1 and OCR-based methods
- **RESTful API**: FastAPI-based microservices architecture
- **Telegram Bot Interface**: User-friendly interface for quick testing and demonstration
- **Docker Deployment**: Containerized setup for easy deployment and reproducibility

## Architecture

The system consists of multiple microservices:

```
┌─────────────────┐
│  Telegram Bot   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Hybrid Service (Port 8003)        │
│   • Wall Detection (Hough)         │
│   • Room Segmentation (SAM2/OCR)   │
└────────┬────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────────┐
│ Cleanup │ │ OCR Service  │
│ Service │ │ (Port 8002)  │
│(Port    │ └──────────────┘
│ 8001)   │
└─────────┘
```

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|------------|
| **Wall Detection** | OpenCV Hough Transform | Classical CV method, fast, interpretable, no training required |
| **Preprocessing** | OpenCV (CLAHE, bilateral, morphology) | Industry standard for image processing, well-documented |
| **Room Segmentation** | SAM 2.1 Large (Meta) | State-of-the-art zero-shot segmentation (2024), high accuracy |
| **OCR** | EasyOCR | Best open-source alternative to Tesseract, supports multiple languages |
| **Backend Framework** | FastAPI | Modern, fast Python web framework with automatic API documentation |
| **Interface** | Telegram Bot (python-telegram-bot) | Simple UX without web development, easy deployment |
| **Deployment** | Docker & Docker Compose | Reproducible environment, easy deployment, service orchestration |

## Processing Pipeline

### Stage 1: Image Preprocessing

The preprocessing stage enhances image quality and prepares it for feature extraction:

```
Input Image (JPG/PNG)
    ↓
CLAHE (Contrast Limited Adaptive Histogram Equalization)
    ↓
Bilateral Filter (Edge-preserving smoothing)
    ↓
Otsu + Adaptive Thresholding (Binarization)
    ↓
Morphological Operations (Noise removal & gap closing)
    ↓
Binary Image (Ready for feature extraction)
```

**Key Parameters:**
- CLAHE: `clipLimit=3.0`, `tileGridSize=(8, 8)`
- Bilateral Filter: `d=9`, `sigmaColor=75`, `sigmaSpace=75`
- Adaptive Threshold: `blockSize=21`, `C=10`

### Stage 2: Wall Detection

Wall detection uses a multi-step approach to extract linear features:

```
Binary Image
    ↓
Skeletonization (scikit-image thin algorithm)
    ↓
Probabilistic Hough Transform
    • threshold=30 (optimized for interior walls)
    • minLineLength=20 pixels
    • maxLineGap=35 pixels
    ↓
Snap to Axis (Alignment to 0°/45°/90°/135°)
    ↓
Deduplication (Remove overlapping segments)
    ↓
Merge Collinear Segments (max_angle=3°, max_gap=50px)
    ↓
JSON Output (Wall coordinates with unique IDs)
```

## Output Format

The service returns structured JSON data:

```json
{
  "meta": {
    "source": "plan.png",
    "width": 1920,
    "height": 1080,
    "model": "hough_transform_v1.0"
  },
  "walls": [
    {
      "id": "w1",
      "points": [[100, 200], [500, 200]]
    },
    {
      "id": "w2",
      "points": [[500, 200], [500, 600]]
    }
  ],
  "rooms": [
    {
      "id": "r1",
      "polygon": [[100, 200], [500, 200], [500, 600], [100, 600]],
      "area": 120000,
      "area_sqm": 12.5,
      "label": "12.5"
    }
  ]
}
```

## Installation

### Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.0+
- Minimum 8GB RAM (16GB recommended)
- 10GB free disk space

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd room-detector
   ```

2. **Create configuration file:**
   ```bash
   echo '{"telegram_bot_token":"YOUR_BOT_TOKEN","hybrid_url":"http://localhost:8003/detect"}' > settings_secret.json
   ```
   
   To get a Telegram bot token:
   - Open [@BotFather](https://t.me/BotFather) on Telegram
   - Send `/newbot` and follow instructions
   - Copy the provided token

3. **Start services:**
   ```bash
   # Windows
   docker-run.bat
   
   # Linux/Mac
   docker-compose up -d
   ```

4. **Verify deployment:**
   ```bash
   curl http://localhost:8003/health
   ```
   
   Expected response:
   ```json
   {"status": "ok", "sam2_large": true}
   ```

For detailed installation instructions, see [QUICKSTART.md](QUICKSTART.md) and [DOCKER.md](DOCKER.md).

## Usage

### Via Telegram Bot

1. Open your Telegram bot
2. Send `/start` to initialize
3. Send a floor plan image (JPG/PNG)
4. Wait 10-20 seconds for processing
5. Receive:
   - Visualization image with detected walls
   - JSON file with coordinates

### Via REST API

**Endpoint:** `POST http://localhost:8003/detect`

**Request:**
```bash
curl -X POST "http://localhost:8003/detect" \
  -F "file=@floor_plan.png"
```

**Response:**
```json
{
  "meta": {"source": "floor_plan.png"},
  "walls": [...],
  "rooms": [...]
}
```

## Current Limitations

### Accuracy Issues

- **Wall Detection Accuracy**: ~60-70% on real-world floor plans
  - Interior walls with breaks are often missed
  - Low-contrast walls on technical drawings may not be detected
  - False positives from furniture, text, and hatching patterns

- **Room Segmentation**: Experimental, not production-ready
  - SAM 2.1 may segment furniture instead of rooms
  - OCR-based method requires clear area labels on plans
  - Limited accuracy without fine-tuned models

- **Geometric Artifacts**:
  - Snap-to-axis may distort non-orthogonal angles
  - Segment merging can incorrectly combine different walls
  - Wall thickness is not considered

### Technical Constraints

- **Performance**: 10-20 seconds per plan (CPU-only mode)
- **Memory**: Requires minimum 4GB RAM, 8GB recommended
- **Scale**: No automatic calibration (output in pixels, not meters)
- **Model Size**: SAM 2.1 Large requires ~224MB download on first run

## Roadmap

### Version 2.0 (Planned)

**Short-term (1-2 months):**
- Dataset collection and annotation (500+ floor plans)
- Fine-tuning YOLO v11 for wall detection (target: 90%+ accuracy)
- Improved filtering with ML-based wall/not-wall classifier
- Graph-based post-processing for wall connectivity

**Medium-term (3-6 months):**
- Custom UNet for room segmentation
- Door and window detection (separate YOLO detector)
- REST API with OpenAPI documentation
- Web-based visualization interface
- Batch processing support

**Long-term (6-12 months):**
- 3D reconstruction from plans and sections
- Export to CAD formats (DXF, DWG)
- AR/VR integration for plan visualization
- Automatic scale calibration via OCR dimensions

### Production Requirements

To achieve production-ready accuracy (>90%), the following is required:

1. **Dataset**: 500-1000 annotated floor plans with:
   - Wall lines (polylines)
   - Room polygons
   - Door and window positions
   - Dimension labels

2. **Model Fine-tuning**:
   - YOLO v8/v11 for wall/door/window detection
   - UNet/DeepLab for room segmentation
   - Custom CRNN-based OCR for dimensions

3. **Pipeline Improvements**:
   - Graph-based connectivity analysis
   - Context-aware filtering
   - Wall thickness estimation
   - Scale auto-calibration

## API Documentation

### Services

| Service | Port | Endpoint | Description |
|---------|------|----------|-------------|
| Hybrid Service | 8003 | `/detect` | Main recognition endpoint |
| Hybrid Service | 8003 | `/health` | Health check |
| OCR Service | 8002 | `/ocr` | Text recognition |
| Cleanup Service | 8001 | `/clean` | Image preprocessing |

### Health Check

```bash
curl http://localhost:8003/health
```

Response:
```json
{
  "status": "ok",
  "sam2_large": true
}
```

## Development

### Project Structure

```
room-detector/
├── services/
│   ├── hybrid_service.py      # Main recognition service
│   ├── ocr_service.py          # OCR service (EasyOCR)
│   └── cleanup_service.py      # Image preprocessing
├── tg_bot.py                   # Telegram bot interface
├── settings.py                 # Configuration management
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Service orchestration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### Running Locally (Without Docker)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   export TELEGRAM_BOT_TOKEN="your_token"
   export HYBRID_URL="http://localhost:8003/detect"
   ```

3. **Start services:**
   ```bash
   # Terminal 1: Hybrid Service
   uvicorn services.hybrid_service:app --port 8003
   
   # Terminal 2: OCR Service
   uvicorn services.ocr_service:app --port 8002
   
   # Terminal 3: Cleanup Service
   uvicorn services.cleanup_service:app --port 8001
   
   # Terminal 4: Telegram Bot
   python tg_bot.py
   ```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is provided as-is for research and development purposes.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{floor_plan_recognition,
  title = {Floor Plan Recognition Service},
  author = {Strekolovsky, Maksim},
  year = {2025},
  url = {https://github.com/yourusername/room-detector}
}
```

## Acknowledgments

- **SAM 2.1** by Meta AI for segmentation capabilities
- **EasyOCR** for text recognition
- **OpenCV** community for computer vision tools
- **FastAPI** for the excellent web framework

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check [QUICKSTART.md](QUICKSTART.md) for setup help
- Review [DOCKER.md](DOCKER.md) for deployment details

---

**Status**: Prototype / Proof of Concept  
**Version**: 1.0  
**Last Updated**: December 2025
