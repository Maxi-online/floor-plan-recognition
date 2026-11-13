# Floor Plan Recognition

**Extract wall geometry from raster floor plans and expose it through a FastAPI service and a Telegram interface.**

This project turns scanned or photographed architectural drawings into editable
wall segments. It combines an inspectable OpenCV pipeline with experimental
OCR-assisted and pretrained-model segmentation components.

## Engineering focus

- Image preprocessing, line detection and geometric post-processing are separate stages.
- Classical wall extraction can be inspected without training a task-specific model.
- A shared recognition service supports both HTTP uploads and a Telegram client.
- OCR/watershed and SAM experiments explore room boundaries without conflating them with the current public response contract.

## Implemented pipeline

```mermaid
flowchart LR
    A[Floor-plan image] --> B[CLAHE / filtering / thresholding]
    B --> C[Skeletonization and Hough lines]
    C --> D[Axis snapping / deduplication / segment merging]
    D --> E[Wall-coordinate JSON]
    A --> F[Experimental OCR / watershed / SAM room processing]
    E --> G[FastAPI and Telegram client]
```

The implementation uses OpenCV, scikit-image, FastAPI, EasyOCR and an optional
SAM 2.1 model loaded through Ultralytics. These are third-party components; this
repository supplies the application pipeline and integration around them.

## Current API contract

| Service | Port | Endpoint | Purpose |
|---|---:|---|---|
| Hybrid recognition | 8003 | `POST /detect` | Upload an image using multipart field `file`; receive `meta` and `walls` |
| Hybrid recognition | 8003 | `GET /health` | Process status and whether the SAM model loaded |
| Cleanup | 8001 | `/clean` | Image preprocessing |
| OCR | 8002 | `/ocr` | Text recognition |

**Room processing is experimental:** `process_image()` calculates candidate
rooms internally, but the current `/detect` response returns `meta` and `walls`.
It does not currently export those room polygons. Coordinates are in pixels;
physical scale and CAD reconstruction are not established by this API.

## Run the service

Docker and Docker Compose are required for the container workflow. Recognition
models may require an initial download and their own runtime dependencies.

```bash
git clone https://github.com/Maxi-online/floor-plan-recognition.git
cd floor-plan-recognition
docker compose up --build -d
curl http://localhost:8003/health
```

The Compose file starts the application services in **one container** and exposes
the three ports. It mounts `settings_secret.json` and persists model caches.
Configure that local JSON file before using the Telegram interface:

- `telegram_bot_token`: the bot credential issued by Telegram;
- `hybrid_url`: `http://localhost:8003/detect` for the supplied single-container setup.

Keep personal credentials out of published configuration. An HTTP health response
with `sam2_large: false` means the optional model is unavailable; it is not an
accuracy measurement. The generated upload interface is at
`http://localhost:8003/docs`.

For the existing platform-specific instructions, see [QUICKSTART.md](QUICKSTART.md)
and [DOCKER.md](DOCKER.md). Source entry points are
[hybrid_service.py](services/hybrid_service.py),
[tg_bot.py](tg_bot.py) and [settings.py](settings.py).

## Design trade-offs and validation

Hough-based walls are explainable and inexpensive relative to a learned detector,
but text, furniture and hatching can produce false positives. Axis snapping and
segment merging can distort oblique or closely spaced geometry. Room experiments
depend on legible labels and segmentation behaviour.

The repository does not include a reproducible labelled benchmark establishing
wall/room accuracy or processing-time guarantees. Useful next measurements are
wall precision/recall, room IoU, connectivity errors and latency on a versioned
set of drawings. Fine-tuning, door/window recognition, scale calibration and CAD
export remain possible extensions rather than delivered capabilities.

## Project status and attribution

Research/application prototype. OpenCV, EasyOCR, Ultralytics and SAM retain their
own licenses and attribution. No standalone project license file is currently
included in this repository.
