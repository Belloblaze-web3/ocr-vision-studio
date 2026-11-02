# OCR Vision Studio

OCR Vision Studio is a polished Project 4 AI vision platform with two selectable inference paths: **Option 1 — Optical Character Recognition** and **Option 2 — Object Detection**. It turns an uploaded image into inspectable evidence: model output, bounding boxes, confidence values, an annotated image, and exportable JSON.

## Modes

**OCR mode** uses Python, OpenCV, and Tesseract OCR. The server converts the source to grayscale, reduces noise with a Gaussian blur, applies adaptive thresholding, and runs word-level Tesseract inference. Accepted text tokens clear the fixed 80% confidence gate; lower-confidence tokens remain in the review queue.

**Object Detection mode** uses MobileNet-SSD through OpenCV DNN. The model detects the 20 VOC classes supported by the pretrained network, including person, bicycle, car, bus, dog, cat, bottle, and other common objects. Detections clear a 50% confidence gate and are returned with object labels, confidence scores, coordinates, and blue annotated bounding boxes. The Caffe model files download automatically into `models/` on the first object-detection request and are cached for subsequent runs.

## Local setup

Install the Tesseract runtime first:

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

Then install dependencies and start the app:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:3000`, choose OCR or Object Detection, upload an image, and run the selected analysis. The server reads `PORT` when supplied; the default is `3000`. Object Detection needs internet access the first time it downloads the MobileNet-SSD model assets; see `models/README.md` for the source.

## Docker deployment

```bash
docker build -t ocr-vision-studio .
docker run --rm -p 3000:3000 -e PORT=3000 ocr-vision-studio
```

The unauthenticated health endpoint is `GET /api/health` and returns the service plus both available model names. The container uses Gunicorn, installs the Tesseract runtime during image build, and honors `PORT`.

## API contract

`POST /api/analyze` accepts a multipart form field named `image` and an optional `mode` field with value `ocr` or `objects` (default `ocr`). Both modes return `mode`, `threshold`, `accepted_count`, `review_count`, `total_count`, `average_confidence`, `processing_ms`, `image_size`, `engine`, `preprocessing`, and `annotated_image`. OCR results include `accepted`, `review`, and `recognized_text`; object results include `detections`, where each item has `label`, `confidence`, and `box` coordinates. `GET /api/health` provides a lightweight readiness check.

## Project map

- `app.py` — Flask routes, upload validation, mode routing, and API response shaping.
- `ocr_engine.py` — OpenCV preprocessing, Tesseract integration, MobileNet-SSD integration, confidence gates, and annotation renderer.
- `models/README.md` — MobileNet-SSD asset source and caching behavior.
- `templates/index.html` — accessible mode selector, upload workflow, and results dashboard.
- `static/styles.css` and `static/mode.css` — Signal Noir visual system and responsive layout.
- `static/app.js` — mode state, drag/drop, API calls, metrics, table rendering, and downloads.
- `Dockerfile` — production container with Tesseract and Gunicorn.
- `plan.md` and `TODO.md` — implementation decisions and delivery evidence.

## Notes on inference

OCR confidence is produced by Tesseract and is not a calibrated probability. MobileNet-SSD confidence is the network's object score. Both thresholds are practical training-project acceptance rules, not guarantees of correctness. For difficult imagery, improve preprocessing, add language packs, or replace the model with a domain-specific detector.
