# OCR Vision Studio

[![CI](https://github.com/Belloblaze-web3/ocr-vision-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/Belloblaze-web3/ocr-vision-studio/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

OCR Vision Studio is a polished Project 4 AI vision platform with two selectable inference paths: **Option 1 — Optical Character Recognition** and **Option 2 — Object Detection**. It turns an uploaded image into inspectable evidence: model output, bounding boxes, confidence values, an annotated image, and exportable JSON.

## Modes

**OCR mode** uses Python, OpenCV, and Tesseract OCR. The server converts the source to grayscale, reduces noise with a Gaussian blur, applies adaptive thresholding, and runs word-level Tesseract inference. Accepted text tokens clear the fixed 80% confidence gate; lower-confidence tokens remain in the review queue.

**Object Detection mode** now uses YOLOv8n pretrained on COCO as the primary detector and retains MobileNet-SSD through OpenCV DNN as a comparison model. YOLO covers a broader set of household categories, including `tv`, `remote`, `laptop`, `chair`, `couch`, `table`, and `person`. Primary detections clear a 70% confidence gate and are returned with object labels, confidence scores, coordinates, and blue annotated bounding boxes. The MobileNet-SSD Caffe files download automatically into `models/` on first use and are returned as secondary comparison detections.

Important limitation: even COCO does not have dedicated classes for DVD players, receivers, speakers, electrical sockets, or woofers. It can still map an unfamiliar electronic object to a nearby learned class. The included custom-training workflow is the correct route for those parlour-specific categories.

## Custom parlour training

The repository includes `train_custom.py`, `datasets/parlour.yaml.example`, and annotation guidance. Copy the YAML example, add YOLO-format labeled images for `tv`, `dvd_player`, `receiver`, `speaker`, `socket`, `woofer`, `chair`, `sofa`, `table`, and `person`, then train:

```bash
cp datasets/parlour.yaml.example datasets/parlour.yaml
python train_custom.py --data datasets/parlour.yaml --epochs 100
```

Point the running app at the resulting checkpoint with `CUSTOM_MODEL_PATH=runs/parlour-vision/weights/best.pt`. At least 50–100 varied labeled images per class is a practical starting point; small objects such as sockets need especially careful boxes and varied lighting.

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

Open `http://localhost:3000`, choose OCR or Object Detection, upload an image, and run the selected analysis. The server reads `PORT` when supplied; the default is `3000`. Object Detection downloads the standard YOLOv8n and MobileNet-SSD assets on first use when they are not already cached; see `models/README.md` for the comparison-model source.

## Docker deployment

```bash
docker build -t ocr-vision-studio .
docker run --rm -p 3000:3000 -e PORT=3000 ocr-vision-studio
```

The unauthenticated health endpoint is `GET /api/health` and returns the service plus both available model names. The container uses Gunicorn, installs the Tesseract runtime during image build, honors `PORT`, and defaults to one worker to avoid duplicating the YOLO model in memory on small instances.

## Render deployment

The repository includes `render.yaml` for a Docker-based Render Web Service. In Render, choose **New → Blueprint**, connect `Belloblaze-web3/ocr-vision-studio`, and apply the blueprint. It provisions the `ocr-vision-studio` service, uses `/api/health` for readiness checks, and enables automatic deploys from `main`. The free plan may suspend the service when idle; the first request can take longer while the container starts and downloads model assets.

For a custom parlour checkpoint, add `CUSTOM_MODEL_PATH` as a Render environment variable pointing to a durable packaged path or adapt the Docker image to include the trained weights. Do not commit private training images or secrets.

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
