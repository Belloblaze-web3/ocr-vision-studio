# OCR Vision Studio

OCR Vision Studio is a polished Project 4 implementation for image-based text recognition. It turns an uploaded image into inspectable evidence: preprocessed pixels, word-level OCR, bounding boxes, confidence values, an annotated image, and exportable JSON.

## Stack

The application uses Flask for the web server, OpenCV for preprocessing, and Tesseract OCR through `pytesseract` for inference. The browser UI is dependency-free HTML, CSS, and JavaScript so the project stays easy to run and GitHub-ready.

## Features

The upload surface accepts PNG, JPG, WEBP, BMP, and TIFF images up to 10 MB. The server converts each image to grayscale, reduces noise with a Gaussian blur, and applies adaptive thresholding before running Tesseract in word-data mode. Accepted detections clear the fixed 80% confidence gate; lower-confidence tokens remain visible in the review queue. The report includes counts, average confidence, processing latency, recognized text, coordinates, and a labeled annotated image. JSON export preserves the detection metadata.

## Local setup

Install the Tesseract runtime first:

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

Then install the Python dependencies and start the app:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:3000` and upload an image. The server reads `PORT` when supplied; the default is `3000`.

## Docker deployment

```bash
docker build -t ocr-vision-studio .
docker run --rm -p 3000:3000 -e PORT=3000 ocr-vision-studio
```

The unauthenticated health endpoint is `GET /api/health` and returns JSON with the service and model status. The container uses Gunicorn and installs the Tesseract runtime during the image build.

## API contract

`POST /api/analyze` accepts a multipart form field named `image`. It returns `filename`, `threshold`, `accepted`, `review`, `recognized_text`, `accepted_count`, `review_count`, `average_confidence`, `processing_ms`, `image_size`, `engine`, `preprocessing`, and `annotated_image` as a PNG data URL. `GET /api/health` provides a lightweight readiness check.

## Project map

- `app.py` — Flask routes, upload validation, API response shaping.
- `ocr_engine.py` — OpenCV preprocessing, Tesseract integration, confidence gate, annotation renderer.
- `templates/index.html` — accessible upload workflow and results dashboard.
- `static/styles.css` — Signal Noir visual system and responsive layout.
- `static/app.js` — drag/drop, API calls, metrics, table rendering, downloads.
- `Dockerfile` — production container with Tesseract and Gunicorn.
- `plan.md` and `TODO.md` — approved design/implementation decisions and delivery criteria.

## Notes on inference

OCR confidence is produced by Tesseract and is not a calibrated probability. The 80% gate is a practical acceptance rule for this training project, not a claim that every accepted token is correct. For difficult imagery, improve the preprocessing branch, try a different Tesseract page segmentation mode, or add language packs in the container.
