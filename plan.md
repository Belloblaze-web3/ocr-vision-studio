# OCR Vision Studio — implementation plan

## Product scope

OCR Vision Studio is a GitHub-ready Python web application for the selected Project 4 path: OCR text recognition. A user uploads an image, the server preprocesses it, runs Tesseract OCR through `pytesseract`, keeps detections at or above an 80% confidence threshold, and returns recognized text, bounding boxes, confidence scores, an annotated image, and machine-readable JSON.

The platform now also includes Option 2 — object detection. A mode selector routes the same upload to MobileNet-SSD through OpenCV DNN, using the VOC class set, a stricter 70% detection threshold, and non-maximum suppression. Object labels, confidence scores, bounding boxes, and blue annotated overlays are returned through the same dashboard and export flow.

## Implementation

- **Backend:** Flask serves the responsive UI, `/api/analyze` for multipart image processing, and `/api/health` for deployment checks.
- **Vision pipeline:** OpenCV converts the upload to grayscale, applies a small Gaussian blur, and uses adaptive thresholding. Tesseract `image_to_data` returns word-level text, coordinates, and confidence values. The alternate MobileNet-SSD path loads the Caffe model, runs a 300×300 DNN blob, filters detections at 70%, applies non-maximum suppression, and draws labeled boxes onto the original image.
- **Frontend:** A dependency-free HTML/CSS/JavaScript interface provides mode selection, drag-and-drop upload, validation, processing states, preview, result metrics, confidence values, detection table, JSON/text downloads, and a reset flow.
- **Deployment:** `Dockerfile` installs the Tesseract runtime and Python packages. `requirements.txt`, `.env.example`, and README instructions cover local and container usage.
- **Project structure:** `app.py` owns HTTP and mode routing; `ocr_engine.py` contains the testable OCR and object-detection pipelines; `models/` documents cached MobileNet-SSD assets; `templates/index.html` is the app shell; `static/styles.css`, `static/mode.css` contain the visual system; `static/app.js` contains browser state and rendering; `public/manus-routes.json` declares page routes.

## Design direction

Three possible directions were considered: **Signal Noir / mission-control dashboard** (0.07), **Paper Lab / editorial research tool** (0.06), and **Soft Spectrum / friendly AI utility** (0.05). The chosen direction is Signal Noir.

- **Design Movement:** Signal Noir blends industrial mission-control interfaces with editorial data tooling.
- **Core Principles:** evidence first; quiet confidence; structured density; purposeful motion.
- **Color Philosophy:** a near-black blue canvas makes the image evidence feel luminous; cobalt marks actions and controls; teal signals healthy model output; amber flags uncertainty without alarm.
- **Layout Paradigm:** an asymmetric split workspace: a narrow operational rail anchors identity and system status, while the main area moves from upload evidence to result evidence rather than using a generic centered grid.
- **Signature Elements:** a vertical signal rail, thin technical dividers, and confidence meters that read like instrument panels.
- **Interaction Philosophy:** every state has a clear status line, a visible next action, and no hidden processing; drag/drop and button upload converge on the same flow.
- **Animation:** subtle scan-line shimmer while processing, 180–240ms panel reveals, and no decorative looping motion after results arrive.
- **Typography System:** `Manrope` for interface text with `IBM Plex Mono` for metadata, coordinates, and confidence values; large headlines are compact and slightly tracked out.
- **Brand Essence:** “Turn pixels into evidence, not guesswork.” Precise, analytical, direct.
- **Brand Voice:** CTAs say “Run recognition” and “Inspect evidence”; helper text says “Every accepted token clears the 80% gate.”
- **Wordmark & Logo:** a square mark built from three staggered scan bars and a small cobalt focus dot, paired with the OCR Vision name.
- **Signature Brand Color:** electric cobalt `#6d7cff`, reserved for the primary action and focus state.
