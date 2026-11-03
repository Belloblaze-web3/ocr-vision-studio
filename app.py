from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

from ocr_engine import analyze_image, analyze_objects  # type: ignore[import-not-found]

BASE_DIR = Path(__file__).resolve().parent
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
Image.MAX_IMAGE_PIXELS = 25_000_000


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/manus-routes.json")
def routes_manifest():
    return send_from_directory(BASE_DIR / "public", "manus-routes.json")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "ocr-vision-studio", "models": ["Tesseract OCR", "YOLOv8n COCO", "MobileNet-SSD", "custom YOLO checkpoint (optional)"]})


@app.post("/api/analyze")
def analyze():
    started = time.perf_counter()
    upload = request.files.get("image")
    mode = request.form.get("mode", "ocr").lower()
    if mode not in {"ocr", "objects"}:
        return jsonify({"error": "Mode must be 'ocr' or 'objects'."}), 400
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose an image before running analysis."}), 400
    if not allowed_file(upload.filename):
        return jsonify({"error": "Unsupported format. Use PNG, JPG, WEBP, BMP, or TIFF."}), 415
    raw = upload.read()
    if not raw:
        return jsonify({"error": "The uploaded image is empty."}), 400
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "Images must be 10 MB or smaller."}), 413
    try:
        source = Image.open(io.BytesIO(raw)).convert("RGB")
        result = analyze_objects(source) if mode == "objects" else analyze_image(source, threshold=80.0)
        annotated = result.pop("annotated_png")
        result.update({"filename": upload.filename, "processing_ms": round((time.perf_counter() - started) * 1000), "annotated_image": data_url(annotated)})
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Vision analysis failed")
        return jsonify({"error": f"{mode.title()} analysis failed: {exc}"}), 422


@app.get("/sample/<path:filename>")
def sample(filename: str):
    return send_from_directory(BASE_DIR / "public" / "sample", filename)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Images must be 10 MB or smaller."}), 413


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "API route not found.", "path": request.path}), 404
    return error


@app.errorhandler(405)
def method_not_allowed(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "HTTP method not allowed for this API route."}), 405
    return error


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "3000")), debug=os.environ.get("FLASK_DEBUG") == "1")
