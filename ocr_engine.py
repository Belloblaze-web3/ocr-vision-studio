from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output

THRESHOLD = 80.0


def _font(size: int = 18):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def preprocess(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return BGR source pixels and a high-contrast OCR image."""
    source = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    thresholded = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return source, thresholded


def _extract_words(processed: np.ndarray, threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = pytesseract.image_to_data(processed, config="--oem 3 --psm 6", output_type=Output.DICT)
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for index, text in enumerate(data["text"]):
        text = str(text).strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if not text or confidence < 0:
            continue
        detection = {
            "text": text,
            "confidence": round(confidence, 2),
            "box": {
                "left": int(data["left"][index]),
                "top": int(data["top"][index]),
                "width": int(data["width"][index]),
                "height": int(data["height"][index]),
            },
        }
        (accepted if confidence >= threshold else review).append(detection)
    return accepted, review


def _annotate(source: np.ndarray, accepted: list[dict[str, Any]]) -> bytes:
    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    label_font = _font(16)
    for detection in accepted:
        box = detection["box"]
        x, y, w, h = box["left"], box["top"], box["width"], box["height"]
        draw.rectangle((x, y, x + w, y + h), outline="#36d399", width=3)
        label = f"{detection['text']}  {detection['confidence']:.1f}%"
        bbox = draw.textbbox((0, 0), label, font=label_font)
        label_y = max(0, y - (bbox[3] - bbox[1]) - 8)
        draw.rounded_rectangle(
            (x, label_y, x + (bbox[2] - bbox[0]) + 12, label_y + (bbox[3] - bbox[1]) + 8),
            radius=4,
            fill="#10231f",
        )
        draw.text((x + 6, label_y + 3), label, fill="#d9fff0", font=label_font)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def analyze_image(image: Image.Image, threshold: float = THRESHOLD) -> dict[str, Any]:
    source, processed = preprocess(image)
    accepted, review = _extract_words(processed, threshold)
    annotated = _annotate(source, accepted)
    all_detections = accepted + review
    average = sum(item["confidence"] for item in all_detections) / len(all_detections) if all_detections else 0
    return {
        "threshold": threshold,
        "accepted": accepted,
        "review": review,
        "accepted_count": len(accepted),
        "review_count": len(review),
        "total_count": len(all_detections),
        "recognized_text": " ".join(item["text"] for item in accepted),
        "average_confidence": round(average, 1),
        "image_size": {"width": image.width, "height": image.height},
        "annotated_png": annotated,
        "engine": "Tesseract OCR via pytesseract",
        "preprocessing": ["grayscale", "Gaussian blur", "adaptive thresholding"],
    }
