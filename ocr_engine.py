from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
from pytesseract import Output

THRESHOLD = 80.0
OBJECT_THRESHOLD = 0.70
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
PROTO_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
MODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
PROTO_PATH = MODEL_DIR / "deploy.prototxt"
MODEL_PATH = MODEL_DIR / "mobilenet_iter_73000.caffemodel"
YOLO_MODEL_NAME = os.environ.get("YOLO_MODEL", str(MODEL_DIR / "yolov8n.pt"))
CUSTOM_MODEL_PATH = os.environ.get("CUSTOM_MODEL_PATH", "")
_MOBILE_NET: Any | None = None
_YOLO_MODEL: Any | None = None

OBJECT_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


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
            "box": {"left": int(data["left"][index]), "top": int(data["top"][index]),
                    "width": int(data["width"][index]), "height": int(data["height"][index])},
        }
        (accepted if confidence >= threshold else review).append(detection)
    return accepted, review


def _ensure_object_model() -> Any:  # type: ignore[name-defined]
    """Download the small MobileNet-SSD Caffe files on first object-detection use."""
    global _MOBILE_NET
    if _MOBILE_NET is not None:
        return _MOBILE_NET
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in ((PROTO_PATH, PROTO_URL), (MODEL_PATH, MODEL_URL)):
        if not path.exists() or path.stat().st_size < 1024:
            temporary = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
    _MOBILE_NET = cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(MODEL_PATH))  # type: ignore[attr-defined]
    return _MOBILE_NET


def _object_detections(source: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    net = _ensure_object_model()
    height, width = source.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(source, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    output = net.forward()
    candidates: list[dict[str, Any]] = []
    boxes: list[list[int]] = []
    scores: list[float] = []
    for index in range(output.shape[2]):
        confidence = float(output[0, 0, index, 2])
        class_index = int(output[0, 0, index, 1])
        if confidence < threshold or class_index >= len(OBJECT_CLASSES):
            continue
        box_values = output[0, 0, index, 3:7] * np.array([width, height, width, height])
        x1, y1, x2, y2 = box_values.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        detection = {
            "label": OBJECT_CLASSES[class_index],
            "confidence": round(confidence * 100, 2),
            "box": {"left": int(x1), "top": int(y1), "width": int(max(0, x2 - x1)), "height": int(max(0, y2 - y1))},
        }
        candidates.append(detection)
        boxes.append([int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))])
        scores.append(confidence)
    keep = cv2.dnn.NMSBoxes(boxes, scores, threshold, 0.35) if boxes else []
    indices = np.array(keep).reshape(-1).tolist() if len(keep) else []
    detections = [candidates[index] for index in indices]
    return sorted(detections, key=lambda item: item["confidence"], reverse=True)


def _yolo_detections(source: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    """Run a stronger COCO detector, or a user-trained model when configured."""
    global _YOLO_MODEL
    from ultralytics import YOLO  # type: ignore[import-not-found]

    model_ref = CUSTOM_MODEL_PATH or YOLO_MODEL_NAME
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(model_ref)
    results = _YOLO_MODEL.predict(source, conf=threshold, verbose=False)  # type: ignore[union-attr]
    detections: list[dict[str, Any]] = []
    for result in results:
        names = result.names
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            class_index = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            label = names[class_index] if isinstance(names, dict) else names[class_index]
            detections.append({
                "label": str(label), "confidence": round(confidence * 100, 2),
                "box": {"left": x1, "top": y1, "width": max(0, x2 - x1), "height": max(0, y2 - y1)},
            })
    return sorted(detections, key=lambda item: item["confidence"], reverse=True)


def _annotate(source: np.ndarray, detections: list[dict[str, Any]], mode: str) -> bytes:
    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    label_font = _font(16)
    color = "#36d399" if mode == "ocr" else "#6d7cff"
    label_color = "#d9fff0" if mode == "ocr" else "#e5e8ff"
    fill = "#10231f" if mode == "ocr" else "#151b3b"
    for detection in detections:
        box = detection["box"]
        x, y, w, h = box["left"], box["top"], box["width"], box["height"]
        draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
        label_text = detection.get("text", detection.get("label", "object"))
        label = f"{label_text}  {detection['confidence']:.1f}%"
        bbox = draw.textbbox((0, 0), label, font=label_font)
        label_y = max(0, y - (bbox[3] - bbox[1]) - 8)
        draw.rounded_rectangle((x, label_y, x + (bbox[2] - bbox[0]) + 12, label_y + (bbox[3] - bbox[1]) + 8), radius=4, fill=fill)
        draw.text((x + 6, label_y + 3), label, fill=label_color, font=label_font)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def analyze_image(image: Image.Image, threshold: float = THRESHOLD) -> dict[str, Any]:
    source, processed = preprocess(image)
    accepted, review = _extract_words(processed, threshold)
    annotated = _annotate(source, accepted, "ocr")
    all_detections = accepted + review
    average = sum(item["confidence"] for item in all_detections) / len(all_detections) if all_detections else 0
    return {
        "mode": "ocr", "threshold": threshold, "accepted": accepted, "review": review,
        "accepted_count": len(accepted), "review_count": len(review), "total_count": len(all_detections),
        "recognized_text": " ".join(item["text"] for item in accepted), "average_confidence": round(average, 1),
        "image_size": {"width": image.width, "height": image.height}, "annotated_png": annotated,
        "engine": "Tesseract OCR via pytesseract", "preprocessing": ["grayscale", "Gaussian blur", "adaptive thresholding"],
    }


def analyze_objects(image: Image.Image, threshold: float = OBJECT_THRESHOLD) -> dict[str, Any]:
    source = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    yolo_threshold = max(0.25, threshold)
    yolo_detections = _yolo_detections(source, yolo_threshold)
    mobile_detections = _object_detections(source, threshold)
    detections = yolo_detections
    annotated = _annotate(source, detections, "objects")
    average = sum(item["confidence"] for item in detections) / len(detections) if detections else 0
    return {
        "mode": "objects", "threshold": round(yolo_threshold * 100, 1), "detections": detections,
        "secondary_detections": mobile_detections,
        "accepted_count": len(detections), "review_count": 0, "total_count": len(detections),
        "recognized_text": ", ".join(item["label"] for item in detections), "average_confidence": round(average, 1),
        "image_size": {"width": image.width, "height": image.height}, "annotated_png": annotated,
        "engine": "YOLOv8n COCO + MobileNet-SSD comparison", "model_sources": ["custom YOLO checkpoint" if CUSTOM_MODEL_PATH else "YOLOv8n COCO", "MobileNet-SSD VOC"],
        "preprocessing": ["YOLO letterbox inference", "MobileNet-SSD 300×300 blob", "confidence filter", "non-maximum suppression"],
        "supported_classes": OBJECT_CLASSES[1:],
        "model_note": "Neither the pretrained COCO nor VOC model has dedicated DVD-player, receiver, speaker, socket, or woofer classes; use the included custom-training pipeline for those categories.",
    }
