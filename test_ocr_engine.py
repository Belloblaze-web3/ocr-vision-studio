from pathlib import Path

from PIL import Image

from ocr_engine import analyze_image, analyze_objects  # type: ignore[import-not-found]


if __name__ == "__main__":
    sample = Path(__file__).parent / "public" / "sample" / "sample_text.png"
    image = Image.open(sample).convert("RGB")
    ocr_result = analyze_image(image)
    assert ocr_result["threshold"] == 80.0
    assert ocr_result["accepted_count"] > 0
    assert all(item["confidence"] >= 80.0 for item in ocr_result["accepted"])
    assert ocr_result["annotated_png"].startswith(b"\x89PNG")
    object_result = analyze_objects(image)
    assert object_result["mode"] == "objects"
    assert object_result["threshold"] == 70.0
    assert object_result["model_sources"][0] == "YOLOv8n COCO"
    assert "MobileNet-SSD VOC" in object_result["model_sources"]
    assert object_result["annotated_png"].startswith(b"\x89PNG")
    print(f"Vision regression check passed: {ocr_result['accepted_count']} OCR tokens; {object_result['total_count']} objects")
