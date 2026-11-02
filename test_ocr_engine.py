from pathlib import Path

from PIL import Image

from ocr_engine import analyze_image  # type: ignore[import-not-found]


if __name__ == "__main__":
    sample = Path(__file__).parent / "public" / "sample" / "sample_text.png"
    result = analyze_image(Image.open(sample).convert("RGB"))
    assert result["threshold"] == 80.0
    assert result["accepted_count"] > 0
    assert all(item["confidence"] >= 80.0 for item in result["accepted"])
    assert result["annotated_png"].startswith(b"\x89PNG")
    print(f"OCR regression check passed: {result['accepted_count']} accepted detections")
