from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom parlour object detector")
    parser.add_argument("--data", type=Path, default=Path("datasets/parlour.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="base YOLO checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="parlour-vision")
    args = parser.parse_args()
    if not args.data.exists():
        raise SystemExit(f"Dataset config not found: {args.data}. Copy datasets/parlour.yaml.example first.")
    model = YOLO(args.model)
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, name=args.name, project="runs")
