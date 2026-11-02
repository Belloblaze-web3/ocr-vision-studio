# Custom parlour-object training

To teach the detector categories that MobileNet-SSD and COCO do not contain, create a labeled YOLO dataset. Suggested classes are `tv`, `dvd_player`, `receiver`, `speaker`, `socket`, `woofer`, `chair`, `sofa`, `table`, and `person`.

Copy `parlour.yaml.example` to `parlour.yaml`. Put images in `datasets/parlour/images/train` and `datasets/parlour/images/val`; put one matching YOLO label text file per image in `datasets/parlour/labels/train` and `datasets/parlour/labels/val`. Each label line is `class_id center_x center_y width height` with coordinates normalized from 0 to 1.

For useful results, collect at least 50–100 varied images per class with different lighting, angles, distances, and partial occlusions. Draw tight bounding boxes around every instance. More data is better for small objects such as sockets and remote speakers.

Train with:

```bash
cp datasets/parlour.yaml.example datasets/parlour.yaml
python train_custom.py --data datasets/parlour.yaml --epochs 100
```

The resulting checkpoint is normally `runs/parlour-vision/weights/best.pt`. Point the web app at it with `CUSTOM_MODEL_PATH=runs/parlour-vision/weights/best.pt`; when configured, the web app uses the custom checkpoint as its primary detector and retains MobileNet-SSD as a comparison path.
