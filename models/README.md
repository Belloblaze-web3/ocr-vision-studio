# Pretrained model assets

The Option 2 object-detection path uses YOLOv8n COCO as its primary detector and the original MobileNet-SSD Caffe model as a secondary comparison. Ultralytics downloads `yolov8n.pt` on first use. The application downloads `deploy.prototxt` and `mobilenet_iter_73000.caffemodel` automatically on first comparison use, then caches them in this directory.

Source model: https://github.com/chuanqi305/MobileNet-SSD

MobileNet-SSD is loaded through `cv2.dnn.readNetFromCaffe`, resized to a 300×300 blob, filtered at 70% confidence, and passed through non-maximum suppression before boxes are rendered. The VOC taxonomy does not include DVD players, receivers, speakers, sockets, or woofers. For those classes, train the custom YOLO checkpoint described in `datasets/README.md` and set `CUSTOM_MODEL_PATH`.
