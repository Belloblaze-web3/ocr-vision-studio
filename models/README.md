# MobileNet-SSD model assets

The Option 2 object-detection path uses the original MobileNet-SSD Caffe model and the 21-class VOC label set. The application downloads `deploy.prototxt` and `mobilenet_iter_73000.caffemodel` automatically on the first object-detection request, then caches them in this directory.

Source model: https://github.com/chuanqi305/MobileNet-SSD

The model is loaded through `cv2.dnn.readNetFromCaffe`, resized to a 300×300 blob, and filtered at 50% confidence before boxes are rendered.
