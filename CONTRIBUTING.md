# Contributing to OCR Vision Studio

Thanks for contributing. Please open an issue before large changes so the intended behavior and model impact can be discussed.

## Development

Create a virtual environment, install the requirements, install the Tesseract runtime, and run the app with `python app.py`. Run the regression and syntax checks before opening a pull request:

```bash
python3 test_ocr_engine.py
python3 -m py_compile app.py ocr_engine.py train_custom.py test_ocr_engine.py
python3 -m pip check
```

## Pull requests

Keep changes focused, document user-visible behavior in `README.md`, preserve the 80% OCR and 70% object confidence gates unless there is a clear reason to change them, and include tests or reproducible evidence for model changes. Do not commit model weights, uploaded images, secrets, virtual environments, or generated training runs.

## Custom models

Custom YOLO weights should be referenced through `CUSTOM_MODEL_PATH` and should not be committed to the repository. Include the dataset class list and training configuration when sharing reproducible model work.
