# Streamlit Cloud Deployment

## Why ONNX?

Streamlit Cloud free tier often **fails to install PyTorch + Ultralytics** (~1GB+).
MariaVision uses **ONNX Runtime** on cloud (~50MB) with a pre-exported `yolov8n.onnx`.

## Required files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit entry point |
| `requirements.txt` | Lightweight cloud deps (no torch) |
| `yolov8n.onnx` | Pre-exported YOLOv8 nano model |
| `.python-version` | Python 3.11 |
| `.streamlit/config.toml` | Server settings |

**Do not add `packages.txt`** unless OpenCV import fails at runtime (then add only `libgl1`).

## Local development with .pt models

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Deploy

1. Push to GitHub (`master` or `main` branch).
2. Connect repo at [share.streamlit.io](https://share.streamlit.io).
3. Set main file: `app.py`.
4. Set branch to match your repo (`master` or `main`).
5. Reboot app after each push.

## Cloud usage

- Use **Upload Video** only (no webcam).
- Model: **yolov8n.onnx** (automatic on cloud).
- First **Start** loads the ONNX model in seconds.
