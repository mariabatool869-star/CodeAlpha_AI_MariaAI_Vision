<div align="center">

# MariaVision

### Real-Time Object Detection & Multi-Object Tracking

**YOLOv8 · Deep SORT · OpenCV · Streamlit**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://codealphaaimariaaivision-n3eprjsxdgrkhcqd822xxj.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

*Portfolio project for the CodeAlpha AI Internship — Task 4: Computer Vision & Deep Learning*

[Live Demo](https://codealphaaimariaaivision-n3eprjsxdgrkhcqd822xxj.streamlit.app/) · [Features](#features) · [Installation](#installation) · [Documentation](#usage)

</div>

---

## Overview

**MariaVision** is an end-to-end computer vision application that detects objects in video streams and assigns persistent tracking IDs across frames. Built for real-world scenarios such as crowd monitoring, traffic analysis, and surveillance analytics, it combines state-of-the-art detection with robust multi-object tracking in an interactive web interface.

| Capability | Technology |
|------------|------------|
| Object Detection | YOLOv8 (80+ COCO classes) |
| Multi-Object Tracking | Deep SORT |
| Video I/O | OpenCV |
| User Interface | Streamlit |
| Configuration | YAML-based settings |

---

## Demo

### Detection & Tracking in Action

<p align="center">
  <img src="docs/screenshots/detection_tracking_demo.jpg" alt="MariaVision detecting and tracking pedestrians in a crowded crosswalk scene" width="90%">
</p>

<p align="center"><em>Multi-person detection on a busy crosswalk — bounding boxes, class labels, confidence scores, and unique Deep SORT tracking IDs.</em></p>

**What this screenshot shows:**

| Element | Description |
|---------|-------------|
| Bounding boxes | Color-coded per tracked object |
| Labels | Class name + confidence (e.g. `person 0.61`) |
| Tracking IDs | Persistent IDs (e.g. `ID:96`, `ID:124`) across frames |
| FPS counter | Real-time performance overlay (top-left) |
| Crowd handling | Multiple overlapping detections in dense scenes |

> **Try it live:** [codealphaaimariaaivision-n3eprjsxdgrkhcqd822xxj.streamlit.app](https://codealphaaimariaaivision-n3eprjsxdgrkhcqd822xxj.streamlit.app/) — upload a video and click **Start**.

---

## Features

### Detection
- 80+ object classes from the COCO dataset
- Configurable confidence threshold (0.1 – 1.0)
- Multiple YOLOv8 model sizes (`n`, `s`, `m`, `l`)
- Automatic NMS and GPU/CPU device selection

### Tracking
- Unique ID assignment with Deep SORT
- Occlusion handling and re-identification
- Optional trajectory path visualization
- Per-track color coding for visual clarity

### Interface
- Webcam and video file input (`mp4`, `avi`, `mov`, `mkv`)
- Start · Stop · Pause · Resume controls
- Live statistics panel (FPS, object counts, active tracks)
- Detection log with timestamped events
- Save annotated frames and export processed video

### Engineering
- Modular architecture (`detector`, `tracker`, `utils`)
- YAML configuration file
- Unit tests with pytest
- Docker and Streamlit Cloud deployment support

---

## Tech Stack

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ Video Input │───▶│  YOLOv8      │───▶│  Deep SORT  │───▶│  Streamlit   │
│ Webcam/File │    │  Detection   │    │  Tracking   │    │  Dashboard   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

| Layer | Tools |
|-------|-------|
| Detection | Ultralytics YOLOv8, PyTorch |
| Tracking | deep-sort-realtime, filterpy, scipy |
| Vision | OpenCV, NumPy |
| Frontend | Streamlit |
| Config | PyYAML |

---

## Installation

### Prerequisites

- Python 3.9+ (3.11 recommended for deployment)
- Webcam (optional — local use only)
- NVIDIA GPU with CUDA (optional — faster inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/MariaAI-Vision.git
cd MariaAI-Vision

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

YOLOv8 weights are downloaded automatically on first run.

### Run Locally

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. **Select a source** — Webcam (local) or upload a video file.
2. **Choose a model** — `yolov8n.pt` for speed, `yolov8s.pt` or larger for accuracy.
3. **Set confidence** — Lower values detect more objects; higher values reduce false positives.
4. **Configure display** — Toggle labels, tracking IDs, and trajectories.
5. **Click Start** — Processing begins with live stats and a detection log.
6. **Export results** — Save individual frames or enable video export before starting.

### Recommended Settings

| Scenario | Model | Confidence | Notes |
|----------|-------|------------|-------|
| Real-time webcam | `yolov8n.pt` | 0.4 – 0.5 | Best FPS on CPU |
| Crowd / street video | `yolov8s.pt` | 0.5 – 0.6 | Better accuracy |
| Sparse scenes | `yolov8n.pt` | 0.6+ | Fewer false positives |

---

## Project Structure

```
MariaAI-Vision/
├── app.py                      # Streamlit UI and VideoProcessor pipeline
├── detector.py                 # YOLOv8 object detection module
├── tracker.py                  # Deep SORT tracking module
├── utils.py                    # Config, FPS, I/O helpers
├── config.yaml                 # Default application settings
├── requirements.txt            # Python dependencies
├── packages.txt                # System packages (Streamlit Cloud)
├── Dockerfile                  # Container deployment
├── data/
│   ├── input/                  # Uploaded media
│   ├── output/                 # Saved frames and exports
│   └── cache/                  # Temporary uploads
├── docs/
│   └── screenshots/            # README and portfolio assets
├── models/weights/             # Optional local model weights
└── tests/                      # Unit tests (detector, tracker)
```

---

## Configuration

Edit `config.yaml` to change defaults without modifying code:

```yaml
model:
  name: yolov8n.pt
  confidence: 0.5
  device: auto          # auto | cpu | cuda

tracker:
  max_age: 30           # Frames before a lost track is removed
  min_hits: 3           # Detections needed to confirm a track
  iou_threshold: 0.3

display:
  show_labels: true
  show_tracking: true
  show_trajectory: false
```

---

## Testing

```bash
python -m pytest tests/ -v
```

| Module | Tests |
|--------|-------|
| `test_detector.py` | Model init, detection, drawing, confidence |
| `test_tracker.py` | Tracking updates, IOU, reset, trajectories |

---

## Performance

Approximate frame rates (640×480 input):

| Hardware | Model | FPS |
|----------|-------|-----|
| CPU (Intel i7) | YOLOv8n | 15 – 20 |
| CPU (Intel i7) | YOLOv8s | 8 – 12 |
| GPU (RTX 3060) | YOLOv8n | 45 – 60 |
| GPU (RTX 3060) | YOLOv8s | 30 – 40 |

*Dense crowd scenes may run slower due to increased detection and tracking load.*

---

## Deployment

### Streamlit Cloud

**Live deployment:** https://codealphaaimariaaivision-n3eprjsxdgrkhcqd822xxj.streamlit.app/

1. Push the repository to GitHub.
2. Connect the repo at [share.streamlit.io](https://share.streamlit.io).
3. Set `app.py` as the main file.
4. Ensure `.python-version` (3.11), `packages.txt`, and `opencv-python-headless` are included.

> **Note:** Webcam capture is not available on Streamlit Cloud. Use **Upload Video** for cloud demos.

### Docker

```bash
docker build -t mariavision .
docker run -p 8501:8501 mariavision
```

---

## Technical Details

**YOLOv8** — Single-shot detector with 640×640 input, built-in NMS, and 80 COCO classes. Supports nano through extra-large model variants.

**Deep SORT** — Combines Kalman filter motion prediction, Hungarian algorithm assignment, and MobileNet appearance embeddings for stable IDs across occlusions.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Maria** — AI Internship Portfolio · CodeAlpha · 2026

---

<div align="center">

**MariaVision** — Built for CodeAlpha AI Internship Task 4

[Live Demo](https://codealphaaimariaaivision-n3eprjsxdgrkhcqd822xxj.streamlit.app/) · Computer Vision · Deep Learning · Real-Time Systems

</div>
