"""Utility functions for MariaVision."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def is_streamlit_cloud() -> bool:
    """Return True when running on Streamlit Community Cloud."""
    return os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud"


def get_device(preference: str = "auto") -> str:
    """
    Resolve compute device for inference.

    Args:
        preference: 'auto', 'cpu', or 'cuda'

    Returns:
        'cuda' if available and requested, otherwise 'cpu'
    """
    if is_streamlit_cloud():
        return "cpu"

    if preference == "cpu":
        return "cpu"

    try:
        import torch

        if preference == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def ensure_directories(config: Dict[str, Any]) -> None:
    """Create data directories if they do not exist."""
    paths = config.get("paths", {})
    for key in ("input_dir", "output_dir", "cache_dir"):
        directory = paths.get(key)
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)

    Path("models/weights").mkdir(parents=True, exist_ok=True)
    Path("docs/screenshots").mkdir(parents=True, exist_ok=True)
    Path("docs/demo_video").mkdir(parents=True, exist_ok=True)


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a consistent BGR color for a tracking ID."""
    rng = np.random.default_rng(track_id)
    color = rng.integers(64, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def resize_frame(frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """Resize frame while preserving aspect ratio with letterboxing."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


class FPSCounter:
    """Rolling FPS estimator for real-time display."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.timestamps: List[float] = []

    def tick(self) -> float:
        """Record a frame timestamp and return current FPS."""
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Clear FPS history."""
        self.timestamps.clear()


def draw_fps(frame: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """Draw FPS counter on frame."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def save_frame(frame: np.ndarray, output_dir: str, prefix: str = "frame") -> str:
    """Save an annotated frame to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    output_path = str(Path(output_dir) / filename)
    cv2.imwrite(output_path, frame)
    return output_path


def create_video_writer(
    output_path: str,
    width: int,
    height: int,
    fps: float = 30.0,
) -> cv2.VideoWriter:
    """Create an OpenCV video writer for export."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def format_detection_log(
    class_name: str,
    confidence: float,
    bbox: Tuple[int, int, int, int],
    track_id: Optional[int] = None,
) -> str:
    """Format a detection log entry."""
    x1, y1, x2, y2 = bbox
    base = f"Object detected: {class_name} ({confidence:.2f}) at [{x1}, {y1}, {x2}, {y2}]"
    if track_id is not None:
        base += f" | Track ID: {track_id}"
    return base


def save_uploaded_file(uploaded_file: Any, cache_dir: str) -> str:
    """Persist an uploaded Streamlit file to the cache directory."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(cache_dir) / uploaded_file.name
    with open(output_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return str(output_path)
