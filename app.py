# LINE 1: This is where the code goes!
import subprocess
import sys

def install_missing_packages():
    packages = [
        "scipy==1.11.4",
        "filterpy==1.4.5", 
        "lap==0.4.0",
        "PyYAML==6.0.1",
        "pillow==10.1.0"
    ]
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except:
            pass

install_missing_packages()

# LINE 15-ish: Now your regular imports
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# ... rest of your code continues
"""Main application controller and Streamlit UI for MariaVision."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

from detector import ObjectDetector
from tracker import ObjectTracker
from utils import (
    FPSCounter,
    create_video_writer,
    draw_fps,
    ensure_directories,
    format_detection_log,
    get_device,
    load_config,
    save_frame,
    save_uploaded_file,
)


class VideoProcessor:
    """
    Main application controller managing video processing pipeline.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or load_config()
        ensure_directories(self.config)

        model_cfg = self.config["model"]
        tracker_cfg = self.config["tracker"]

        device = get_device(model_cfg.get("device", "auto"))
        self.detector = ObjectDetector(
            model_name=model_cfg.get("name", "yolov8n.pt"),
            conf_threshold=model_cfg.get("confidence", 0.5),
            device=device,
        )
        self.tracker = ObjectTracker(
            max_age=tracker_cfg.get("max_age", 30),
            min_hits=tracker_cfg.get("min_hits", 3),
            iou_threshold=tracker_cfg.get("iou_threshold", 0.3),
            max_cosine_distance=tracker_cfg.get("max_cosine_distance", 0.3),
        )

        self.is_running = False
        self.is_paused = False
        self.current_frame: Optional[np.ndarray] = None
        self.fps_counter = FPSCounter()
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.export_path: Optional[str] = None

        self.stats: Dict[str, Any] = {
            "fps": 0.0,
            "total_objects": 0,
            "active_tracks": 0,
            "object_counts": {},
        }
        self.detection_log: List[str] = []

        display_cfg = self.config.get("display", {})
        self.show_labels = display_cfg.get("show_labels", True)
        self.show_tracking = display_cfg.get("show_tracking", True)
        self.show_trajectory = display_cfg.get("show_trajectory", False)
        self.show_confidence = display_cfg.get("show_confidence", True)
        self.line_thickness = display_cfg.get("line_thickness", 2)

    def configure_model(self, model_name: str, confidence: float) -> None:
        """Reinitialize detector with new model settings."""
        device = get_device(self.config["model"].get("device", "auto"))
        self.detector = ObjectDetector(
            model_name=model_name,
            conf_threshold=confidence,
            device=device,
        )

    def update_display_options(
        self,
        show_labels: bool,
        show_tracking: bool,
        show_trajectory: bool,
    ) -> None:
        """Update visualization toggles."""
        self.show_labels = show_labels
        self.show_tracking = show_tracking
        self.show_trajectory = show_trajectory

    def start(self, source: Any = 0) -> cv2.VideoCapture:
        """Start video processing from webcam or file."""
        self.is_running = True
        self.is_paused = False
        self.fps_counter.reset()
        self.tracker.reset()
        self.detection_log.clear()
        return cv2.VideoCapture(source)

    def stop(self) -> None:
        """Stop processing and release resources."""
        self.is_running = False
        self.is_paused = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def pause(self) -> None:
        """Pause processing."""
        self.is_paused = True

    def resume(self) -> None:
        """Resume processing."""
        self.is_paused = False

    def begin_export(self, output_path: str, width: int, height: int, fps: float) -> str:
        """Start exporting processed video."""
        self.export_path = output_path
        self.video_writer = create_video_writer(output_path, width, height, fps)
        return output_path

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame through detection and tracking pipeline."""
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)

        annotated = frame.copy()

        if self.show_trajectory:
            annotated = self.tracker.draw_all_trajectories(annotated)

        track_detections: List[List[float]] = []
        track_ids: List[int] = []

        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, confidence = track
            track_detections.append([x1, y1, x2, y2, confidence, class_id])
            track_ids.append(int(track_id))

            class_name = self.detector.get_class_name(int(class_id))
            self.detection_log.append(
                format_detection_log(
                    class_name,
                    confidence,
                    (int(x1), int(y1), int(x2), int(y2)),
                    int(track_id) if self.show_tracking else None,
                )
            )

        if track_detections:
            annotated = self.detector.draw_boxes(
                annotated,
                track_detections,
                track_ids=track_ids if self.show_tracking else None,
                show_labels=self.show_labels,
                show_confidence=self.show_confidence,
                line_thickness=self.line_thickness,
            )
        elif detections and not self.show_tracking:
            annotated = self.detector.draw_boxes(
                annotated,
                detections,
                show_labels=self.show_labels,
                show_confidence=self.show_confidence,
                line_thickness=self.line_thickness,
            )

        fps = self.fps_counter.tick()
        annotated = draw_fps(annotated, fps)

        object_counts = Counter(
            self.detector.get_class_name(int(track[5])) for track in tracks
        )
        if not tracks:
            object_counts = Counter(
                self.detector.get_class_name(int(det[5])) for det in detections
            )

        self.stats = {
            "fps": fps,
            "total_objects": len(detections),
            "active_tracks": len(tracks),
            "object_counts": dict(object_counts),
        }

        self.current_frame = annotated

        if self.video_writer is not None:
            self.video_writer.write(annotated)

        if len(self.detection_log) > 100:
            self.detection_log = self.detection_log[-100:]

        return annotated

    def save_current_frame(self) -> Optional[str]:
        """Save current frame with annotations."""
        if self.current_frame is None:
            return None
        output_dir = self.config["paths"]["output_dir"]
        return save_frame(self.current_frame, output_dir)

    def export_video(self, output_path: Optional[str] = None) -> Optional[str]:
        """Finalize exported video."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        return output_path or self.export_path


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR frame to RGB for Streamlit."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def run_processing_loop(
    processor: VideoProcessor,
    cap: cv2.VideoCapture,
    frame_placeholder: Any,
    stats_placeholder: Any,
    log_placeholder: Any,
    export_video: bool = False,
) -> None:
    """Process video frames until stopped or stream ends."""
    output_dir = processor.config["paths"]["output_dir"]
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    export_path = None
    if export_video:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_path = str(Path(output_dir) / f"mariavision_export_{timestamp}.mp4")
        processor.begin_export(export_path, width, height, fps)

    progress_bar = st.progress(0.0, text="Processing video...")
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_index = 0

    while processor.is_running and cap.isOpened():
        if processor.is_paused:
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        annotated = processor.process_frame(frame)
        frame_placeholder.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)
        update_stats_panel(stats_placeholder, processor.stats)
        update_log_panel(log_placeholder, processor.detection_log)

        frame_index += 1
        if frame_total > 0:
            progress_bar.progress(min(frame_index / frame_total, 1.0), text="Processing video...")

    progress_bar.empty()
    cap.release()
    processor.stop()

    if export_path:
        processor.export_video(export_path)
        st.success(f"Video exported to `{export_path}`")


def update_stats_panel(placeholder: Any, stats: Dict[str, Any]) -> None:
    """Render live statistics in the sidebar/main panel."""
    with placeholder.container():
        st.metric("FPS", f"{stats.get('fps', 0):.1f}")
        st.metric("Total Objects", stats.get("total_objects", 0))
        st.metric("Active Tracks", stats.get("active_tracks", 0))

        st.write("**Detected Objects:**")
        object_counts = stats.get("object_counts", {})
        if object_counts:
            for name, count in object_counts.items():
                st.write(f"- {name}: {count}")
        else:
            st.write("- No objects detected")


def update_log_panel(placeholder: Any, logs: List[str]) -> None:
    """Render detection log."""
    with placeholder.container():
        if logs:
            for entry in reversed(logs[-15:]):
                st.text(entry)
        else:
            st.text("Waiting for detections...")


def main() -> None:
    """Streamlit application entry point."""
    st.set_page_config(
        page_title="MariaVision - Object Detection & Tracking",
        page_icon="👁️",
        layout="wide",
    )

    st.title("👁️ MariaVision")
    st.caption("Real-time object detection and tracking with YOLOv8 and Deep SORT")

    if "processor" not in st.session_state:
        st.session_state.processor = VideoProcessor()
    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    processor: VideoProcessor = st.session_state.processor

    with st.sidebar:
        st.title("Controls")
        st.markdown("---")

        source_type = st.radio("Source", ["Webcam", "Upload Video"], horizontal=True)
        uploaded_file = None
        webcam_index = 0

        if source_type == "Webcam":
            webcam_index = st.number_input("Webcam Index", min_value=0, max_value=5, value=0, step=1)
        else:
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

        st.markdown("---")
        model_options = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
        selected_model = st.selectbox("Model", model_options, index=0)
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

        processor.configure_model(selected_model, confidence)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("Start", use_container_width=True, type="primary")
        with col2:
            stop_btn = st.button("Stop", use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            pause_btn = st.button("Pause", use_container_width=True)
        with col4:
            resume_btn = st.button("Resume", use_container_width=True)

        show_labels = st.checkbox("Show Labels", value=True)
        show_tracking = st.checkbox("Show Tracking IDs", value=True)
        show_trajectory = st.checkbox("Show Trajectory", value=False)
        processor.update_display_options(show_labels, show_tracking, show_trajectory)

        st.markdown("---")
        save_frame_btn = st.button("Save Frame", use_container_width=True)
        export_while_processing = st.checkbox("Export processed video", value=False)

        device = get_device(processor.config["model"].get("device", "auto"))
        st.info(f"Device: **{device.upper()}**")

    col_main, col_stats = st.columns([3, 1])

    with col_main:
        st.subheader("Video Feed")
        frame_placeholder = st.empty()
        frame_placeholder.info("Click **Start** to begin detection and tracking.")

    with col_stats:
        st.subheader("Live Statistics")
        stats_placeholder = st.empty()
        update_stats_panel(stats_placeholder, processor.stats)

    with st.expander("Detection Log", expanded=False):
        log_placeholder = st.empty()
        update_log_panel(log_placeholder, processor.detection_log)

    if save_frame_btn:
        saved_path = processor.save_current_frame()
        if saved_path:
            st.sidebar.success(f"Frame saved to `{saved_path}`")
        else:
            st.sidebar.warning("No frame available to save. Start processing first.")

    if stop_btn:
        st.session_state.is_running = False
        processor.stop()

    if pause_btn:
        processor.pause()

    if resume_btn:
        processor.resume()

    if start_btn:
        source: Any
        if source_type == "Upload Video":
            if uploaded_file is None:
                st.error("Please upload a video file first.")
                return
            cache_dir = processor.config["paths"]["cache_dir"]
            source = save_uploaded_file(uploaded_file, cache_dir)
        else:
            source = int(webcam_index)

        processor.start(source)
        st.session_state.is_running = True
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            st.error("Unable to open video source. Check webcam connection or uploaded file.")
            processor.stop()
            return

        run_processing_loop(
            processor=processor,
            cap=cap,
            frame_placeholder=frame_placeholder,
            stats_placeholder=stats_placeholder,
            log_placeholder=log_placeholder,
            export_video=export_while_processing,
        )


if __name__ == "__main__":
    main()
