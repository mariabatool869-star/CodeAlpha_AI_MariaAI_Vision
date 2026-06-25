"""Unit tests for ObjectDetector."""

from __future__ import annotations

import numpy as np
import pytest

from detector import ObjectDetector


@pytest.fixture(scope="module")
def detector() -> ObjectDetector:
    """Create a lightweight detector for testing."""
    return ObjectDetector(model_name="yolov8n.pt", conf_threshold=0.25, device="cpu")


def test_detector_initialization(detector: ObjectDetector) -> None:
    """Detector should load model and expose COCO class names."""
    assert detector.model is not None
    assert len(detector.classes) > 0
    assert detector.conf_threshold == 0.25


def test_detect_returns_list(detector: ObjectDetector) -> None:
    """Detection on a blank frame should return a list."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)


def test_draw_boxes_does_not_modify_shape(detector: ObjectDetector) -> None:
    """Drawing boxes should preserve frame dimensions."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_detections = [[100, 100, 200, 200, 0.9, 0]]
    annotated = detector.draw_boxes(frame, fake_detections, track_ids=[1])
    assert annotated.shape == frame.shape


def test_update_confidence(detector: ObjectDetector) -> None:
    """Confidence threshold should update within valid range."""
    detector.update_confidence(0.75)
    assert detector.conf_threshold == 0.75

    detector.update_confidence(1.5)
    assert detector.conf_threshold == 1.0

    detector.update_confidence(-0.5)
    assert detector.conf_threshold == 0.0


def test_get_class_name(detector: ObjectDetector) -> None:
    """Class name lookup should return a string."""
    name = detector.get_class_name(0)
    assert isinstance(name, str)
    assert len(name) > 0
