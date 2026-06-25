"""Unit tests for ObjectTracker."""

from __future__ import annotations

import numpy as np
import pytest

from tracker import ObjectTracker


@pytest.fixture
def tracker() -> ObjectTracker:
    """Create a tracker instance for testing."""
    return ObjectTracker(max_age=5, min_hits=1, iou_threshold=0.3)


def test_tracker_initialization(tracker: ObjectTracker) -> None:
    """Tracker should initialize with empty history."""
    assert tracker.tracker is not None
    assert tracker.track_history == {}
    assert tracker.track_colors == {}


def test_update_with_empty_detections(tracker: ObjectTracker) -> None:
    """Empty detections should return an empty track list."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = tracker.update([], frame)
    assert isinstance(tracks, list)
    assert len(tracks) == 0


def test_update_with_single_detection(tracker: ObjectTracker) -> None:
    """A single detection should eventually produce a confirmed track."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detection = [100.0, 100.0, 200.0, 200.0, 0.95, 0.0]

    tracks = []
    for _ in range(3):
        tracks = tracker.update([detection], frame)

    assert len(tracks) >= 1
    track = tracks[0]
    assert len(track) == 7
    assert track[4] >= 0  # track_id


def test_get_trajectory(tracker: ObjectTracker) -> None:
    """Trajectory lookup should return a list."""
    trajectory = tracker.get_trajectory(999)
    assert trajectory == []


def test_compute_iou() -> None:
    """IOU computation should handle overlapping boxes."""
    iou = ObjectTracker._compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.0 < iou < 1.0

    identical = ObjectTracker._compute_iou((0, 0, 10, 10), (0, 0, 10, 10))
    assert identical == pytest.approx(1.0)


def test_reset_clears_state(tracker: ObjectTracker) -> None:
    """Reset should clear track history."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detection = [50.0, 50.0, 150.0, 150.0, 0.9, 0.0]
    tracker.update([detection], frame)
    tracker.reset()
    assert tracker.track_history == {}
    assert tracker.track_colors == {}
