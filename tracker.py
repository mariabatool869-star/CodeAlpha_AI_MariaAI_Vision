"""Deep SORT tracking module for MariaVision."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from utils import get_track_color

Track = List[float]  # [x1, y1, x2, y2, track_id, class_id, confidence]


class ObjectTracker:
    """
    Deep SORT tracker for multi-object tracking.
    Handles occlusions and re-identification.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.3,
    ) -> None:
        """
        Initialize Deep SORT tracker.

        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections to confirm track
            iou_threshold: IOU threshold for matching
            max_cosine_distance: Appearance embedding distance threshold
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=min_hits,
            max_iou_distance=1.0 - iou_threshold,
            max_cosine_distance=max_cosine_distance,
            embedder="mobilenet",
            half=True,
            bgr=True,
        )
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.track_colors: Dict[int, Tuple[int, int, int]] = {}
        self._last_class_map: Dict[int, int] = {}
        self._last_conf_map: Dict[int, float] = {}

    def update(self, detections: List[List[float]], frame: np.ndarray) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of [x1, y1, x2, y2, conf, class_id]
            frame: Current frame

        Returns:
            List of tracks: [x1, y1, x2, y2, track_id, class_id, confidence]
        """
        formatted = []
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            class_name = str(int(class_id))
            formatted.append(([x1, y1, width, height], confidence, class_name))

        tracks = self.tracker.update_tracks(formatted, frame=frame)
        results: List[Track] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            left, top, right, bottom = map(int, track.to_ltrb())

            class_id = self._resolve_class_id(track, detections)
            confidence = self._resolve_confidence(track_id, detections, left, top, right, bottom)

            self._last_class_map[track_id] = class_id
            self._last_conf_map[track_id] = confidence

            center = ((left + right) // 2, (bottom + top) // 2)
            history = self.track_history.setdefault(track_id, [])
            history.append(center)
            if len(history) > 30:
                history.pop(0)

            if track_id not in self.track_colors:
                self.track_colors[track_id] = get_track_color(track_id)

            results.append([left, top, right, bottom, track_id, class_id, confidence])

        return results

    def _resolve_class_id(self, track, detections: List[List[float]]) -> int:
        """Match a track to the closest detection for class ID."""
        left, top, right, bottom = map(int, track.to_ltrb())
        best_class = self._last_class_map.get(int(track.track_id), 0)
        best_iou = 0.0

        for detection in detections:
            x1, y1, x2, y2, _, class_id = detection
            iou = self._compute_iou((left, top, right, bottom), (x1, y1, x2, y2))
            if iou > best_iou:
                best_iou = iou
                best_class = int(class_id)

        return best_class

    def _resolve_confidence(
        self,
        track_id: int,
        detections: List[List[float]],
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> float:
        """Match a track to the closest detection for confidence score."""
        best_conf = self._last_conf_map.get(track_id, 0.0)
        best_iou = 0.0

        for detection in detections:
            x1, y1, x2, y2, confidence, _ = detection
            iou = self._compute_iou((left, top, right, bottom), (x1, y1, x2, y2))
            if iou > best_iou:
                best_iou = iou
                best_conf = float(confidence)

        return best_conf

    @staticmethod
    def _compute_iou(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        """Compute intersection over union for two boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """Get historical positions for a track."""
        return self.track_history.get(track_id, [])

    def draw_trajectory(
        self,
        frame: np.ndarray,
        track_id: int,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Draw trajectory path for a tracked object."""
        points = self.get_trajectory(track_id)
        if len(points) < 2:
            return frame

        draw_color = color or self.track_colors.get(track_id, get_track_color(track_id))
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(frame, start, end, draw_color, 2)
        return frame

    def draw_all_trajectories(self, frame: np.ndarray) -> np.ndarray:
        """Draw trajectories for all active tracks."""
        annotated = frame.copy()
        for track_id in self.track_history:
            self.draw_trajectory(annotated, track_id)
        return annotated

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.min_hits,
            max_iou_distance=1.0 - self.iou_threshold,
            embedder="mobilenet",
            half=True,
            bgr=True,
        )
        self.track_history.clear()
        self.track_colors.clear()
        self._last_class_map.clear()
        self._last_conf_map.clear()
