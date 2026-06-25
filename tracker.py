"""SORT-style object tracking module for MariaVision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils import get_track_color

Track = List[float]  # [x1, y1, x2, y2, track_id, class_id, confidence]


@dataclass
class _InternalTrack:
    track_id: int
    bbox: Tuple[float, float, float, float]
    class_id: int
    confidence: float
    hits: int = 1
    age: int = 0
    time_since_update: int = 0


class ObjectTracker:
    """
    SORT-inspired tracker using IOU matching and Hungarian assignment.
    Assigns persistent IDs without heavy native dependencies.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.3,
    ) -> None:
        """
        Initialize tracker.

        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections to confirm track
            iou_threshold: IOU threshold for matching
            max_cosine_distance: Reserved for API compatibility
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance

        self._tracks: List[_InternalTrack] = []
        self._next_id = 1
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.track_colors: Dict[int, Tuple[int, int, int]] = {}

    def update(self, detections: List[List[float]], frame: np.ndarray) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of [x1, y1, x2, y2, conf, class_id]
            frame: Current frame (unused, kept for API compatibility)

        Returns:
            List of tracks: [x1, y1, x2, y2, track_id, class_id, confidence]
        """
        del frame  # Frame reserved for future appearance-based matching.

        for track in self._tracks:
            track.age += 1
            track.time_since_update += 1

        if not detections and not self._tracks:
            return []

        det_boxes = [tuple(det[:4]) for det in detections]
        det_meta = [(float(det[4]), int(det[5])) for det in detections]

        matched_tracks: Dict[int, int] = {}
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(self._tracks)))

        if self._tracks and detections:
            iou_matrix = np.zeros((len(self._tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self._tracks):
                for d_idx, box in enumerate(det_boxes):
                    iou_matrix[t_idx, d_idx] = self._compute_iou(track.bbox, box)

            cost_matrix = 1.0 - iou_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for row, col in zip(row_indices, col_indices):
                if iou_matrix[row, col] >= self.iou_threshold:
                    matched_tracks[row] = col
                    unmatched_tracks.discard(row)
                    unmatched_dets.discard(col)

        for track_idx, det_idx in matched_tracks.items():
            track = self._tracks[track_idx]
            confidence, class_id = det_meta[det_idx]
            track.bbox = det_boxes[det_idx]
            track.class_id = class_id
            track.confidence = confidence
            track.hits += 1
            track.time_since_update = 0

        for det_idx in unmatched_dets:
            confidence, class_id = det_meta[det_idx]
            self._tracks.append(
                _InternalTrack(
                    track_id=self._next_id,
                    bbox=det_boxes[det_idx],
                    class_id=class_id,
                    confidence=confidence,
                )
            )
            self._next_id += 1

        self._tracks = [
            track
            for idx, track in enumerate(self._tracks)
            if idx in matched_tracks or track.time_since_update <= self.max_age
        ]

        results: List[Track] = []
        for track in self._tracks:
            if track.hits < self.min_hits or track.time_since_update > 0:
                continue

            x1, y1, x2, y2 = track.bbox
            track_id = track.track_id
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            history = self.track_history.setdefault(track_id, [])
            history.append(center)
            if len(history) > 30:
                history.pop(0)

            if track_id not in self.track_colors:
                self.track_colors[track_id] = get_track_color(track_id)

            results.append(
                [x1, y1, x2, y2, track_id, track.class_id, track.confidence]
            )

        return results

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
        self._tracks.clear()
        self._next_id = 1
        self.track_history.clear()
        self.track_colors.clear()
