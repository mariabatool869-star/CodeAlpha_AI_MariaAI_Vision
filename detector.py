"""YOLOv8 object detection module for MariaVision."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from utils import get_track_color


Detection = List[float]  # [x1, y1, x2, y2, confidence, class_id]


class ObjectDetector:
    """
    YOLOv8-based object detector for real-time inference.
    Supports 80+ COCO classes with configurable confidence threshold.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the YOLO detector.

        Args:
            model_name: YOLO model variant (n, s, m, l, x)
            conf_threshold: Minimum confidence for detections
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = YOLO(model_name)
        self.model.to(device)
        self.classes: Dict[int, str] = self.model.names

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input image (BGR format)

        Returns:
            List of detections: [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                detections.append([x1, y1, x2, y2, confidence, class_id])

        return detections

    def draw_boxes(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        track_ids: Optional[List[int]] = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bounding boxes with labels and optional tracking IDs.

        Args:
            frame: Input image
            detections: List of detections
            track_ids: List of tracking IDs aligned with detections
            show_labels: Whether to draw class labels
            show_confidence: Whether to include confidence in labels
            line_thickness: Bounding box line thickness

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for index, detection in enumerate(detections):
            x1, y1, x2, y2, confidence, class_id = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)

            track_id = track_ids[index] if track_ids and index < len(track_ids) else None
            color = get_track_color(track_id if track_id is not None else class_id)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

            if show_labels:
                class_name = self.classes.get(class_id, str(class_id))
                label_parts = [class_name]
                if show_confidence:
                    label_parts.append(f"{confidence:.2f}")
                if track_id is not None:
                    label_parts.append(f"ID:{track_id}")

                label = " ".join(label_parts)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(y1, label_size[1] + 8)
                cv2.rectangle(
                    annotated,
                    (x1, label_y - label_size[1] - 8),
                    (x1 + label_size[0] + 4, label_y + 4),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return annotated

    def update_confidence(self, threshold: float) -> None:
        """Update confidence threshold."""
        self.conf_threshold = max(0.0, min(1.0, threshold))

    def get_class_name(self, class_id: int) -> str:
        """Return human-readable class name."""
        return self.classes.get(class_id, str(class_id))
