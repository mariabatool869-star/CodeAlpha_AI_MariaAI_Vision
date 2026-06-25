"""YOLOv8 object detection module for MariaVision."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from coco_classes import COCO_CLASSES
from utils import get_track_color, is_streamlit_cloud

Detection = List[float]  # [x1, y1, x2, y2, confidence, class_id]

INPUT_SIZE = 640
IOU_THRESHOLD = 0.45


def _resolve_model_path(model_name: str) -> str:
    """Return the first existing model path from common locations."""
    candidates = [
        model_name,
        Path("models/weights") / model_name,
        Path(model_name).name,
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return str(candidate)
    return model_name


def _letterbox(
    image: np.ndarray,
    new_shape: int = INPUT_SIZE,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize image with unchanged aspect ratio using padding."""
    height, width = image.shape[:2]
    scale = min(new_shape / height, new_shape / width)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)

    pad_x = (new_shape - new_width) / 2
    pad_y = (new_shape - new_height) / 2
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas, scale, (pad_x, pad_y)


class ObjectDetector:
    """
    YOLOv8 object detector for real-time inference.
    Uses ONNX Runtime on Streamlit Cloud; Ultralytics optional locally for .pt models.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.onnx",
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.device = device
        self.classes: Dict[int, str] = COCO_CLASSES
        self.backend = "onnx"
        self.model = None
        self.session = None
        self.input_name = ""
        self.output_name = ""

        resolved = _resolve_model_path(model_name)
        use_onnx = model_name.endswith(".onnx") or is_streamlit_cloud()

        if use_onnx:
            onnx_path = resolved if resolved.endswith(".onnx") else _resolve_model_path("yolov8n.onnx")
            self._load_onnx(onnx_path)
        else:
            self._load_ultralytics(resolved)

    def _load_onnx(self, model_path: str) -> None:
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}. "
                "Ensure yolov8n.onnx is committed to the repository."
            )

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.model_name = model_path
        self.backend = "onnx"

    def _load_ultralytics(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed. Use yolov8n.onnx or install ultralytics locally."
            ) from exc

        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.classes = self.model.names
        self.backend = "ultralytics"
        self.model_name = model_path

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a frame."""
        if self.backend == "onnx":
            return self._detect_onnx(frame)
        return self._detect_ultralytics(frame)

    def _detect_onnx(self, frame: np.ndarray) -> List[Detection]:
        letterboxed, scale, (pad_x, pad_y) = _letterbox(frame, INPUT_SIZE)
        blob = letterboxed[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        outputs = self.session.run([self.output_name], {self.input_name: blob})
        return self._postprocess_onnx(outputs[0], frame.shape[:2], scale, pad_x, pad_y)

    def _postprocess_onnx(
        self,
        output: np.ndarray,
        original_shape: Tuple[int, int],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> List[Detection]:
        predictions = np.squeeze(output)
        if predictions.ndim != 2:
            predictions = predictions[0]

        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes_xywh = predictions[:, :4]
        scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]

        mask = confidences >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            IOU_THRESHOLD,
        )

        if len(indices) == 0:
            return []

        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        orig_h, orig_w = original_shape
        detections: List[Detection] = []
        for idx in indices:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            x1 = float(np.clip(x1, 0, orig_w))
            y1 = float(np.clip(y1, 0, orig_h))
            x2 = float(np.clip(x2, 0, orig_w))
            y2 = float(np.clip(y2, 0, orig_h))

            detections.append(
                [x1, y1, x2, y2, float(confidences[idx]), int(class_ids[idx])]
            )

        return detections

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert center xywh boxes to corner xyxy format."""
        converted = np.empty_like(boxes)
        converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return converted

    def _detect_ultralytics(self, frame: np.ndarray) -> List[Detection]:
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
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    [x1, y1, x2, y2, float(box.conf[0].item()), int(box.cls[0].item())]
                )
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
        """Draw bounding boxes with labels and optional tracking IDs."""
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
