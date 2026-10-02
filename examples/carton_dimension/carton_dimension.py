"""Video carton dimension estimation pipeline.

A carton is sampled only while its detection is fully inside the image.  When it
first touches an image boundary, the collected bounding-box features are
aggregated and fed into a trained MLP regressor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class BBox:
    """Pixel coordinates of one detection, in xyxy order."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    def clipped(self, width: int, height: int) -> "BBox":
        return BBox(max(0, self.x1), max(0, self.y1), min(width, self.x2), min(height, self.y2), self.confidence)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def fully_inside(self, frame_width: int, frame_height: int, margin: int) -> bool:
        """Return true if every side stays away from the image border."""
        return (
            self.x1 >= margin
            and self.y1 >= margin
            and self.x2 <= frame_width - margin
            and self.y2 <= frame_height - margin
        )


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> Sequence[BBox]: ...


class Regressor(Protocol):
    def predict(self, features: np.ndarray) -> tuple[float, float, float]: ...


@dataclass(frozen=True)
class DimensionResult:
    length: float
    width: float
    height: float
    sampled_frames: int


@dataclass
class CartonDimensionPipeline:
    detector: Detector
    regressor: Regressor
    border_margin: int = 2
    min_samples: int = 8
    _features: list[tuple[float, float, float]] = field(default_factory=list, init=False)
    _collecting: bool = field(default=False, init=False)
    last_detection: BBox | None = field(default=None, init=False)

    @property
    def collecting(self) -> bool:
        """Whether the current carton is contributing valid feature frames."""
        return self._collecting

    def process(self, frame: np.ndarray) -> DimensionResult | None:
        """Process one BGR/RGB frame and return a result when sampling ends."""
        height, width = frame.shape[:2]
        bbox = self._best_detection(self.detector.detect(frame))
        self.last_detection = bbox
        fully_inside = bbox is not None and bbox.fully_inside(width, height, self.border_margin)

        if not self._collecting:
            if fully_inside:
                self._collecting = True
                self._features = []
                self._append_feature(bbox, width, height)
            return None

        if fully_inside:
            self._append_feature(bbox, width, height)
            return None

        # A missing detection or a box touching the border is the exit event.
        result = self._finish()
        self._collecting = False
        self._features = []
        return result

    def flush(self) -> DimensionResult | None:
        """Finish the active carton at end-of-video and return its prediction."""
        if not self._collecting:
            return None
        result = self._finish()
        self._collecting = False
        self._features = []
        return result

    def _best_detection(self, detections: Sequence[BBox]) -> BBox | None:
        return max(detections, key=lambda box: box.confidence, default=None)

    def _append_feature(self, bbox: BBox, frame_width: int, frame_height: int) -> None:
        # Normalize to make model training independent of the camera resolution.
        self._features.append((bbox.width / frame_width, bbox.height / frame_height, bbox.area / (frame_width * frame_height)))

    def _finish(self) -> DimensionResult | None:
        if len(self._features) < self.min_samples:
            return None
        values = np.asarray(self._features, dtype=np.float32)
        # mean, standard deviation, max for width, height and area => 9 inputs.
        features = np.concatenate((values.mean(axis=0), values.std(axis=0), values.max(axis=0)))
        length, width, height = self.regressor.predict(features)
        return DimensionResult(float(length), float(width), float(height), len(values))


class UltralyticsCartonDetector:
    """Lazy Ultralytics adapter; import is delayed so tests need no YOLO install."""

    def __init__(self, weights: str, carton_class: int | None = None, confidence: float = 0.5):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.carton_class = carton_class
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> Sequence[BBox]:
        classes = None if self.carton_class is None else [self.carton_class]
        result = self.model.predict(frame, conf=self.confidence, classes=classes, verbose=False)[0]
        return [BBox(*box.xyxy[0].tolist(), float(box.conf[0])) for box in result.boxes]


class TorchMLPRegressor:
    def __init__(self, checkpoint: str | Path):
        import torch

        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.mean = data["feature_mean"].numpy().astype(np.float32)
        self.std = data["feature_std"].numpy().astype(np.float32)
        self.target_scale = data["target_scale"].numpy().astype(np.float32)
        self.model = torch.nn.Sequential(torch.nn.Linear(9, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
        self.model.load_state_dict(data["model"])
        self.model.eval()

    def predict(self, features: np.ndarray) -> tuple[float, float, float]:
        import torch

        normalized = (features - self.mean) / np.maximum(self.std, 1e-6)
        with torch.inference_mode():
            output = self.model(torch.tensor(normalized[None], dtype=torch.float32)).numpy()[0] * self.target_scale
        return tuple(output.tolist())


def extract_video_features(
    video_path: str | Path,
    detector: Detector,
    border_margin: int = 2,
    min_samples: int = 8,
) -> np.ndarray | None:
    """Extract one 9-value feature vector from a labeled training video.

    The selection logic is intentionally identical to inference: use the
    highest-confidence carton, collect only while fully inside the frame, and
    complete the track as soon as it touches a border (or reaches EOF).
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open training video: {video_path}")
    feature_frames: list[tuple[float, float, float]] = []
    collecting = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_height, frame_width = frame.shape[:2]
            bbox = max(detector.detect(frame), key=lambda box: box.confidence, default=None)
            inside = bbox is not None and bbox.fully_inside(frame_width, frame_height, border_margin)
            if inside:
                collecting = True
                feature_frames.append((bbox.width / frame_width, bbox.height / frame_height, bbox.area / (frame_width * frame_height)))
            elif collecting:
                break
    finally:
        capture.release()

    if len(feature_frames) < min_samples:
        return None
    values = np.asarray(feature_frames, dtype=np.float32)
    return np.concatenate((values.mean(axis=0), values.std(axis=0), values.max(axis=0)))


class OrangeDemoDetector:
    """Detect the orange synthetic vehicles emitted by generate_demo_data.py."""

    def detect(self, frame: np.ndarray) -> Sequence[BBox]:
        import cv2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((5, 120, 120)), np.array((25, 255, 255)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width * height >= 100:
                boxes.append(BBox(x, y, x + width, y + height, 1.0))
        return boxes


def build_detector(kind: str, yolo_weights: str | Path, carton_class: int | None) -> Detector:
    """Create the configured production YOLO detector or deterministic demo detector."""
    if kind == "yolo":
        return UltralyticsCartonDetector(str(yolo_weights), carton_class)
    if kind == "orange_demo":
        return OrangeDemoDetector()
    raise ValueError(f"Unsupported DETECTOR_KIND: {kind}")
