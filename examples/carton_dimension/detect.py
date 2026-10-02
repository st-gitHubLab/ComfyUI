"""Write object-detection results for the configured video as JSONL and MP4."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from carton_dimension import BBox, build_detector, serialize_detections
from config import CARTON_CLASS, DETECTION_OUTPUT_VIDEO, DETECTION_RESULTS_JSONL, DETECTOR_KIND, INPUT_VIDEO, YOLO_WEIGHTS



def detection_record(frame_index: int, detections: Sequence[BBox]) -> dict:
    """Serialize configured target detections for one frame."""
    return serialize_detections(frame_index, detections, CARTON_CLASS, "carton_or_demo_vehicle")


def draw_detections(frame, detections: Sequence[BBox]) -> None:
    import cv2

    for box in detections:
        x1, y1, x2, y2 = map(round, (box.x1, box.y1, box.x2, box.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"target {box.confidence:.2f}", (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


def main() -> None:
    import cv2

    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video configured in config.py: {INPUT_VIDEO}")
    detector = build_detector(DETECTOR_KIND, YOLO_WEIGHTS, CARTON_CLASS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    Path(DETECTION_OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)
    Path(DETECTION_RESULTS_JSONL).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(DETECTION_OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Cannot create configured detection video: {DETECTION_OUTPUT_VIDEO}")
    try:
        with Path(DETECTION_RESULTS_JSONL).open("w", encoding="utf-8") as results:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                detections = detector.detect(frame)
                results.write(json.dumps(detection_record(frame_index, detections), ensure_ascii=False) + "\n")
                draw_detections(frame, detections)
                writer.write(frame)
                frame_index += 1
    finally:
        capture.release()
        writer.release()
    print(f"Wrote detection JSONL: {DETECTION_RESULTS_JSONL}")
    print(f"Wrote annotated detection video: {DETECTION_OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
