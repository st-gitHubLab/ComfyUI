"""Run prediction for the input/output video paths configured in config.py."""
from __future__ import annotations

import json
from pathlib import Path

from carton_dimension import CartonDimensionPipeline, CartonInspectionPipeline, DimensionResult, ForeignObjectDetector, InspectionResult, TorchMLPRegressor, build_detector, serialize_inspection
from config import BORDER_MARGIN, CARTON_CLASS, DETECTOR_KIND, HULL_DIFF_THRESHOLD, INPUT_VIDEO, INSPECTION_RESULTS_JSONL, MIN_SAMPLES, MLP_CHECKPOINT, OUTPUT_VIDEO, YOLO_WEIGHTS


def draw_overlay(frame, pipeline: CartonDimensionPipeline, result: InspectionResult) -> None:
    import cv2

    bbox = pipeline.last_detection
    if bbox is not None:
        color = (0, 200, 0) if pipeline.collecting else (0, 165, 255)
        cv2.rectangle(frame, (round(bbox.x1), round(bbox.y1)), (round(bbox.x2), round(bbox.y2)), color, 2)
    state = "COLLECTING VALID FRAMES" if pipeline.collecting else "SEARCHING / EXIT"
    cv2.putText(frame, state, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    if result.foreign_object is not None:
        foreign = result.foreign_object
        label = "FOREIGN OBJECT" if foreign.is_abnormal else "FOREIGN OBJECT: NORMAL"
        color = (0, 0, 255) if foreign.is_abnormal else (0, 255, 0)
        cv2.putText(frame, f"{label} ratio={foreign.hull_diff_ratio:.3f}", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if result.dimension is not None:
        dimension = result.dimension
        text = f"PREDICTION  L={dimension.length:.1f} W={dimension.width:.1f} H={dimension.height:.1f}"
        cv2.putText(frame, text, (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


def format_result(result: DimensionResult) -> str:
    return f"PREDICTION: Length={result.length:.1f}, Width={result.width:.1f}, Height={result.height:.1f}; valid_frames={result.sampled_frames}"


def main() -> None:
    import cv2

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video configured in config.py: {INPUT_VIDEO}")
    pipeline = CartonDimensionPipeline(
        build_detector(DETECTOR_KIND, YOLO_WEIGHTS, CARTON_CLASS),
        TorchMLPRegressor(MLP_CHECKPOINT),
        border_margin=BORDER_MARGIN,
        min_samples=MIN_SAMPLES,
    )
    inspection_pipeline = CartonInspectionPipeline(pipeline, ForeignObjectDetector(HULL_DIFF_THRESHOLD))
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    Path(OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)
    Path(INSPECTION_RESULTS_JSONL).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise SystemExit(f"Cannot create output video configured in config.py: {OUTPUT_VIDEO}")
    try:
        with Path(INSPECTION_RESULTS_JSONL).open("w", encoding="utf-8") as inspection_file:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = inspection_pipeline.process(frame)
                inspection_file.write(json.dumps(serialize_inspection(result)) + "\n")
                if result.dimension is not None:
                    print(format_result(result.dimension))
                if result.foreign_object is not None and result.foreign_object.is_abnormal:
                    print(f"FOREIGN OBJECT: hull_diff_ratio={result.foreign_object.hull_diff_ratio:.3f}")
                draw_overlay(frame, pipeline, result)
                writer.write(frame)
            if result := inspection_pipeline.flush():
                inspection_file.write(json.dumps(serialize_inspection(result)) + "\n")
                if result.dimension is not None:
                    print(format_result(result.dimension))
    finally:
        cap.release()
        writer.release()


if __name__ == "__main__":
    main()
