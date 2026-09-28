"""Run carton detection, print predictions, and optionally create an annotated MP4."""
from __future__ import annotations

import argparse
from pathlib import Path


from carton_dimension import CartonDimensionPipeline, DimensionResult, TorchMLPRegressor, UltralyticsCartonDetector


def draw_overlay(frame, pipeline: CartonDimensionPipeline, result: DimensionResult | None) -> None:
    """Draw detection/collection status and the prediction on a BGR frame."""
    import cv2

    bbox = pipeline.last_detection
    if bbox is not None:
        color = (0, 200, 0) if pipeline.collecting else (0, 165, 255)
        cv2.rectangle(frame, (round(bbox.x1), round(bbox.y1)), (round(bbox.x2), round(bbox.y2)), color, 2)
    state = "COLLECTING VALID FRAMES" if pipeline.collecting else "SEARCHING / EXIT"
    cv2.putText(frame, state, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    if result is not None:
        text = f"PREDICTION  L={result.length:.1f} W={result.width:.1f} H={result.height:.1f}"
        cv2.putText(frame, text, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


def format_result(result: DimensionResult) -> str:
    return f"PREDICTION: Length={result.length:.1f}, Width={result.width:.1f}, Height={result.height:.1f}; valid_frames={result.sampled_frames}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--yolo", required=True, help="YOLO weights trained with a carton class")
    parser.add_argument("--mlp", required=True, help="checkpoint produced by train_mlp.py")
    parser.add_argument("--carton-class", type=int)
    parser.add_argument("--output", help="optional annotated MP4 output path")
    args = parser.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    pipeline = CartonDimensionPipeline(UltralyticsCartonDetector(args.yolo, args.carton_class), TorchMLPRegressor(args.mlp))
    writer = None
    if args.output:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
        if not writer.isOpened():
            raise SystemExit(f"Cannot create output video: {args.output}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process(frame)
            if result is not None:
                print(format_result(result))
            draw_overlay(frame, pipeline, result)
            if writer is not None:
                writer.write(frame)
        if result := pipeline.flush():
            print(format_result(result))
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
