"""Run carton detection and print Length/Width/Height in the training unit."""
import argparse
import cv2
from carton_dimension import CartonDimensionPipeline, TorchMLPRegressor, UltralyticsCartonDetector

parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("--yolo", required=True, help="YOLO weights trained with a carton class")
parser.add_argument("--mlp", required=True, help="checkpoint produced by train_mlp.py")
parser.add_argument("--carton-class", type=int)
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {args.video}")
pipeline = CartonDimensionPipeline(UltralyticsCartonDetector(args.yolo, args.carton_class), TorchMLPRegressor(args.mlp))
while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = pipeline.process(frame)
    if result:
        print(f"Length={result.length:.1f}, Width={result.width:.1f}, Height={result.height:.1f}; frames={result.sampled_frames}")
cap.release()
