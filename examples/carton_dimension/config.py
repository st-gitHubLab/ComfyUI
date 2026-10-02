"""Edit this file once to configure the carton dimension example."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Model files. Train first to create MLP_CHECKPOINT; put your YOLO carton model here.
YOLO_WEIGHTS = PROJECT_DIR / "models" / "carton_yolo.pt"
MLP_CHECKPOINT = PROJECT_DIR / "models" / "carton_mlp.pt"
CARTON_CLASS = 0
# Default runs generated Bug-car demo data. Change to "yolo" only when using a real YOLO model.
DETECTOR_KIND = "orange_demo"

# Dataset and video paths. The CSV header is: video,length,width,height.
DATASET_CSV = PROJECT_DIR / "data" / "carton_dataset.csv"
INPUT_VIDEO = PROJECT_DIR / "data" / "input.mp4"
OUTPUT_VIDEO = PROJECT_DIR / "output" / "predicted.mp4"
DETECTION_OUTPUT_VIDEO = PROJECT_DIR / "output" / "detections.mp4"
DETECTION_RESULTS_JSONL = PROJECT_DIR / "output" / "detections.jsonl"

# Keep these values identical for feature extraction during training and inference.
BORDER_MARGIN = 2
MIN_SAMPLES = 8
TRAIN_EPOCHS = 300

# Used by predict.py for direct MLP verification only. Feature order: mean/std/max of width, height, area.
PREDICTION_FEATURES = (0.35, 0.22, 0.077, 0.01, 0.01, 0.004, 0.37, 0.24, 0.089)
