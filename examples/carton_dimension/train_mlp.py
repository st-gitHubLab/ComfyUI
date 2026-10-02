"""Train the carton MLP from the video dataset configured in config.py."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from carton_dimension import build_detector, extract_video_features
from config import BORDER_MARGIN, CARTON_CLASS, DETECTOR_KIND, DATASET_CSV, MIN_SAMPLES, MLP_CHECKPOINT, TRAIN_EPOCHS, YOLO_WEIGHTS


def load_video_dataset(manifest_path: Path, detector) -> tuple[np.ndarray, np.ndarray]:
    """Convert configured `video,length,width,height` manifest rows into MLP samples."""
    features: list[np.ndarray] = []
    targets: list[tuple[float, float, float]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        required = {"video", "length", "width", "height"}
        if not rows.fieldnames or not required.issubset(rows.fieldnames):
            raise SystemExit("CSV header must be: video,length,width,height")
        for row_number, row in enumerate(rows, start=2):
            video = Path(row["video"])
            if not video.is_absolute():
                video = manifest_path.parent / video
            feature_vector = extract_video_features(video, detector, BORDER_MARGIN, MIN_SAMPLES)
            if feature_vector is None:
                print(f"Skipping row {row_number}: insufficient fully-visible detections in {video}")
                continue
            features.append(feature_vector)
            targets.append((float(row["length"]), float(row["width"]), float(row["height"])))
    if not features:
        raise SystemExit("No usable training videos. Check config.py, YOLO weights, labels, and MIN_SAMPLES.")
    return np.stack(features).astype("float32"), np.asarray(targets, dtype="float32")


def main() -> None:
    detector = build_detector(DETECTOR_KIND, YOLO_WEIGHTS, CARTON_CLASS)
    x, y = load_video_dataset(DATASET_CSV, detector)
    mean, std = x.mean(0), x.std(0).clip(1e-6)
    target_scale = y.std(0).clip(1e-6)
    model = torch.nn.Sequential(torch.nn.Linear(9, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs, targets = torch.tensor((x - mean) / std), torch.tensor(y / target_scale)
    for _ in range(TRAIN_EPOCHS):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
    MLP_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "feature_mean": torch.tensor(mean), "feature_std": torch.tensor(std), "target_scale": torch.tensor(target_scale)}, MLP_CHECKPOINT)
    print(f"saved {MLP_CHECKPOINT}; videos={len(x)}; final normalized MSE={loss.item():.6f}")


if __name__ == "__main__":
    main()
