"""Train the carton MLP from labeled videos listed in a CSV manifest."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from carton_dimension import UltralyticsCartonDetector, extract_video_features


def load_video_dataset(manifest_path: str, detector: UltralyticsCartonDetector, border_margin: int, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert `video,length,width,height` manifest rows into MLP samples."""
    manifest = Path(manifest_path)
    features: list[np.ndarray] = []
    targets: list[tuple[float, float, float]] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        required = {"video", "length", "width", "height"}
        if not rows.fieldnames or not required.issubset(rows.fieldnames):
            raise SystemExit("CSV header must be: video,length,width,height")
        for row_number, row in enumerate(rows, start=2):
            video = Path(row["video"])
            if not video.is_absolute():
                video = manifest.parent / video
            feature_vector = extract_video_features(video, detector, border_margin, min_samples)
            if feature_vector is None:
                print(f"Skipping row {row_number}: insufficient fully-visible detections in {video}")
                continue
            features.append(feature_vector)
            targets.append((float(row["length"]), float(row["width"]), float(row["height"])))
    if not features:
        raise SystemExit("No usable training videos. Check YOLO weights, labels, and min-samples.")
    return np.stack(features).astype("float32"), np.asarray(targets, dtype="float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="CSV: video,length,width,height; video paths are relative to this CSV")
    parser.add_argument("--yolo", required=True, help="YOLO carton detector weights")
    parser.add_argument("--carton-class", type=int)
    parser.add_argument("--border-margin", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--output", default="carton_mlp.pt")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    detector = UltralyticsCartonDetector(args.yolo, args.carton_class)
    x, y = load_video_dataset(args.dataset, detector, args.border_margin, args.min_samples)
    mean, std = x.mean(0), x.std(0).clip(1e-6)
    target_scale = y.std(0).clip(1e-6)
    model = torch.nn.Sequential(torch.nn.Linear(9, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs, targets = torch.tensor((x - mean) / std), torch.tensor(y / target_scale)
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
    torch.save({"model": model.state_dict(), "feature_mean": torch.tensor(mean), "feature_std": torch.tensor(std), "target_scale": torch.tensor(target_scale)}, args.output)
    print(f"saved {args.output}; videos={len(x)}; final normalized MSE={loss.item():.6f}")


if __name__ == "__main__":
    main()
