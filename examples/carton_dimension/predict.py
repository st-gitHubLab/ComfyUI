"""Run the trained MLP directly on one aggregated 9-value feature vector.

This script is useful for validating the regression model independently from
YOLO/video processing.  The feature order is:
[mean_w, mean_h, mean_area, std_w, std_h, std_area, max_w, max_h, max_area].
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from carton_dimension import TorchMLPRegressor


def load_features(path: str) -> np.ndarray:
    """Load exactly nine aggregated features from .npy, .npz, or comma-separated text."""
    if path.endswith(".npy"):
        features = np.load(path)
    elif path.endswith(".npz"):
        archive = np.load(path)
        if "features" not in archive:
            raise SystemExit("The .npz prediction input must contain a 'features' array.")
        features = archive["features"]
    else:
        features = np.fromstring(path, sep=",", dtype=np.float32)
    features = np.asarray(features, dtype=np.float32).reshape(-1)
    if features.shape != (9,):
        raise SystemExit(f"Expected exactly 9 aggregated features, received shape {features.shape}.")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict carton dimensions from one 9-value feature vector.")
    parser.add_argument("--mlp", required=True, help="checkpoint produced by train_mlp.py")
    parser.add_argument("--features", required=True, help="comma-separated values, a .npy file, or an .npz file with 'features'")
    args = parser.parse_args()

    length, width, height = TorchMLPRegressor(args.mlp).predict(load_features(args.features))
    print(json.dumps({"length": length, "width": width, "height": height}, ensure_ascii=False))


if __name__ == "__main__":
    main()
