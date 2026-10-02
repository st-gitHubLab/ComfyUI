"""Run the configured MLP directly on the configured 9-value feature vector."""
from __future__ import annotations

import json

import numpy as np

from carton_dimension import TorchMLPRegressor
from config import MLP_CHECKPOINT, PREDICTION_FEATURES


def main() -> None:
    features = np.asarray(PREDICTION_FEATURES, dtype=np.float32)
    if features.shape != (9,):
        raise SystemExit("PREDICTION_FEATURES in config.py must contain exactly 9 values.")
    length, width, height = TorchMLPRegressor(MLP_CHECKPOINT).predict(features)
    print(json.dumps({"length": length, "width": width, "height": height}, ensure_ascii=False))


if __name__ == "__main__":
    main()
