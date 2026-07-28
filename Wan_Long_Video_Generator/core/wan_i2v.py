"""Wan2.1 image-to-video continuation adapter.

Replace the placeholder implementation with your local Wan2.1 inference call.
"""
from __future__ import annotations

from pathlib import Path

from config import GenerationConfig, I2V_MODEL_DIR


def generate_i2v(prompt: str, image_path: Path, output_path: Path, config: GenerationConfig) -> Path:
    """Generate a continuation clip from the previous clip's last frame."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "Connect Wan2.1 I2V inference here. "
        f"Model path: {I2V_MODEL_DIR}; image: {image_path}; prompt: {prompt!r}; "
        f"output: {output_path}; config: {config}"
    )
