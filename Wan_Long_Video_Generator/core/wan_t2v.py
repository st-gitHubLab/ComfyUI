"""Wan2.1 text-to-video adapter.

Replace the placeholder implementation with your local Wan2.1 inference call.
"""
from __future__ import annotations

from pathlib import Path

from config import GenerationConfig, T2V_MODEL_DIR


def generate_t2v(prompt: str, output_path: Path, config: GenerationConfig) -> Path:
    """Generate the first clip from text using the Wan2.1 T2V model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "Connect Wan2.1 T2V inference here. "
        f"Model path: {T2V_MODEL_DIR}; prompt: {prompt!r}; output: {output_path}; config: {config}"
    )
