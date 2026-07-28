"""Configuration for Wan Long Video Generator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
CLIPS_DIR = OUTPUT_DIR / "clips"
FRAMES_DIR = OUTPUT_DIR / "frames"
FINAL_DIR = OUTPUT_DIR / "final"
MODELS_DIR = ROOT_DIR / "models"
T2V_MODEL_DIR = MODELS_DIR / "Wan2.1-T2V-1.3B"
I2V_MODEL_DIR = MODELS_DIR / "Wan2.1-I2V-1.3B"


@dataclass(frozen=True)
class GenerationConfig:
    """Runtime parameters shared by the generation pipeline."""

    clip_seconds: int = 5
    fps: int = 16
    width: int = 832
    height: int = 480
    seed: int = 42
    guidance_scale: float = 6.0
    negative_prompt: str = "low quality, blurry, distorted, watermark, text"


DEFAULT_CONFIG = GenerationConfig()
