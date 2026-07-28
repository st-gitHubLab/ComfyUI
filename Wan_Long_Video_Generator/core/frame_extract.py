"""Extract continuity frames from generated clips."""
from __future__ import annotations

from pathlib import Path

from .utils import run_command


def extract_last_frame(video_path: Path, frame_path: Path) -> Path:
    """Save the last frame of a clip for image-to-video continuation."""
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg",
        "-y",
        "-sseof",
        "-0.1",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(frame_path),
    ])
    return frame_path
