"""Utility helpers for paths, command execution, and filenames."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable

from config import CLIPS_DIR, FINAL_DIR, FRAMES_DIR


def ensure_directories() -> None:
    """Create runtime output directories if they do not exist."""
    for directory in (CLIPS_DIR, FRAMES_DIR, FINAL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def safe_slug(text: str, fallback: str = "video") -> str:
    """Return a filesystem-safe slug from user-provided text."""
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", text).strip("_").lower()
    return slug[:64] or fallback


def run_command(command: Iterable[str]) -> None:
    """Run a command and raise a helpful error if it fails."""
    completed = subprocess.run(list(command), check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def require_file(path: Path, label: str) -> Path:
    """Validate that a generated artifact exists."""
    if not path.exists():
        raise FileNotFoundError(f"{label} was not created: {path}")
    return path
