"""FFmpeg helpers for stitching generated clips."""
from __future__ import annotations

from pathlib import Path

from .utils import run_command


def concat_videos(clips: list[Path], output_path: Path) -> Path:
    """Concatenate clips into a final MP4 using ffmpeg's concat demuxer."""
    if not clips:
        raise ValueError("At least one clip is required for concatenation")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_path.with_suffix(".txt")
    list_file.write_text(
        "\n".join(f"file '{clip.resolve().as_posix()}'" for clip in clips),
        encoding="utf-8",
    )
    run_command([
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output_path),
    ])
    return output_path
