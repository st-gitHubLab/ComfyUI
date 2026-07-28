"""Long-video generation orchestration."""
from __future__ import annotations

from pathlib import Path

from config import CLIPS_DIR, DEFAULT_CONFIG, FINAL_DIR, FRAMES_DIR, GenerationConfig
from .ffmpeg import concat_videos
from .frame_extract import extract_last_frame
from .storyboard import build_storyboard
from .utils import ensure_directories, require_file, safe_slug
from .wan_i2v import generate_i2v
from .wan_t2v import generate_t2v


class LongVideoPipeline:
    """Generate long videos by chaining T2V and I2V five-second clips."""

    def __init__(self, config: GenerationConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        ensure_directories()

    def generate(self, prompt: str, total_seconds: int) -> Path:
        """Generate a long video for a prompt and return the final MP4 path."""
        storyboard = build_storyboard(prompt, total_seconds, self.config.clip_seconds)
        slug = safe_slug(prompt)
        clips: list[Path] = []
        continuity_frame: Path | None = None

        for shot in storyboard:
            clip_path = CLIPS_DIR / f"{slug}_{shot.index:03d}.mp4"
            if shot.index == 0:
                generate_t2v(shot.prompt, clip_path, self.config)
            else:
                if continuity_frame is None:
                    raise RuntimeError("Missing continuity frame for I2V generation")
                generate_i2v(shot.prompt, continuity_frame, clip_path, self.config)

            clips.append(require_file(clip_path, f"Clip {shot.index}"))
            continuity_frame = FRAMES_DIR / f"{slug}_{shot.index:03d}.png"
            extract_last_frame(clip_path, continuity_frame)

        final_path = FINAL_DIR / f"{slug}.mp4"
        return concat_videos(clips, final_path)
