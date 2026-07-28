"""Simple prompt-to-storyboard expansion."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StoryboardShot:
    """A single five-second video shot."""

    index: int
    prompt: str
    seconds: int


def build_storyboard(prompt: str, total_seconds: int, clip_seconds: int = 5) -> list[StoryboardShot]:
    """Split a long-video idea into five-second storyboard shots."""
    if total_seconds <= 0:
        raise ValueError("total_seconds must be greater than zero")
    if clip_seconds <= 0:
        raise ValueError("clip_seconds must be greater than zero")

    shot_count = max(1, math.ceil(total_seconds / clip_seconds))
    return [
        StoryboardShot(
            index=index,
            prompt=(
                f"{prompt}. Shot {index + 1} of {shot_count}, cinematic continuity, "
                "consistent subject, smooth camera movement."
            ),
            seconds=min(clip_seconds, total_seconds - index * clip_seconds),
        )
        for index in range(shot_count)
    ]
