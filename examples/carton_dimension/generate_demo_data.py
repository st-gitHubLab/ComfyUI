"""Generate a deterministic labeled video dataset for an end-to-end demo."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from config import DATASET_CSV, INPUT_VIDEO

VIDEO_SIZE = (640, 480)
FPS = 20
# (pixel width, pixel height); labels below intentionally have a learnable relation.
CARTONS = [(130 + 10 * index, 90 + 7 * (index % 8)) for index in range(18)]


def write_carton_video(path: Path, box_width: int, box_height: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, VIDEO_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create demo video: {path}")
    image_width, image_height = VIDEO_SIZE
    # Starts and finishes outside; frames in the middle are fully visible samples.
    positions = list(range(-box_width, image_width + 1, 18))
    y = (image_height - box_height) // 2
    try:
        for x in positions:
            frame = np.full((image_height, image_width, 3), 35, dtype=np.uint8)
            cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (0, 140, 255), -1)
            cv2.putText(frame, "DEMO CARTON", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    dataset_dir = DATASET_CSV.parent / "videos"
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (box_width, box_height) in enumerate(CARTONS, start=1):
        filename = f"carton_{index:03}.mp4"
        write_carton_video(dataset_dir / filename, box_width, box_height)
        # Synthetic ground truth in mm, tied to video geometry for a learnable demo.
        length = box_width * 2.0
        width = box_height * 2.5
        height = 0.35 * length + 0.25 * width
        rows.append((f"videos/{filename}", length, width, height))
    with DATASET_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("video", "length", "width", "height"))
        writer.writerows(rows)
    # Reuse one unseen-size carton as the configured inference input video.
    write_carton_video(INPUT_VIDEO, 215, 133)
    print(f"Created {len(rows)} training videos: {DATASET_CSV}")
    print(f"Created inference video: {INPUT_VIDEO}")
    print('Set DETECTOR_KIND = "orange_demo" in config.py before training this synthetic demo.')


if __name__ == "__main__":
    main()
