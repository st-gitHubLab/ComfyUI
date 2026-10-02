"""Generate a deterministic labeled vehicle-video dataset for an end-to-end demo."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from config import DATASET_CSV, INPUT_VIDEO

VIDEO_SIZE = (640, 480)
FPS = 20
ROAD_TOP = 320
# (vehicle body width, vehicle body height); labels below have a learnable relation.
VEHICLES = [(130 + 10 * index, 70 + 5 * (index % 8)) for index in range(18)]


def draw_car(frame: np.ndarray, x: int, body_bottom: int, body_width: int, body_height: int) -> None:
    """Draw one orange side-view car; its orange contour is used as the demo BBox."""
    import cv2

    roof_height = round(body_height * 0.55)
    wheel_radius = max(8, round(body_height * 0.18))
    y = body_bottom - body_height
    roof_left = x + round(body_width * 0.25)
    roof_right = x + round(body_width * 0.72)
    body_top = y + roof_height
    body_bottom = y + body_height
    orange = (0, 140, 255)
    # A single connected silhouette ensures the color detector gets one BBox.
    outline = np.array([(x, body_bottom), (x, body_top), (roof_left, body_top), (roof_left + 18, y), (roof_right - 18, y), (roof_right, body_top), (x + body_width, body_top), (x + body_width, body_bottom)], dtype=np.int32)
    cv2.fillPoly(frame, [outline], orange)
    for wheel_x in (x + round(body_width * 0.22), x + round(body_width * 0.78)):
        cv2.circle(frame, (wheel_x, body_bottom), wheel_radius, (20, 20, 20), -1)
        cv2.circle(frame, (wheel_x, body_bottom), max(3, wheel_radius // 2), (180, 180, 180), -1)


def draw_straight_road(frame: np.ndarray) -> None:
    """Render a level, straight road with horizontal lane markings."""
    import cv2

    image_width, image_height = VIDEO_SIZE
    frame[:ROAD_TOP] = (235, 190, 120)  # blue sky in BGR
    frame[ROAD_TOP:image_height] = (70, 70, 70)
    cv2.rectangle(frame, (0, ROAD_TOP - 12), (image_width, ROAD_TOP), (65, 140, 65), -1)
    for x in range(-30, image_width, 90):
        cv2.rectangle(frame, (x, ROAD_TOP + 86), (x + 48, ROAD_TOP + 92), (230, 230, 230), -1)


def write_vehicle_video(path: Path, body_width: int, body_height: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, VIDEO_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create demo video: {path}")
    image_width, image_height = VIDEO_SIZE
    positions = list(range(-body_width, image_width + 1, 18))
    wheel_radius = max(8, round(body_height * 0.18))
    body_bottom = ROAD_TOP - wheel_radius
    try:
        for x in positions:
            frame = np.empty((image_height, image_width, 3), dtype=np.uint8)
            draw_straight_road(frame)
            draw_car(frame, x, body_bottom, body_width, body_height)
            cv2.putText(frame, "DEMO CAR ON A STRAIGHT ROAD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    dataset_dir = DATASET_CSV.parent / "videos"
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (body_width, body_height) in enumerate(VEHICLES, start=1):
        filename = f"vehicle_{index:03}.mp4"
        write_vehicle_video(dataset_dir / filename, body_width, body_height)
        # Synthetic ground truth in mm, tied to video geometry for a learnable demo.
        length = body_width * 30.0
        width = body_height * 18.0
        height = 0.35 * length + 0.25 * width
        rows.append((f"videos/{filename}", length, width, height))
    with DATASET_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("video", "length", "width", "height"))
        writer.writerows(rows)
    write_vehicle_video(INPUT_VIDEO, 215, 108)
    print(f"Created {len(rows)} vehicle training videos: {DATASET_CSV}")
    print(f"Created vehicle inference video: {INPUT_VIDEO}")
    print('Set DETECTOR_KIND = "orange_demo" in config.py before training this synthetic demo.')


if __name__ == "__main__":
    main()
