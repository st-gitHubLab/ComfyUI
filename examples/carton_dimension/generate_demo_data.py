"""Generate fixed-camera videos of differently sized Bug (Beetle-style) cars on a straight road."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from config import DATASET_CSV, INPUT_VIDEO

VIDEO_SIZE = (640, 480)
FPS = 20
ROAD_TOP = 320
# Fixed camera: all vehicles share this road, horizon, resolution, and scale.
# Each tuple is a distinct real-world (length_mm, width_mm, height_mm).
BUG_CARS_MM = [
    (3600, 1550, 1420), (3850, 1680, 1510), (4100, 1720, 1480),
    (4300, 1800, 1600), (4500, 1750, 1520), (4700, 1880, 1650),
    (4900, 1900, 1580), (5100, 1980, 1720), (5300, 1850, 1680),
    (3750, 1620, 1500), (4050, 1780, 1570), (4350, 1700, 1630),
    (4600, 1920, 1700), (4800, 1810, 1550), (5000, 2000, 1750),
    (5200, 1860, 1660), (4400, 1960, 1740), (3950, 1640, 1460),
]


def pixel_geometry(length_mm: int, width_mm: int, height_mm: int) -> tuple[int, int, int]:
    """Project L/W/H to a repeatable fixed-camera three-quarter-view silhouette."""
    body_length = round(length_mm / 21)
    body_height = round(height_mm / 19)
    visible_depth = round(width_mm / 42)
    return body_length, body_height, visible_depth


def draw_bug_car(frame: np.ndarray, x: int, body_bottom: int, length_px: int, height_px: int, depth_px: int) -> None:
    """Draw an orange Bug/Beetle-style car; its orange contour is the demo BBox."""
    import cv2

    roof_height = round(height_px * 0.55)
    wheel_radius = max(8, round(height_px * 0.18))
    y = body_bottom - height_px
    roof_left = x + round(length_px * 0.25)
    roof_right = x + round(length_px * 0.72)
    body_top = y + roof_height
    orange = (0, 140, 255)
    # The depth offset makes physical width visible with an unchanged camera.
    # Rounded roof/body make this a Bug (Beetle-style) car rather than a box.
    outline = np.array([
        (x, body_bottom), (x + 8, body_top), (roof_left, body_top),
        (roof_right + depth_px, body_top), (x + length_px + depth_px, body_top + 10),
        (x + length_px, body_bottom),
    ], dtype=np.int32)
    cv2.fillPoly(frame, [outline], orange)
    cv2.ellipse(frame, (x + length_px // 2, body_top), (max(18, length_px // 4), roof_height), 0, 180, 360, orange, -1)
    for wheel_x in (x + round(length_px * 0.22), x + round(length_px * 0.78)):
        cv2.circle(frame, (wheel_x, body_bottom), wheel_radius, (20, 20, 20), -1)
        cv2.circle(frame, (wheel_x, body_bottom), max(3, wheel_radius // 2), (180, 180, 180), -1)


def draw_straight_road(frame: np.ndarray) -> None:
    """Render the stationary camera background: a level road and lane markings."""
    import cv2

    image_width, image_height = VIDEO_SIZE
    frame[:ROAD_TOP] = (235, 190, 120)  # blue sky in BGR
    frame[ROAD_TOP:image_height] = (70, 70, 70)
    cv2.rectangle(frame, (0, ROAD_TOP - 12), (image_width, ROAD_TOP), (65, 140, 65), -1)
    for x in range(-30, image_width, 90):
        cv2.rectangle(frame, (x, ROAD_TOP + 86), (x + 48, ROAD_TOP + 92), (230, 230, 230), -1)


def write_bug_car_video(path: Path, length_mm: int, width_mm: int, height_mm: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, VIDEO_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create demo video: {path}")
    image_width, image_height = VIDEO_SIZE
    length_px, height_px, depth_px = pixel_geometry(length_mm, width_mm, height_mm)
    positions = list(range(-(length_px + depth_px), image_width + 1, 18))
    wheel_radius = max(8, round(height_px * 0.18))
    body_bottom = ROAD_TOP - wheel_radius
    try:
        for x in positions:
            frame = np.empty((image_height, image_width, 3), dtype=np.uint8)
            draw_straight_road(frame)
            draw_bug_car(frame, x, body_bottom, length_px, height_px, depth_px)
            cv2.putText(frame, "FIXED CAMERA - BUG CAR ON STRAIGHT ROAD", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    dataset_dir = DATASET_CSV.parent / "videos"
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (length_mm, width_mm, height_mm) in enumerate(BUG_CARS_MM, start=1):
        filename = f"bug_car_{index:03}.mp4"
        write_bug_car_video(dataset_dir / filename, length_mm, width_mm, height_mm)
        rows.append((f"videos/{filename}", length_mm, width_mm, height_mm))
    with DATASET_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("video", "length", "width", "height"))
        writer.writerows(rows)
    # An unseen L/W/H combination uses exactly the same fixed-camera road.
    write_bug_car_video(INPUT_VIDEO, 4650, 1830, 1620)
    print(f"Created {len(rows)} fixed-camera Bug-car training videos: {DATASET_CSV}")
    print(f"Created Bug-car inference video: {INPUT_VIDEO}")
    print('Set DETECTOR_KIND = "orange_demo" in config.py before training this synthetic demo.')


if __name__ == "__main__":
    main()
