"""Generate a short synthetic test video with moving rectangles.

This creates a simple video for testing the RoadVision pipeline without
requiring a real road video.  The video contains coloured rectangles
moving vertically and horizontally to simulate vehicles crossing a line.

Usage:
    python scripts/make_test_video.py --output assets/test_road.mp4
"""

import argparse

import cv2
import numpy as np


def make_test_video(output: str = "assets/test_road.mp4", num_frames: int = 120, fps: float = 30.0) -> None:
    """Create a synthetic test video."""
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    import os
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    # Define "vehicles": (start_x, start_y, speed_x, speed_y, size, color)
    vehicles = [
        (100, -50, 0, 4, (50, 80), (0, 200, 0)),     # green moving down
        (300, -120, 0, 3, (60, 100), (0, 0, 200)),   # red moving down
        (500, h + 50, 0, -4, (50, 80), (200, 0, 0)),  # blue moving up
        (200, -200, 0, 5, (45, 70), (0, 165, 255)),  # orange moving down
    ]

    for frame_idx in range(num_frames):
        frame = np.full((h, w, 3), 50, dtype=np.uint8)  # grey background

        # Draw road lanes
        cv2.line(frame, (0, h // 2), (w, h // 2), (80, 80, 80), 2)

        for vx, vy, sx, sy, (bw, bh), color in vehicles:
            cy = vy + sy * frame_idx
            cx = vx + sx * frame_idx
            if -bh < cy < h:
                cv2.rectangle(frame,
                               (int(cx - bw / 2), int(cy - bh)),
                               (int(cx + bw / 2), int(cy)),
                               color, -1)

        writer.write(frame)

    writer.release()
    print(f"Test video created: {output} ({num_frames} frames, {fps} fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="assets/test_road.mp4")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    make_test_video(args.output, args.frames, args.fps)
