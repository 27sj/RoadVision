"""Trajectory manager: maintains per-vehicle position history for trail drawing.

The Ultralytics ``model.track()`` call already assigns stable ByteTrack IDs.
This module complements it by keeping a short history of each track's center
coordinates so that we can:
  1. Draw motion trails on the output video.
  2. Provide position data to the line-crossing counter.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.detector import Detection


def _box_center(xyxy: np.ndarray) -> tuple[float, float]:
    """Return the bottom-centre point of a bounding box (ground contact point)."""
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, y2)


class TrajectoryManager:
    """Maintain a rolling history of each track's centre position.

    Args:
        max_length (int): Maximum number of points to keep per track ID.
    """

    def __init__(self, max_length: int = 50) -> None:
        self.max_length = max_length
        # track_id -> deque of (x, y) points
        self._trails: dict[int, deque] = {}
        # track_id -> last seen (x, y) — used for line-crossing logic
        self._prev_centres: dict[int, tuple[float, float]] = {}

    def update(self, detections: list[Detection]) -> None:
        """Append the current centre of each tracked vehicle to its trail.

        Detections without a valid track_id (track_id == -1) are ignored.
        """
        for det in detections:
            tid = det.track_id
            if tid == -1:
                continue
            cx, cy = _box_center(det.xyxy)

            if tid not in self._trails:
                self._trails[tid] = deque(maxlen=self.max_length)
            self._trails[tid].append((cx, cy))

    def get_trail(self, track_id: int) -> list[tuple[float, float]]:
        """Return the trajectory points for *track_id* as a list of (x, y)."""
        trail = self._trails.get(track_id)
        return list(trail) if trail else []

    def get_all_trails(self) -> dict[int, list[tuple[float, float]]]:
        """Return a snapshot of all active trails (track_id -> list of points)."""
        return {tid: list(trail) for tid, trail in self._trails.items()}

    def get_prev_centre(self, track_id: int) -> tuple[float, float] | None:
        """Return the previous centre for *track_id*, or None if none recorded."""
        return self._prev_centres.get(track_id)

    def set_prev_centre(self, track_id: int, centre: tuple[float, float]) -> None:
        """Record the previous-frame centre for *track_id*."""
        self._prev_centres[track_id] = centre

    def clear(self) -> None:
        """Reset all trajectory state (call before processing a new video)."""
        self._trails.clear()
        self._prev_centres.clear()
