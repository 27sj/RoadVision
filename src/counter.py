"""Virtual counting line: detects line crossings, determines direction, and
classifies counts by vehicle type and travel direction.

The counter works by comparing each vehicle's position between consecutive
frames.  When the vehicle's center crosses the virtual line, a count event is
recorded with: track_id, class name, direction ("up"/"down" or
"left"/"right"), timestamp, and the frame number.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.detector import Detection
from src.tracker import TrajectoryManager, _box_center


@dataclass
class CountEvent:
    """A single line-crossing event.

    Attributes:
        track_id (int): ByteTrack stable ID.
        cls_name (str): Vehicle class name (car, truck, etc.).
        direction (str): Travel direction: "up", "down", "left", or "right".
        frame_idx (int): Frame number in the video.
        timestamp (float): Time in seconds from the start of the video.
    """

    track_id: int
    cls_name: str
    direction: str
    frame_idx: int
    timestamp: float


@dataclass
class CountSummary:
    """Aggregated counting results."""

    events: list[CountEvent] = field(default_factory=list)
    by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_direction: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def total(self) -> int:
        return len(self.events)


class VehicleCounter:
    """Virtual-line crossing counter.

    Args:
        orientation (str): "horizontal" (vehicles cross left-right or counting up/down) or "vertical" (vehicles cross
            top-bottom or left/right).
        line_position (float): Line position as a fraction of the frame dimension (0.0-1.0). For horizontal lines this
            is a fraction of height; for vertical lines, a fraction of width.
        class_names (dict[int, str]): Mapping from COCO class ID to name.
    """

    def __init__(
        self,
        orientation: str = "horizontal",
        line_position: float = 0.5,
        class_names: dict[int, str] | None = None,
    ) -> None:
        self.orientation = orientation.lower()
        if self.orientation not in ("horizontal", "vertical"):
            raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation!r}")
        self.line_position = line_position
        self.class_names = class_names or {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self._events: list[CountEvent] = []
        self._counted_ids: set[int] = set()

    def update(
        self,
        detections: list[Detection],
        traj: TrajectoryManager,
        frame_idx: int,
        timestamp: float,
        frame_h: int,
        frame_w: int,
    ) -> list[CountEvent]:
        """Check for line crossings in this frame.

        Returns the list of new crossing events detected in this frame.
        """
        new_events: list[CountEvent] = []

        for det in detections:
            tid = det.track_id
            if tid == -1:
                continue

            cx, cy = _box_center(det.xyxy)
            prev = traj.get_prev_centre(tid)

            # Always update the previous center for the next frame
            traj.set_prev_centre(tid, (cx, cy))

            if prev is None:
                continue

            if tid in self._counted_ids:
                continue

            crossed = False
            direction = ""

            if self.orientation == "horizontal":
                line_y = self.line_position * frame_h
                py = prev[1]
                # Crossing detected when the vehicle moves past the line between frames
                if (py < line_y <= cy) or (py > line_y >= cy):
                    crossed = True
                    direction = "down" if cy > py else "up"
            else:  # vertical
                line_x = self.line_position * frame_w
                px = prev[0]
                if (px < line_x <= cx) or (px > line_x >= cx):
                    crossed = True
                    direction = "right" if cx > px else "left"

            if crossed:
                cls_name = self.class_names.get(det.cls_id, f"class_{det.cls_id}")
                event = CountEvent(
                    track_id=tid,
                    cls_name=cls_name,
                    direction=direction,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                )
                self._events.append(event)
                self._counted_ids.add(tid)
                new_events.append(event)

        return new_events

    def get_summary(self) -> CountSummary:
        """Return aggregated counts by type and direction."""
        by_type: dict[str, int] = defaultdict(int)
        by_dir: dict[str, int] = defaultdict(int)
        for e in self._events:
            by_type[e.cls_name] += 1
            by_dir[e.direction] += 1
        return CountSummary(events=list(self._events), by_type=by_type, by_direction=by_dir)

    def get_events(self) -> list[CountEvent]:
        """Return all recorded crossing events."""
        return list(self._events)

    def clear(self) -> None:
        """Reset all counting state."""
        self._events.clear()
        self._counted_ids.clear()
