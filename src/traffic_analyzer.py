"""Traffic flow analysis: computes veh/h rate over a time window and exports
per-vehicle and per-event CSV statistics.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass

from src.counter import CountEvent


@dataclass
class FlowRate:
    """A single flow-rate measurement.

    Attributes:
        window_start (float): Start of the time window (seconds).
        window_end (float): End of the time window (seconds).
        vehicle_count (int): Vehicles counted in this window.
        rate_per_hour (float): Extrapolated vehicles per hour.
    """

    window_start: float
    window_end: float
    vehicle_count: int
    rate_per_hour: float


class TrafficAnalyzer:
    """Compute traffic flow statistics from crossing events.

    Args:
        window_minutes (int): Length of each time window in minutes. The final ``veh/h`` rate is computed over the
            entire video duration, and also per-window for trend analysis.
    """

    def __init__(self, window_minutes: int = 60) -> None:
        self.window_minutes = window_minutes
        self._events: list[CountEvent] = []

    def add_events(self, events: list[CountEvent]) -> None:
        """Append a batch of crossing events."""
        self._events.extend(events)

    def get_total_duration(self) -> float:
        """Return the total video duration covered by events (seconds), or 0 if none."""
        if not self._events:
            return 0.0
        return max(e.timestamp for e in self._events)

    def get_overall_rate(self) -> float:
        """Compute the overall traffic flow rate (vehicles per hour).

        If the video is shorter than 1 minute the rate is extrapolated
        from the available data; this should be noted when reporting.
        """
        duration = self.get_total_duration()
        if duration <= 0:
            return 0.0
        return len(self._events) / duration * 3600.0

    def get_window_rates(self) -> list[FlowRate]:
        """Compute per-window flow rates (each window = window_minutes minutes)."""
        if not self._events:
            return []

        window_sec = self.window_minutes * 60
        max_ts = self.get_total_duration()
        num_windows = max(1, int(max_ts // window_sec) + (1 if max_ts % window_sec > 0 else 0))

        rates: list[FlowRate] = []
        for i in range(num_windows):
            start = i * window_sec
            end = (i + 1) * window_sec
            count = sum(1 for e in self._events if start <= e.timestamp < end)
            actual_end = min(end, max_ts) if i == num_windows - 1 else end
            duration = actual_end - start
            rate = (count / duration * 3600.0) if duration > 0 else 0.0
            rates.append(
                FlowRate(
                    window_start=start,
                    window_end=actual_end,
                    vehicle_count=count,
                    rate_per_hour=round(rate, 1),
                )
            )
        return rates

    def get_by_type(self) -> dict[str, int]:
        """Return counts by vehicle type."""
        counts: dict[str, int] = defaultdict(int)
        for e in self._events:
            counts[e.cls_name] += 1
        return dict(counts)

    def get_by_direction(self) -> dict[str, int]:
        """Return counts by direction."""
        counts: dict[str, int] = defaultdict(int)
        for e in self._events:
            counts[e.direction] += 1
        return dict(counts)

    def export_csv(self, csv_path: str, fps: float = 30.0) -> str:
        """Write per-vehicle crossing events to a CSV file.

        Args:
            csv_path (str): Output file path.
            fps (float): Video FPS, used to add a frame-number column.

        Returns:
            str: The path written.
        """
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "class", "direction", "frame_idx", "timestamp_s"])
            for e in sorted(self._events, key=lambda x: x.timestamp):
                writer.writerow([e.track_id, e.cls_name, e.direction, e.frame_idx, f"{e.timestamp:.3f}"])
        return csv_path

    def clear(self) -> None:
        """Reset all stored events."""
        self._events.clear()
