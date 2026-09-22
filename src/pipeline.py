"""RoadVision processing pipeline.

Orchestrates detection, tracking, counting, analysis, and visualization
into a single end-to-end video processing flow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.counter import VehicleCounter
from src.detector import VehicleDetector
from src.tracker import TrajectoryManager
from src.traffic_analyzer import TrafficAnalyzer
from src.visualizer import Visualizer


@dataclass
class ProcessResult:
    """Final processing results returned by the pipeline."""

    total_vehicles: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_direction: dict[str, int] = field(default_factory=dict)
    overall_rate_per_hour: float = 0.0
    video_duration_s: float = 0.0
    total_frames: int = 0
    fps: float = 0.0
    output_video_path: str = ""
    output_csv_path: str = ""


class RoadVisionPipeline:
    """End-to-end video processing pipeline.

    Args:
        config (dict): Configuration dict (from config.yaml).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        m = config.get("model", {})
        d = config.get("detection", {})
        t = config.get("tracking", {})
        c = config.get("counting", {})
        tr = config.get("traffic", {})
        o = config.get("output", {})
        names = config.get("class_names", {})

        # 1. Detector (loads YOLO model)
        self.detector = VehicleDetector(
            model_path=m.get("path", "yolo11n.pt"),
            conf=d.get("conf_threshold", 0.3),
            iou=d.get("iou_threshold", 0.5),
            device=m.get("device", ""),
            classes=d.get("vehicle_classes", [2, 3, 5, 7]),
            tracker=t.get("tracker", "bytetrack.yaml"),
        )

        # 2. Trajectory manager
        self.trajectory = TrajectoryManager(
            max_length=t.get("trajectory_length", 50),
        )

        # 3. Counter
        self.counter = VehicleCounter(
            orientation=c.get("line_orientation", "horizontal"),
            line_position=c.get("line_position", 0.5),
            class_names={int(k): v for k, v in names.items()},
        )

        # 4. Traffic analyzer
        self.analyzer = TrafficAnalyzer(
            window_minutes=tr.get("time_window_minutes", 60),
        )

        # 5. Visualizer
        self.visualizer = Visualizer(
            class_names={int(k): v for k, v in names.items()},
            orientation=c.get("line_orientation", "horizontal"),
            line_position=c.get("line_position", 0.5),
            show_trails=True,
            show_hud=True,
        )

        self.output_dir = o.get("save_dir", "outputs")
        self.save_video = o.get("save_video", True)
        self.save_csv = o.get("save_csv", True)
        self.video_codec = o.get("video_codec", "mp4v")

        # Reset state
        self._reset_state()

    def _reset_state(self) -> None:
        """Clear all per-video state."""
        self.trajectory.clear()
        self.counter.clear()
        self.analyzer.clear()

    def process_video(
        self,
        video_path: str,
        output_video_path: str | None = None,
        output_csv_path: str | None = None,
        progress_callback=None,
    ) -> ProcessResult:
        """Process a single video file end-to-end.

        Args:
            video_path: Path to the input video.
            output_video_path: Path for the annotated output video. If None, uses
                ``<save_dir>/<input_name>_annotated.mp4``.
            output_csv_path: Path for the CSV statistics. If None, uses ``<save_dir>/<input_name>_stats.csv``.
            progress_callback: Optional callable(current_frame, total_frames) for progress reporting.

        Returns:
            ProcessResult with all statistics.
        """
        self._reset_state()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine output paths
        stem = Path(video_path).stem
        os.makedirs(self.output_dir, exist_ok=True)
        if output_video_path is None:
            output_video_path = os.path.join(self.output_dir, f"{stem}_annotated.mp4")
        if output_csv_path is None:
            output_csv_path = os.path.join(self.output_dir, f"{stem}_stats.csv")

        # Video writer
        writer = None
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_w, frame_h))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create output video: {output_video_path}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detection + tracking
            detections = self.detector.track_frame(frame, persist=True)

            # Trajectory update
            self.trajectory.update(detections)

            # Counting
            timestamp = frame_idx / fps
            self.counter.update(detections, self.trajectory, frame_idx, timestamp, frame_h, frame_w)

            # Visualize
            self.visualizer.draw_frame(frame, detections, self.trajectory, self.counter, frame_idx, fps)

            if writer is not None:
                writer.write(frame)

            frame_idx += 1
            if progress_callback is not None and total_frames > 0:
                progress_callback(frame_idx, total_frames)

        cap.release()
        if writer is not None:
            writer.release()

        # Collect events for analyzer
        events = self.counter.get_events()
        self.analyzer.add_events(events)

        # Export CSV
        if self.save_csv:
            self.analyzer.export_csv(output_csv_path, fps=fps)

        # Build summary
        summary = self.counter.get_summary()
        by_type = dict(summary.by_type)
        by_dir = dict(summary.by_direction)
        duration = frame_idx / fps if fps > 0 else 0.0

        result = ProcessResult(
            total_vehicles=summary.total,
            by_type=by_type,
            by_direction=by_dir,
            overall_rate_per_hour=round(self.analyzer.get_overall_rate(), 1),
            video_duration_s=round(duration, 1),
            total_frames=frame_idx,
            fps=fps,
            output_video_path=output_video_path if self.save_video else "",
            output_csv_path=output_csv_path if self.save_csv else "",
        )
        return result


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load a YAML config file and return as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
