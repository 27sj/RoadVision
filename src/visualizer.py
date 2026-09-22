"""Visualizer: draws detection boxes, track IDs, motion trails, the virtual
counting line, and a HUD with live statistics on output video frames.

All drawing uses OpenCV (cv2) for performance.  The color palette is fixed
per class so that the same vehicle type always uses the same color.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.counter import VehicleCounter
from src.detector import Detection
from src.tracker import TrajectoryManager

# Class color palette (BGR) — fixed per class for visual consistency.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    2: (0, 255, 128),  # car      — green
    3: (0, 165, 255),  # motorcycle — orange
    5: (0, 0, 255),  # bus      — red
    7: (255, 0, 0),  # truck    — blue
}
DEFAULT_COLOR = (200, 200, 200)


class Visualizer:
    """Draw detection results and overlay statistics on video frames.

    Args:
        class_names (dict[int, str]): COCO class ID -> name.
        orientation (str): "horizontal" or "vertical" counting line.
        line_position (float): Line position as fraction of frame dimension.
        show_trails (bool): Draw motion trails behind each vehicle.
        show_hud (bool): Draw the statistics HUD overlay.
    """

    def __init__(
        self,
        class_names: dict[int, str],
        orientation: str = "horizontal",
        line_position: float = 0.5,
        show_trails: bool = True,
        show_hud: bool = True,
    ) -> None:
        self.class_names = class_names
        self.orientation = orientation.lower()
        self.line_position = line_position
        self.show_trails = show_trails
        self.show_hud = show_hud

    def draw_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        traj: TrajectoryManager,
        counter: VehicleCounter,
        frame_idx: int,
        fps: float,
    ) -> np.ndarray:
        """Draw all overlays onto *frame* (in-place) and return it.

        Args:
            frame: BGR image (HxWx3 uint8).
            detections: Current-frame detections.
            traj: Trajectory manager with trail data.
            counter: Counter with crossing events.
            frame_idx: Current frame number.
            fps: Video FPS (for timestamp display).

        Returns:
            The annotated frame (same array, drawn in-place).
        """
        img = frame

        # 1. Motion trails
        if self.show_trails:
            self._draw_trails(img, traj)

        # 2. Detection boxes + labels
        for det in detections:
            self._draw_detection(img, det)

        # 3. Virtual counting line
        self._draw_counting_line(img)

        # 4. HUD with live stats
        if self.show_hud:
            self._draw_hud(img, counter, frame_idx, fps)

        return img

    # ------------------------------------------------------------------ #
    #  Private drawing helpers                                             #
    # ------------------------------------------------------------------ #

    def _draw_trails(self, img: np.ndarray, traj: TrajectoryManager) -> None:
        """Draw fading motion trails for each tracked vehicle."""
        all_trails = traj.get_all_trails()
        for points in all_trails.values():
            if len(points) < 2:
                continue
            n = len(points)
            for i in range(1, n):
                # Fade older points toward transparent
                alpha = i / n
                thickness = max(1, int(2 * alpha))
                cv2.line(
                    img,
                    (int(points[i - 1][0]), int(points[i - 1][1])),
                    (int(points[i][0]), int(points[i][1])),
                    (0, 200, 255),
                    thickness,
                    cv2.LINE_AA,
                )

    def _draw_detection(self, img: np.ndarray, det: Detection) -> None:
        """Draw a bounding box with class name, confidence, and track ID."""
        x1, y1, x2, y2 = det.xyxy.astype(int)
        color = CLASS_COLORS.get(det.cls_id, DEFAULT_COLOR)
        label_parts = []
        if det.track_id != -1:
            label_parts.append(f"ID:{det.track_id}")
        cls_name = self.class_names.get(det.cls_id, f"cls{det.cls_id}")
        label_parts.append(cls_name)
        label_parts.append(f"{det.conf:.2f}")
        label = " ".join(label_parts)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        self._draw_label(img, label, (x1, y1 - 10), color)

    @staticmethod
    def _draw_label(img: np.ndarray, label: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
        """Draw a filled rectangle with text for a label."""
        x, y = pos
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Ensure the label stays within the frame
        y = max(y, th + 2)
        cv2.rectangle(img, (x, y - th - baseline), (x + tw + 4, y), color, -1)
        cv2.putText(img, label, (x + 2, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_counting_line(self, img: np.ndarray) -> None:
        """Draw the virtual counting line across the frame."""
        h, w = img.shape[:2]
        line_color = (0, 255, 255)  # yellow
        if self.orientation == "horizontal":
            ly = int(self.line_position * h)
            cv2.line(img, (0, ly), (w, ly), line_color, 2, cv2.LINE_AA)
            cv2.putText(img, "Counting Line", (10, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1, cv2.LINE_AA)
        else:
            lx = int(self.line_position * w)
            cv2.line(img, (lx, 0), (lx, h), line_color, 2, cv2.LINE_AA)
            cv2.putText(img, "Counting Line", (lx + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1, cv2.LINE_AA)

    def _draw_hud(self, img: np.ndarray, counter: VehicleCounter, frame_idx: int, fps: float) -> None:
        """Draw a semi-transparent stats panel in the top-right corner."""
        summary = counter.get_summary()
        ts = frame_idx / fps if fps > 0 else 0.0

        lines = [
            f"Frame: {frame_idx}  Time: {ts:.1f}s",
            f"Total: {summary.total}",
        ]
        for cls_name, cnt in sorted(summary.by_type.items()):
            lines.append(f"  {cls_name}: {cnt}")
        for direction, cnt in sorted(summary.by_direction.items()):
            lines.append(f"  {direction}: {cnt}")

        # Draw a semi-transparent background panel
        panel_w = 260
        line_h = 22
        panel_h = line_h * len(lines) + 10
        _h, w = img.shape[:2]
        x0 = w - panel_w - 10
        y0 = 10
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        for i, text in enumerate(lines):
            cv2.putText(
                img,
                text,
                (x0 + 8, y0 + 20 + i * line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
