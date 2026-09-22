"""Vehicle detection module using YOLO pretrained models.

This module wraps the Ultralytics YOLO API to provide a clean interface for
loading models and running detection / tracking on video frames.  The actual
deep-learning inference is performed by the ultralytics library — we do **not**
re-implement the network here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Detection:
    """A single vehicle detection result for one frame.

    Attributes:
        track_id (int): Stable tracking ID assigned by ByteTrack (-1 if none).
        cls_id (int): COCO class ID (2=car, 3=motorcycle, 5=bus, 7=truck).
        conf (float): Detection confidence score (0-1).
        xyxy (np.ndarray): Bounding box [x1, y1, x2, y2] in pixel coordinates.
    """

    track_id: int
    cls_id: int
    conf: float
    xyxy: np.ndarray


class VehicleDetector:
    """YOLO-based vehicle detector.

    Loads a pretrained YOLO model and runs combined detection + ByteTrack tracking
    on video frames via ``model.track()``.

    Args:
        model_path (str): Model name or path, e.g. ``"yolo11n.pt"``.  Auto-downloads on first use.
        conf (float): Confidence threshold (detections below this are discarded).
        iou (float): NMS IoU threshold.
        device (str): ``""`` for auto, ``"cpu"`` for CPU, ``"0"`` for GPU 0.
        classes (list[int] | None): COCO class IDs to keep.  ``None`` = all classes.
            Default ``[2, 3, 5, 7]`` (car, motorcycle, bus, truck).
        tracker (str): Tracker config name.  Default ``"bytetrack.yaml"``.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.3,
        iou: float = 0.5,
        device: str = "",
        classes: list[int] | None = None,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        from ultralytics import YOLO

        if classes is None:
            classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.classes = classes
        self.tracker = tracker
        # COCO class names from the model's metadata
        self.names: dict[int, str] = self.model.names

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def track_frame(self, frame: np.ndarray, persist: bool = True) -> list[Detection]:
        """Run detection + ByteTrack tracking on a single BGR frame.

        Args:
            frame (np.ndarray): BGR image (HxWx3, uint8).
            persist (bool): Keep tracker state across calls (required for stable IDs).

        Returns:
            list[Detection]: One entry per detected vehicle in this frame.
        """
        results = self.model.track(
            source=frame,
            persist=persist,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device or None,
            verbose=False,
        )
        return self._parse_result(results[0])

    @staticmethod
    def _parse_result(result: Any) -> list[Detection]:
        """Extract Detection list from a single Ultralytics Results object."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        ids_raw = boxes.id
        ids = (
            ids_raw.cpu().numpy() if ids_raw is not None and hasattr(ids_raw, "cpu")
            else (np.asarray(ids_raw) if ids_raw is not None else None)
        )

        detections: list[Detection] = []
        for i in range(len(xyxy)):
            tid = int(ids[i]) if ids is not None else -1
            detections.append(
                Detection(
                    track_id=tid,
                    cls_id=int(clss[i]),
                    conf=float(confs[i]),
                    xyxy=xyxy[i].copy(),
                )
            )
        return detections
