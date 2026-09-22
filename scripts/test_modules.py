"""Test RoadVision modules that don't require PyTorch.

This script tests:
1. Config loading (config.yaml)
2. Counter logic (line crossing + direction + classification)
3. Traffic analyzer (CSV export, flow rate)
4. Visualizer (drawing on a synthetic frame)
5. Video I/O (create test video, read it back)
6. Module imports (src package)
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np

# --- Test 1: Config loading ---
print("=" * 60)
print("Test 1: Config loading")
print("=" * 60)
from src.pipeline import load_config

config_path = os.path.join(project_root, "configs", "config.yaml")
config = load_config(config_path)
print(f"  Config loaded: {list(config.keys())}")
assert "model" in config
assert "detection" in config
assert "counting" in config
print("  [PASS] Config loaded successfully")

# --- Test 2: Counter logic ---
print()
print("=" * 60)
print("Test 2: Counter logic (line crossing + direction)")
print("=" * 60)
from src.counter import VehicleCounter
from src.detector import Detection
from src.tracker import TrajectoryManager

traj = TrajectoryManager(max_length=10)
counter = VehicleCounter(
    orientation="horizontal",
    line_position=0.5,
    class_names={2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
)

frame_h, frame_w = 480, 640
line_y = 0.5 * frame_h  # 240

# Simulate a car crossing downward: frame 0 above line, frame 1 below
det_f0 = Detection(track_id=1, cls_id=2, conf=0.9, xyxy=np.array([100, 200, 150, 230]))
det_f1 = Detection(track_id=1, cls_id=2, conf=0.9, xyxy=np.array([100, 250, 150, 280]))

traj.update([det_f0])
counter.update([det_f0], traj, 0, 0.0, frame_h, frame_w)

traj.update([det_f1])
events = counter.update([det_f1], traj, 1, 1 / 30, frame_h, frame_w)

assert len(events) == 1, f"Expected 1 event, got {len(events)}"
assert events[0].direction == "down", f"Expected 'down', got {events[0].direction}"
assert events[0].cls_name == "car"
print(f"  Crossing event: id={events[0].track_id}, class={events[0].cls_name}, dir={events[0].direction}")
print("  [PASS] Counter logic works")

# Test direction: upward crossing
counter.clear()
traj.clear()
det_f0_up = Detection(track_id=2, cls_id=7, conf=0.85, xyxy=np.array([300, 280, 350, 310]))
det_f1_up = Detection(track_id=2, cls_id=7, conf=0.85, xyxy=np.array([300, 200, 350, 230]))

traj.update([det_f0_up])
counter.update([det_f0_up], traj, 0, 0.0, frame_h, frame_w)
traj.update([det_f1_up])
events = counter.update([det_f1_up], traj, 1, 1 / 30, frame_h, frame_w)

assert len(events) == 1
assert events[0].direction == "up", f"Expected 'up', got {events[0].direction}"
assert events[0].cls_name == "truck"
print(f"  Upward crossing: id={events[0].track_id}, class={events[0].cls_name}, dir={events[0].direction}")
print("  [PASS] Direction detection works")

# Test deduplication (same ID counted only once)
counter.clear()
traj.clear()
det_a = Detection(track_id=3, cls_id=2, conf=0.9, xyxy=np.array([100, 200, 150, 230]))
det_b = Detection(track_id=3, cls_id=2, conf=0.9, xyxy=np.array([100, 250, 150, 280]))
traj.update([det_a])
counter.update([det_a], traj, 0, 0.0, frame_h, frame_w)
traj.update([det_b])
counter.update([det_b], traj, 1, 1 / 30, frame_h, frame_w)
traj.update([det_b])
events2 = counter.update([det_b], traj, 2, 2 / 30, frame_h, frame_w)
assert len(events2) == 0, "Duplicate count should not happen"
print("  [PASS] Deduplication works (same ID counted only once)")

# --- Test 3: Traffic analyzer ---
print()
print("=" * 60)
print("Test 3: Traffic analyzer (CSV export + flow rate)")
print("=" * 60)
from src.traffic_analyzer import TrafficAnalyzer

analyzer = TrafficAnalyzer(window_minutes=60)
all_events = counter.get_events() + events
analyzer.add_events(all_events)

rate = analyzer.get_overall_rate()
print(f"  Overall rate: {rate} veh/h")
assert rate > 0, "Rate should be > 0 with events"

# CSV export
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    csv_path = f.name
analyzer.export_csv(csv_path, fps=30.0)
assert os.path.exists(csv_path)
import csv as csv_mod

with open(csv_path, encoding="utf-8") as f:
    rows = list(csv_mod.reader(f))
print(f"  CSV exported: {len(rows) - 1} data rows (header + {len(rows) - 1} events)")
assert len(rows) >= 2  # at least header + 1 event
print(f"  Header: {rows[0]}")
print("  [PASS] CSV export works")

# --- Test 4: Visualizer ---
print()
print("=" * 60)
print("Test 4: Visualizer (drawing on synthetic frame)")
print("=" * 60)
from src.visualizer import Visualizer

viz = Visualizer(
    class_names={2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
    orientation="horizontal",
    line_position=0.5,
    show_trails=True,
    show_hud=True,
)

frame = np.full((480, 640, 3), 50, dtype=np.uint8)
dets = [
    Detection(track_id=1, cls_id=2, conf=0.88, xyxy=np.array([100, 100, 200, 200])),
    Detection(track_id=2, cls_id=7, conf=0.75, xyxy=np.array([300, 150, 400, 280])),
]
traj2 = TrajectoryManager(max_length=10)
traj2.update(dets)
counter2 = VehicleCounter(
    orientation="horizontal", line_position=0.5, class_names={2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
)
viz.draw_frame(frame, dets, traj2, counter2, 0, 30.0)

# Verify frame was modified (not all grey anymore)
assert frame.sum() != np.full((480, 640, 3), 50, dtype=np.uint8).sum()
print("  [PASS] Visualizer drew on frame without error")

# Save the test frame for visual inspection
test_frame_path = os.path.join(project_root, "outputs", "test_visualizer.png")
cv2.imwrite(test_frame_path, frame)
print(f"  Test frame saved to: {test_frame_path}")

# --- Test 5: Video I/O ---
print()
print("=" * 60)
print("Test 5: Video I/O (create test video + read back)")
print("=" * 60)
from scripts.make_test_video import make_test_video

test_video = os.path.join(project_root, "assets", "test_road.mp4")
make_test_video(test_video, num_frames=30, fps=30.0)
assert os.path.exists(test_video), "Test video was not created"

cap = cv2.VideoCapture(test_video)
assert cap.isOpened(), "Cannot open test video"
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
cap.release()
assert frame_count == 30, f"Expected 30 frames, got {frame_count}"
print(f"  Video created and read back: {frame_count} frames")
print("  [PASS] Video I/O works")

# --- Test 6: Module imports ---
print()
print("=" * 60)
print("Test 6: Module imports")
print("=" * 60)
# These don't require torch
from src.counter import VehicleCounter
from src.detector import Detection
from src.tracker import TrajectoryManager
from src.traffic_analyzer import TrafficAnalyzer
from src.visualizer import Visualizer

print("  All src modules imported successfully")
print("  [PASS] Module imports work")

# Clean up
os.remove(csv_path)
print()
print("=" * 60)
print("ALL TESTS PASSED (modules not requiring PyTorch)")
print("=" * 60)
