#!/usr/bin/env python3
"""RoadVision CLI: batch-process road videos from the command line.

Usage:
    python scripts/run_video.py --video path/to/road.mp4
    python scripts/run_video.py --video path/to/road.mp4 --model yolov8s.pt --conf 0.4
    python scripts/run_video.py --video path/to/road.mp4 --line-orientation vertical --line-position 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import RoadVisionPipeline, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RoadVision: Road vehicle detection, tracking, and traffic flow analysis.",
    )
    p.add_argument("--video", required=True, help="Path to the input video file.")
    p.add_argument(
        "--config", default="configs/config.yaml", help="Path to config.yaml (default: configs/config.yaml)."
    )
    p.add_argument("--model", default=None, help="Override model path (e.g. yolo11s.pt).")
    p.add_argument("--conf", type=float, default=None, help="Override confidence threshold (0-1).")
    p.add_argument("--device", default=None, help='Override device ("" for auto, "cpu", "0").')
    p.add_argument(
        "--line-orientation",
        default=None,
        choices=["horizontal", "vertical"],
        help="Override counting line orientation.",
    )
    p.add_argument("--line-position", type=float, default=None, help="Override counting line position (0.0-1.0).")
    p.add_argument("--output-dir", default=None, help="Override output directory.")
    p.add_argument("--no-save-video", action="store_true", help="Skip saving annotated video.")
    p.add_argument("--no-save-csv", action="store_true", help="Skip saving CSV statistics.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load and override config
    config = load_config(args.config)

    if args.model is not None:
        config.setdefault("model", {})["path"] = args.model
    if args.conf is not None:
        config.setdefault("detection", {})["conf_threshold"] = args.conf
    if args.device is not None:
        config.setdefault("model", {})["device"] = args.device
    if args.line_orientation is not None:
        config.setdefault("counting", {})["line_orientation"] = args.line_orientation
    if args.line_position is not None:
        config.setdefault("counting", {})["line_position"] = args.line_position
    if args.output_dir is not None:
        config.setdefault("output", {})["save_dir"] = args.output_dir
    if args.no_save_video:
        config.setdefault("output", {})["save_video"] = False
    if args.no_save_csv:
        config.setdefault("output", {})["save_csv"] = False

    pipeline = RoadVisionPipeline(config)

    print(f"[RoadVision] Processing: {args.video}")
    print(f"  Model: {config['model']['path']}")
    print(f"  Confidence: {config['detection']['conf_threshold']}")

    def progress(cur, total):
        pct = cur / total * 100 if total > 0 else 0
        sys.stdout.write(f"\r  Frame {cur}/{total} ({pct:.1f}%)")
        sys.stdout.flush()

    result = pipeline.process_video(args.video, progress_callback=progress)
    print()  # newline after progress

    print("\n=== RoadVision Results ===")
    print(f"  Total vehicles:   {result.total_vehicles}")
    print(f"  By type:          {result.by_type}")
    print(f"  By direction:     {result.by_direction}")
    print(f"  Overall rate:     {result.overall_rate_per_hour} veh/h")
    print(f"  Duration:        {result.video_duration_s}s ({result.total_frames} frames)")
    if result.output_video_path:
        print(f"  Output video:     {result.output_video_path}")
    if result.output_csv_path:
        print(f"  Output CSV:       {result.output_csv_path}")


if __name__ == "__main__":
    main()
