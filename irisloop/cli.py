"""CLI entry point: irisloop camera video capture (built-in webcam + USB camera)."""

from __future__ import annotations

import argparse
import sys

from .camera import probe_cameras
from .capture import capture


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="irisloop",
        description="Capture a camera video stream and save it locally (built-in webcam or USB camera)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-o", "--output", default=None, help="output file path; default captures/capture_<timestamp>.mp4")
    p.add_argument("-d", "--output-dir", default="captures", help="output directory when --output is omitted")
    p.add_argument(
        "-i", "--camera-index", type=int, default=0,
        help="camera index (0 is usually the built-in webcam; external USB cams are 1+. See --list)",
    )
    p.add_argument("--width", type=int, default=1280, help="requested width")
    p.add_argument("--height", type=int, default=720, help="requested height")
    p.add_argument("--fps", type=int, default=30, help="requested frame rate")
    p.add_argument("-t", "--duration", type=float, default=None, help="record duration in seconds; omit to stop manually")
    p.add_argument("--no-preview", action="store_true", help="do not show preview window (headless record)")
    p.add_argument("--no-timestamp", action="store_true", help="do not overlay a timestamp on frames")
    p.add_argument("--no-measure", action="store_true", help="skip measured fps at startup")
    p.add_argument("--codec", default=None, help="force codec, e.g. mp4v / MJPG / XVID")
    p.add_argument("--list", action="store_true", help="list available cameras and exit")
    return p


def cmd_list() -> int:
    cams = probe_cameras()
    if not cams:
        print("no cameras found")
        return 1
    print("available cameras:")
    for c in cams:
        print(f"  [{c['index']}] {c['width']}x{c['height']} @{c['fps']:.0f}fps backend={c['backend']}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        return cmd_list()

    try:
        capture(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            output=args.output,
            output_dir=args.output_dir,
            duration=args.duration,
            preview=not args.no_preview,
            timestamp=not args.no_timestamp,
            measure=not args.no_measure,
            codec=args.codec,
        )
    except KeyboardInterrupt:
        print("\n[info] interrupted (video saved)")
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
