#!/usr/bin/env python
"""Validate the moving-object tracker against a real Hikrobot camera.

No BLE / projector is used. Two modes:

  (default) Inject a textured square with a known trajectory, plus a small
  bright dot that mimics our own laser spot, into LIVE camera frames. Reports
  static-scene false positives, detection rate, centre error, filter lag,
  lead compensation, and whether the tracker ever locks onto the fake dot.

  --live  No injection. Follow whatever really moves (wave the toy in front of
  the board) and report the tracked trajectory. All outputs go to tmp/.

    python tools/verify_tracking_hik.py
    python tools/verify_tracking_hik.py --live --frames 120
    python tools/verify_tracking_hik.py --exposure 5000 --speed 600
    python tools/verify_tracking_hik.py --occlude-start 25 --occlude-frames 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np

from irisloop.hik_camera import HikCamera
from irisloop.moving_target import MotionDetector, MovingObjectTracker

OUT_DIR = "tmp"
REPORT = os.path.join(OUT_DIR, "hik_tracking_report.json")
PREVIEW = os.path.join(OUT_DIR, "hik_tracking_verify.png")


def _statistics(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "median_px": round(float(np.median(array)), 2),
        "p95_px": round(float(np.percentile(array, 95)), 2),
        "max_px": round(float(array.max()), 2),
    }


def _grab(cam: HikCamera, timeout_ms: int = 2000):
    ok, frame = cam.read(timeout_ms)
    if not ok or frame is None:
        return None
    return frame.copy()


def _inject(frame: np.ndarray, centre, size: int, dot_offset: float, rng) -> None:
    x, y = int(round(centre[0])), int(round(centre[1]))
    half = size // 2
    patch = rng.integers(170, 214, (size, size), dtype=np.uint8)
    region = frame[y - half:y + half, x - half:x + half]
    if region.ndim == 3:
        patch = np.dstack([patch] * region.shape[2])
    region[:] = patch
    colour = 255 if frame.ndim == 2 else (255, 255, 255)
    cv2.circle(frame, (x + int(dot_offset), y), 10, colour, -1)


def _annotate(frame, target, aim, truth, size):
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    if truth is not None:
        x, y = int(truth[0]), int(truth[1])
        half = size // 2
        cv2.rectangle(vis, (x - half, y - half), (x + half, y + half), (255, 0, 0), 2)
    if target is not None:
        x0, y0, x1, y1 = target.bbox
        colour = (0, 255, 0) if target.measured else (0, 165, 255)
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), colour, 2)
        cv2.drawMarker(vis, (int(target.center[0]), int(target.center[1])),
                       colour, cv2.MARKER_CROSS, 20, 2)
    if aim is not None:
        cv2.drawMarker(vis, (int(aim[0]), int(aim[1])), (0, 0, 255),
                       cv2.MARKER_TILTED_CROSS, 20, 2)
    return vis


def run(args) -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    cam = HikCamera(args.mvs_dir)
    cam.enumerate()
    cam.open(serial=args.serial)
    cam.set_gain(args.gain)
    cam.set_exposure_us(args.exposure)
    cam.flush(3)
    cam_info = cam.info()
    print(f"  camera: {cam_info}")
    print(f"  exposure={cam.get_exposure_us()}us gain={cam.get_gain():.1f}dB")

    detector = MotionDetector(min_area=args.min_area, roi=None)
    tracker = MovingObjectTracker(
        detector, max_lost=args.max_lost, smoothing=args.smoothing,
    )
    raw_hits: list = []
    original_detect = detector.detect

    def recording_detect(frame, hint=None):
        hit = original_detect(frame, hint=hint)
        raw_hits.append(hit)
        return hit

    detector.detect = recording_detect
    rng = np.random.default_rng(20261111)

    static_false_positives = 0
    static_frames = 0
    try:
        for i in range(args.warmup):
            frame = _grab(cam)
            if frame is None:
                continue
            found = detector.candidates(frame)
            if i >= args.warmup // 2:  # skip MOG2's first learning frames
                static_frames += 1
                static_false_positives += int(bool(found))
        print(f"  static scene: {static_false_positives}/{static_frames} frames with a false detection")

        errors: list[float] = []
        raw_errors: list[float] = []
        lead_errors: list[float] = []
        hijacks = 0
        ghost_hits = 0
        reacquired = False
        detected = 0
        coasted = 0
        lost = 0
        speeds: list[float] = []
        centres: list[tuple[float, float]] = []
        occluded_count = (
            min(args.occlude_frames, max(0, args.frames - args.occlude_start))
            if args.occlude_start is not None else 0
        )
        start = time.perf_counter()
        previous = start
        last_frame = None
        last_target = None
        last_aim = None
        last_truth = None

        for index in range(args.frames):
            frame = _grab(cam)
            if frame is None:
                print(f"  frame {index}: timeout")
                continue
            now = time.perf_counter()
            elapsed = now - start
            # Real frame interval, measured from the previous grab.
            dt = (1.0 / args.assumed_fps if index == 0
                  else min(max(now - previous, 1e-3), 0.5))
            previous = now

            occluded = (
                args.occlude_start is not None
                and args.occlude_start <= index < args.occlude_start + args.occlude_frames
            )
            truth = None
            if not args.live:
                truth = (
                    args.start[0] + args.speed * elapsed + args.size / 2,
                    args.start[1] + args.size / 2,
                )
                if not occluded:
                    _inject(frame, truth, args.size, args.dot_offset, rng)

            target = tracker.update(frame, dt=dt)
            aim = tracker.lead(dt) if target is not None else None
            speeds.append(float(np.hypot(*tracker.velocity)))
            if target is not None:
                centres.append(target.center)

            if target is None:
                lost += 1
            elif not target.measured:
                coasted += 1
            else:
                detected += 1
                if occluded:
                    ghost_hits += 1  # a detection while the object is hidden
                if args.occlude_start is not None and index >= (
                    args.occlude_start + args.occlude_frames
                ) and not reacquired:
                    reacquired = True
                if truth is not None and not occluded:
                    errors.append(float(np.hypot(
                        target.center[0] - truth[0], target.center[1] - truth[1])))
                    hit = raw_hits[-1]
                    if hit is not None:
                        raw_errors.append(float(np.hypot(
                            hit.center[0] - truth[0], hit.center[1] - truth[1])))
                        # A dot-sized hit far from the truth means our fake laser won.
                        if hit.area < args.min_area * 3 and abs(
                            hit.center[0] - truth[0]
                        ) > args.dot_offset / 2:
                            hijacks += 1
                    if aim is not None:
                        # Compare the lead against where the object WILL be one
                        # frame later, not against the current-frame truth.
                        future_x = truth[0] + args.speed * dt
                        lead_errors.append(float(np.hypot(
                            aim[0] - future_x, aim[1] - truth[1])))
            last_frame, last_target, last_aim, last_truth = frame, target, aim, truth

        if last_frame is not None:
            cv2.imwrite(PREVIEW, _annotate(
                last_frame, last_target, last_aim, last_truth, args.size,
            ))
    finally:
        cam.close()

    report = {
        "mode": "live" if args.live else "injected",
        "camera": cam_info,
        "frames": args.frames,
        "tracked_centres": len(centres),
        "static_false_positives": static_false_positives,
        "static_frames": static_frames,
        "detected": detected,
        "coasted": coasted,
        "lost": lost,
        "fake_laser_hijacks": hijacks,
        "occluded_frames": occluded_count,
        "ghost_detections_during_occlusion": ghost_hits,
        "reacquired_after_occlusion": reacquired,
        "filtered_centre_error": _statistics(errors),
        "raw_detector_error": _statistics(raw_errors),
        "one_frame_lead_error_vs_future_truth": _statistics(lead_errors),
        "estimated_speed_px_s": _statistics(speeds),
        "expected_speed_px_s": None if args.live else args.speed,
        "centre_range_px": ({
            "x": [round(min(c[0] for c in centres), 1), round(max(c[0] for c in centres), 1)],
            "y": [round(min(c[1] for c in centres), 1), round(max(c[1] for c in centres), 1)],
        } if centres else None),
        "preview": PREVIEW,
    }
    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.live:
        return 0
    visible = max(1, args.frames - occluded_count)
    ok = (
        args.frames > 0
        and (detected - ghost_hits) / visible >= 0.9
        and hijacks == 0
        and ghost_hits <= occluded_count // 2
        and (not occluded_count or reacquired)
        and static_false_positives <= max(1, static_frames // 5)
        and (not errors or np.percentile(errors, 95) <= args.max_error)
    )
    print("=== VERDICT:", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verify_tracking_hik", description=__doc__)
    ap.add_argument("--serial", default="K65564865")
    ap.add_argument("--mvs-dir", default=None)
    ap.add_argument("--exposure", type=float, default=10000.0, help="exposure in us")
    ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--min-area", type=float, default=300.0)
    ap.add_argument("--max-lost", type=int, default=8)
    ap.add_argument("--smoothing", type=float, default=0.6)
    ap.add_argument("--assumed-fps", type=float, default=30.0,
                    help="dt for the very first frame, before a real interval exists")
    ap.add_argument("--max-error", type=float, default=25.0,
                    help="allowed p95 centre error in camera px")
    ap.add_argument("--start", type=float, nargs=2, default=(260.0, 460.0),
                    metavar=("X", "Y"), help="start position of the injected square")
    ap.add_argument("--size", type=int, default=60)
    ap.add_argument("--speed", type=float, default=400.0, help="injected px/s")
    ap.add_argument("--dot-offset", type=float, default=120.0,
                    help="fake laser spot offset from the object centre")
    ap.add_argument("--occlude-start", type=int, default=None,
                    help="frame index at which to hide the injected object")
    ap.add_argument("--occlude-frames", type=int, default=5,
                    help="how many frames the object stays hidden")
    ap.add_argument("--live", action="store_true", help="track real motion, no injection")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except RuntimeError as exc:
        print(f"[error] {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
