#!/usr/bin/env python
"""Follow a moving object and keep one IrisGreen laser dot on it.

    camera -> moving-object tracker -> camera centre -> calibration
    -> 640x480 1bpp dot -> group 1 over BLE -> camera

A full frame upload takes seconds, so only the newest dot is sent and --lead
aims ahead of the measured position by the expected latency. The projector is
connected once for the whole process; group 1 is cleared once at startup and
an all-black bitmap is left playing on exit (stop() falls back to the idle
animation, it is not laser-off).

    python tools/track_dot.py
    python tools/track_dot.py --roi 200 100 900 600 --lead 1.0
    python tools/track_dot.py --procalib captures/procalib/run/procalib.npz --hik
    python tools/track_dot.py --skip-project --max-frames 120   # camera only

q / Esc quits. Objects must move roughly in the calibrated plane: the flat-wall
homography cannot compensate for depth changes, and the ray-field mapper uses
one fixed reference depth.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import cv2
import numpy as np

from irisloop.calibrate import (
    DEFAULT_OUT,
    Calibration,
    _black_image,
    _open_camera as _open_usb_camera,
    _project,
)
from irisloop.live_projection import LatestMask
from irisloop.moving_target import (
    MotionDetector,
    MovingObjectTracker,
    camera_to_projector,
    dot_mask,
)
from irisloop.session import open_slot


def _homography_mapper(path: str):
    if not os.path.isfile(path):
        raise RuntimeError(f"missing {path}; run python -m irisloop.calibrate run first")
    return Calibration.load(path).cam_to_proj


def _ray_field_mapper(path: str):
    """Camera pixel -> projector pixel on the procalib reference-depth plane."""
    if not os.path.isfile(path):
        raise RuntimeError(f"missing {path}; run python -m irisloop.procalib solve first")
    from irisloop import procalib

    data = np.load(path)
    field = {
        "grid_pts": data["ray_grid_pts"],
        "origins": data["ray_origins"],
        "dirs": procalib._dirs_toward_scene(data["ray_dirs"]),
    }
    K, dist = data["K_c"], data["dist_c"]
    depth = procalib.reference_depth(field)

    def mapper(points):
        mapped = []
        for x, y in points:
            uv = procalib.proj_pixel_for_camera(
                (float(x), float(y)), K, dist, field, z=depth,
            )
            mapped.append(uv if uv is not None else (np.nan, np.nan))
        return np.asarray(mapped, dtype=np.float64)

    return mapper


def _open_camera(args):
    """Return (camera, read_callable). Both camera kinds yield (ok, frame)."""
    if args.hik:
        from irisloop.hik_camera import HikCamera

        cam = HikCamera(args.mvs_dir)
        cam.enumerate()
        cam.open(serial=args.serial)
        cam.set_gain(args.gain)
        cam.set_exposure_us(args.exposure_us)
        cam.flush(3)
        print(f"  camera: {cam.info()}", flush=True)
        return cam, lambda: cam.read(200)

    cam = _open_usb_camera(args.camera, args.cam_width, args.cam_height)
    cam.set_exposure(args.exposure)
    print(f"  camera: {cam.info()}", flush=True)
    return cam, cam.read


def _kill_close(opened) -> None:
    """Last-ditch BLE close if the process dies without cleanup."""
    try:
        asyncio.run(opened.close())
    except Exception:
        pass


def _draw_preview(frame: np.ndarray, target, aim, status: str) -> np.ndarray:
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    if aim is not None:
        cv2.drawMarker(
            vis, (int(round(aim[0])), int(round(aim[1]))),
            (0, 0, 255), cv2.MARKER_CROSS, 24, 2,
        )
    if target is not None:
        x0, y0, x1, y1 = target.bbox
        color = (0, 255, 0) if target.measured else (0, 165, 255)
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
    cv2.putText(vis, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    cv2.putText(vis, "q / Esc = quit", (16, 62), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 1)
    return vis


async def run(args) -> int:
    mapper = _ray_field_mapper(args.procalib) if args.procalib else _homography_mapper(args.calibration)
    detector = MotionDetector(
        min_area=args.min_area,
        min_fill=args.min_fill,
        roi=tuple(args.roi) if args.roi else None,
        laser_green_excess=args.laser_green_excess,
    )
    tracker = MovingObjectTracker(
        detector, max_lost=args.max_lost, smoothing=args.smoothing,
    )
    blank = _black_image()
    opened = None
    cam = None
    output: LatestMask | None = None
    trace = None
    if args.trace:
        os.makedirs(args.trace, exist_ok=True)
        trace = open(os.path.join(args.trace, "trace.jsonl"), "w", encoding="utf-8")

    try:
        if not args.skip_project:
            await asyncio.sleep(1.0)  # let the Bluetooth stack settle after scanning
            opened = await open_slot(args.address)
            atexit.register(_kill_close, opened)
            print(f"=== CONNECT {opened.address} ===", flush=True)

            async def send(mask: np.ndarray) -> None:
                started = time.perf_counter()
                await _project(opened.client, mask, settle=0.0)
                print(f"  upload={time.perf_counter() - started:.2f}s", flush=True)

            output = LatestMask(send)

        cam, read = _open_camera(args)
        if output is not None:
            await _project(opened.client, blank, settle=0.4)  # truly dark warm-up
        for _ in range(args.warmup):
            ok, warm = await asyncio.to_thread(read)
            if ok and warm is not None:
                detector.candidates(warm)  # teach MOG2 the static background

        if not args.no_preview:
            cv2.namedWindow("track_dot")

        started = time.perf_counter()
        previous = started
        frames = 0
        print("tracking; q / Esc to quit. uploads are seconds-scale, not camera-rate",
              flush=True)
        while True:
            if output is not None:
                output.pump()
            ok, frame = await asyncio.to_thread(read)
            now = time.perf_counter()
            dt = min(max(now - previous, 1e-3), 0.5)
            previous = now
            if not ok or frame is None:
                tracker.reset()
                if output is not None:
                    output.submit(blank)
                continue

            target = tracker.update(frame, dt=dt)
            aim = tracker.lead(args.lead) if args.lead > 0 and target is not None else None
            if aim is None and target is not None:
                aim = target.center
            if aim is not None and target is not None and args.max_lead_px > 0:
                dx = aim[0] - target.center[0]
                dy = aim[1] - target.center[1]
                distance = float(np.hypot(dx, dy))
                if distance > args.max_lead_px:
                    scale = args.max_lead_px / distance
                    aim = (target.center[0] + dx * scale,
                           target.center[1] + dy * scale)
            proj = camera_to_projector(aim, mapper) if aim is not None else None
            mask = dot_mask(proj, args.radius)
            if output is not None:
                output.submit(mask if mask is not None else blank)

            if target is None:
                status = "no object; beam dark"
            elif mask is None:
                status = "target outside light field; beam dark"
            elif not target.measured:
                status = f"coasting ({tracker.lost_frames})"
            else:
                status = f"object ({target.center[0]:.0f},{target.center[1]:.0f}) -> proj ({proj[0]:.0f},{proj[1]:.0f})"

            if args.log_every and frames % args.log_every == 0:
                print(f"  [{frames:04d}] {status} uploads={output.uploads if output else 0}",
                      flush=True)
            if args.trace:
                trace.write(json.dumps({
                    "t": round(now - started, 3),
                    "frame": frames,
                    "measured": bool(target and target.measured),
                    "camera": list(target.center) if target else None,
                    "aim": list(aim) if aim else None,
                    "proj": list(proj) if proj else None,
                    "lit": mask is not None,
                    "uploads": output.uploads if output else 0,
                }) + "\n")
                if frames % args.trace_every == 0:
                    cv2.imwrite(
                        os.path.join(args.trace, f"frame_{frames:05d}.png"),
                        _draw_preview(frame, target, aim, status),
                    )
            if not args.no_preview:
                vis = _draw_preview(frame, target, aim, status)
                if args.preview_scale != 1.0:
                    vis = cv2.resize(
                        vis, None, fx=args.preview_scale, fy=args.preview_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow("track_dot", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
            if args.seconds and now - started >= args.seconds:
                break
        print(f"  frames={frames} uploads={output.uploads if output else 0}", flush=True)
        return 0
    finally:
        if trace is not None:
            trace.close()
        if not args.no_preview:
            cv2.destroyAllWindows()
        if cam is not None:
            if hasattr(cam, "release"):
                cam.release()
            elif hasattr(cam, "close"):
                cam.close()
        if output is not None:
            try:
                # Let any in-flight upload finish before pushing the final black.
                await output.finish()
            finally:
                try:
                    await _project(opened.client, blank, settle=0.2)
                finally:
                    await opened.close()
        elif opened is not None:
            await opened.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="track_dot", description=__doc__)
    ap.add_argument("--address", default=None, help="IrisGreen MAC; omit when only one advertises")
    ap.add_argument("--calibration", default=DEFAULT_OUT, help="flat-wall homography .npz")
    ap.add_argument("--procalib", default=None,
                    help="use the procalib ray field instead of the homography")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--exposure", type=float, default=-7.0, help="USB camera EV")
    ap.add_argument("--hik", action="store_true", help="use the Hikrobot MVS camera")
    ap.add_argument("--serial", default="K65564865")
    ap.add_argument("--mvs-dir", default=None)
    ap.add_argument("--exposure-us", type=float, default=40000.0)
    ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--roi", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="restrict detection to a camera-pixel rectangle")
    ap.add_argument("--min-area", type=float, default=300.0)
    ap.add_argument("--min-fill", type=float, default=0.20,
                    help="min contour fill ratio; lower it when a moving object breaks into fragments")
    ap.add_argument("--max-lead-px", type=float, default=120.0,
                    help="cap the extrapolated lead in camera px (0 disables)")
    ap.add_argument("--laser-green-excess", type=float, default=28.0,
                    help="green dominance used to reject our own dot; 0 disables")
    ap.add_argument("--radius", type=int, default=12, help="dot radius in projector px")
    ap.add_argument("--lead", type=float, default=0.0,
                    help="aim this many seconds ahead to offset BLE latency")
    ap.add_argument("--warmup", type=int, default=25, help="background frames for MOG2")
    ap.add_argument("--max-lost", type=int, default=8)
    ap.add_argument("--smoothing", type=float, default=0.6)
    ap.add_argument("--skip-project", action="store_true",
                    help="camera + tracker only, no BLE connection")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = run until quit")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after this many seconds (0 = until quit / max-frames)")
    ap.add_argument("--log-every", type=int, default=30,
                    help="print a status line every N frames (0 = silent)")
    ap.add_argument("--trace", default=None,
                    help="directory for trace.jsonl and sampled annotated frames")
    ap.add_argument("--trace-every", type=int, default=30,
                    help="save one annotated frame every N frames when --trace is set")
    ap.add_argument("--preview-scale", type=float, default=0.75,
                    help="scale factor for the live preview window")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args(argv)
    if args.radius < 1:
        ap.error("--radius must be positive")
    if args.roi and (args.roi[2] <= args.roi[0] or args.roi[3] <= args.roi[1]):
        ap.error("--roi needs X1>X0 and Y1>Y0")
    try:
        return asyncio.run(run(args))
    except (RuntimeError, OSError) as exc:
        print(f"[error] {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
