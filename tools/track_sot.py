#!/usr/bin/env python
"""Track one user-selected object with UETrack-S and a constant-velocity Kalman filter.

Drag a box around the object. The green box is UETrack-S and the yellow cross
is the filtered center. The projector dot is aimed at that yellow cross. A
missed measurement or a point outside the light field sends an all-black frame.

    python tools/track_sot.py --hik
    python tools/track_sot.py --hik --skip-project
    python tools/track_sot.py --video clip.mp4 --skip-project
    python tools/track_sot.py --demo

q / Esc quits. r clears the selection and turns the beam off.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
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

from irisloop.calibrate import DEFAULT_OUT, Calibration, _black_image, _project
from irisloop.camera import UsbCamera
from irisloop.live_projection import LatestMask
from irisloop.moving_target import camera_to_projector, dot_mask
from irisloop.session import open_slot
from irisloop.single_object import SingleObjectTracker, aim_camera_point
from irisloop.uetrack_s import UETrackS, default_checkpoint


class _Drag:
    def __init__(self, scale: float) -> None:
        self.scale = scale
        self.anchor: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        self.ready: tuple[float, float, float, float] | None = None

    def _image_xy(self, x: int, y: int) -> tuple[int, int]:
        return int(round(x / self.scale)), int(round(y / self.scale))

    def on_mouse(self, event, x, y, flags, param) -> None:
        del flags, param
        point = self._image_xy(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.anchor = point
            self.current = point
            self.ready = None
        elif event == cv2.EVENT_MOUSEMOVE and self.anchor is not None:
            self.current = point
        elif event == cv2.EVENT_LBUTTONUP and self.anchor is not None:
            self.current = point
            box = self.box()
            self.anchor = None
            self.current = None
            if box is not None:
                self.ready = box

    def box(self) -> tuple[float, float, float, float] | None:
        if self.anchor is None or self.current is None:
            return None
        x0, y0 = self.anchor
        x1, y1 = self.current
        x, y = min(x0, x1), min(y0, y1)
        width, height = abs(x1 - x0), abs(y1 - y0)
        if width < 8 or height < 8:
            return None
        return (float(x), float(y), float(width), float(height))


def _draw(frame, track, draft, status: str, aim=None) -> np.ndarray:
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    if draft is not None:
        x, y, width, height = draft
        cv2.rectangle(
            vis, (int(x), int(y)), (int(x + width), int(y + height)),
            (255, 180, 0), 1,
        )
    if track is not None:
        if not track.measured:
            rx, ry, rw, rh = track.raw_bbox
            cv2.rectangle(
                vis, (int(rx), int(ry)), (int(rx + rw), int(ry + rh)),
                (0, 0, 255), 1,
            )
        x, y, width, height = track.bbox
        color = (0, 220, 0) if track.measured else (0, 165, 255)
        cv2.rectangle(
            vis, (int(x), int(y)), (int(x + width), int(y + height)), color, 2,
        )
        cx, cy = int(round(track.center[0])), int(round(track.center[1]))
        cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        tip = (
            int(round(track.center[0] + track.velocity[0] * 0.25)),
            int(round(track.center[1] + track.velocity[1] * 0.25)),
        )
        cv2.arrowedLine(vis, (cx, cy), tip, (0, 255, 255), 2, tipLength=0.25)
    if aim is not None:
        cv2.drawMarker(
            vis, (int(round(aim[0])), int(round(aim[1]))),
            (0, 0, 255), cv2.MARKER_CROSS, 22, 2,
        )
    cv2.putText(vis, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(
        vis, "drag a box    r = reset    q = quit",
        (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
    )
    return vis


def _open_source(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {args.video}")

        def read():
            ok, frame = cap.read()
            return ok, frame

        return cap, read, cap.release

    if args.hik:
        from irisloop.hik_camera import HikCamera

        cam = HikCamera(args.mvs_dir)
        devices = cam.enumerate()
        serial = args.serial
        if serial is None:
            usb = [item for item in devices if item["interface"] == "USB3"]
            if usb:
                serial = usb[0]["serial"]
                print(f"  using {usb[0]['model']} sn={serial}", flush=True)
            elif not devices:
                raise RuntimeError("no Hik camera found")
        cam.open(serial=serial)
        cam.set_gain(args.gain)
        cam.set_exposure_us(args.exposure_us)
        cam.flush(3)
        print(f"  camera: {cam.info()}", flush=True)
        return cam, lambda: cam.read(200), cam.close

    cam = UsbCamera(args.camera, args.cam_width, args.cam_height)
    cam.open()
    print(f"  camera: {cam.actual_size[0]}x{cam.actual_size[1]}", flush=True)
    return cam, cam.read, cam.release


def _marked_patch(seed: int, mark: str, side: int = 72) -> np.ndarray:
    rng = np.random.default_rng(seed)
    patch = rng.integers(0, 255, size=(side, side, 3), dtype=np.uint8)
    if mark == "red":
        patch[8:28, 8:side - 8] = (40, 40, 230)
    else:
        patch[8:side - 8, side // 2 - 8:side // 2 + 8] = (40, 200, 40)
    return patch


def _paste(frame: np.ndarray, patch: np.ndarray, x: float, y: float) -> None:
    height, width = frame.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = x0 + patch.shape[1], y0 + patch.shape[0]
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x1), min(height, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    frame[sy0:sy1, sx0:sx1] = patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]


def _background() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(20, 50, size=(360, 640, 3), dtype=np.uint8)


def _clip_box(x: float, y: float, side: float, width: int = 640, height: int = 360):
    side = float(max(16.0, side))
    x = float(np.clip(x, 0, width - side))
    y = float(np.clip(y, 0, height - side))
    return x, y, side, side


def _demo_cases():
    """Synthetic motions with a known box. Each case returns frames and centers."""
    background = _background()
    target = _marked_patch(1, "red")
    other = _marked_patch(2, "green")
    cases = []

    def add(name, samples, limit, note):
        frames, centers, inits = [], [], []
        for sample in samples:
            frame = background.copy()
            for patch, box in sample["draw"]:
                _paste(frame, patch, box[0], box[1])
            frames.append(frame)
            centers.append(sample["center"])
            inits.append(sample["init"])
        cases.append({
            "name": name,
            "frames": frames,
            "centers": centers,
            "init": inits[0],
            "limit": limit,
            "note": note,
        })

    straight = []
    for index in range(30):
        box = _clip_box(40 + index * 10, 140, 72)
        straight.append({
            "draw": [(target, box)],
            "center": (box[0] + 36, box[1] + 36),
            "init": box,
        })
    add("straight", straight, 20.0, "constant velocity to the right")

    diagonal = []
    for index in range(30):
        box = _clip_box(50 + index * 8, 40 + index * 5, 72)
        diagonal.append({
            "draw": [(target, box)],
            "center": (box[0] + 36, box[1] + 36),
            "init": box,
        })
    add("diagonal", diagonal, 25.0, "right and down together")

    reverse = []
    for index in range(28):
        if index <= 14:
            x = 70 + index * 12
        else:
            x = 70 + 14 * 12 - (index - 14) * 12
        box = _clip_box(x, 140, 72)
        reverse.append({
            "draw": [(target, box)],
            "center": (box[0] + 36, box[1] + 36),
            "init": box,
        })
    add("reverse", reverse, 30.0, "right, then back left")

    stopped = []
    for index in range(30):
        moved = min(index, 12)
        box = _clip_box(60 + moved * 10, 160, 72)
        stopped.append({
            "draw": [(target, box)],
            "center": (box[0] + 36, box[1] + 36),
            "init": box,
        })
    add("stop", stopped, 25.0, "move, then hold still")

    gap = []
    for index in range(30):
        box = _clip_box(48 + index * 8, 150, 72)
        visible = not 12 <= index <= 16
        gap.append({
            "draw": [(target, box)] if visible else [],
            "center": (box[0] + 36, box[1] + 36),
            "init": box,
        })
    add("gap", gap, 40.0, "hidden for five frames, then back")

    crowded = []
    for index in range(40):
        box = _clip_box(80 + index * 6, 150, 72)
        rival = _clip_box(400 - index * 8, 150, 72)
        crowded.append({
            "draw": [(other, rival), (target, box)],
            "center": (box[0] + 36, box[1] + 36),
            "init": box,
        })
    add("distractor", crowded, 30.0, "a different patch crosses the target")

    scaled = []
    for index in range(24):
        side = 64 + index * 2
        box = _clip_box(80 + index * 6, 120, side)
        patch = cv2.resize(target, (int(box[2]), int(box[3])), interpolation=cv2.INTER_LINEAR)
        scaled.append({
            "draw": [(patch, box)],
            "center": (box[0] + box[2] / 2, box[1] + box[3] / 2),
            "init": box,
        })
    add("scale", scaled, 30.0, "patch grows while moving right")
    return cases


def _run_case(measurer, min_score: float, case) -> dict:
    tracker = SingleObjectTracker(measurer, min_score=min_score)
    frames = case["frames"]
    truth = case["centers"]
    started = time.perf_counter()
    track = tracker.start(frames[0], case["init"])
    errors = [float(np.hypot(track.center[0] - truth[0][0], track.center[1] - truth[0][1]))]
    scores = [track.score]
    lost_at = None
    for index, frame in enumerate(frames[1:], start=1):
        track = tracker.update(frame, 1.0 / 30.0)
        if track is None:
            lost_at = index
            break
        errors.append(float(np.hypot(
            track.center[0] - truth[index][0], track.center[1] - truth[index][1],
        )))
        scores.append(track.score)
    elapsed = time.perf_counter() - started
    speed = 0.0 if track is None else float(np.hypot(*track.velocity))
    return {
        "name": case["name"],
        "note": case["note"],
        "limit": case["limit"],
        "frames": len(scores),
        "error": errors[-1],
        "mean": float(np.mean(errors)),
        "score_min": min(scores),
        "score_last": scores[-1],
        "speed": speed,
        "measured": False if track is None else track.measured,
        "lost_at": lost_at,
        "fps": len(scores) / elapsed if elapsed else 0.0,
        "preview": None if track is None else _draw(
            frames[len(scores) - 1], track, None,
            f"{case['name']} err={errors[-1]:.0f}px",
        ),
        "passed": lost_at is None and errors[-1] <= case["limit"],
    }


def run_demo(args) -> int:
    measurer = UETrackS(args.checkpoint, device=args.device)
    results = [_run_case(measurer, args.min_score, case) for case in _demo_cases()]
    os.makedirs("tmp", exist_ok=True)
    print(
        f"{'case':<12} {'err':>7} {'mean':>7} {'limit':>7} "
        f"{'score':>7} {'speed':>8} {'fps':>6}  result",
        flush=True,
    )
    previews = []
    failed = []
    for item in results:
        label = "ok" if item["passed"] else "FAIL"
        if item["lost_at"] is not None:
            label = f"FAIL lost@{item['lost_at']}"
        print(
            f"{item['name']:<12} {item['error']:7.1f} {item['mean']:7.1f} "
            f"{item['limit']:7.1f} {item['score_last']:7.3f} {item['speed']:8.1f} "
            f"{item['fps']:6.1f}  {label}  {item['note']}",
            flush=True,
        )
        if not item["passed"]:
            failed.append(item["name"])
        if item["preview"] is not None:
            previews.append(item["preview"])
    if previews:
        cv2.imwrite(os.path.join("tmp", "sot_demo.png"), np.vstack(previews))
    if failed:
        print("failed: " + ", ".join(failed), flush=True)
        return 1
    return 0


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
    camera_matrix, distortion = data["K_c"], data["dist_c"]
    depth = procalib.reference_depth(field)

    def mapper(points):
        mapped = []
        for x, y in points:
            uv = procalib.proj_pixel_for_camera(
                (float(x), float(y)), camera_matrix, distortion, field, z=depth,
            )
            mapped.append(uv if uv is not None else (np.nan, np.nan))
        return np.asarray(mapped, dtype=np.float64)

    return mapper


def _kill_close(opened) -> None:
    try:
        asyncio.run(opened.close())
    except Exception:
        pass


async def run_live(args) -> int:
    measurer = UETrackS(args.checkpoint, device=args.device)
    tracker = SingleObjectTracker(measurer, min_score=args.min_score)
    mapper = None
    if not args.skip_project:
        mapper = (
            _homography_mapper(args.calibration) if args.flat
            else _ray_field_mapper(args.procalib)
        )
    blank = _black_image()
    opened = None
    output: LatestMask | None = None
    close = None
    upload_s = 1.0
    try:
        source, read, close = _open_source(args)
        if mapper is not None:
            await asyncio.sleep(0.5)
            opened = await open_slot(args.address)
            atexit.register(_kill_close, opened)
            print(f"=== CONNECT {opened.address} ===", flush=True)

            async def send(mask: np.ndarray) -> None:
                nonlocal upload_s
                started = time.perf_counter()
                await _project(opened.client, mask, settle=0.0)
                upload_s = time.perf_counter() - started
                print(f"  upload={upload_s:.2f}s", flush=True)

            output = LatestMask(send)
            await _project(opened.client, blank, settle=0.2)
        drag = _Drag(args.scale)
        window = "track_sot"
        if not args.no_preview:
            cv2.namedWindow(window)
            cv2.setMouseCallback(window, drag.on_mouse)
        previous = time.perf_counter()
        frames = 0
        track = None
        selected = False
        previous_velocity = None
        print("drag a box around one object. q quits, r resets.", flush=True)
        while True:
            if output is not None:
                output.pump()
            ok, frame = await asyncio.to_thread(read)
            if not ok or frame is None:
                break
            now = time.perf_counter()
            dt = now - previous
            previous = now
            fresh = False
            if drag.ready is not None:
                box = drag.ready
                drag.ready = None
                track = tracker.start(frame, box)
                selected = True
                fresh = True
                previous_velocity = None
                print(f"  selected {tuple(round(v) for v in box)}", flush=True)
            elif tracker.active:
                track = tracker.update(frame, dt)
            if track is not None and track.measured:
                if args.lead is not None and args.lead > 0:
                    aim = aim_camera_point(
                        track.center, track.velocity,
                        lead_s=args.lead, max_lead_px=args.max_lead_px,
                        measured=True, previous_velocity=previous_velocity,
                    )
                    previous_velocity = track.velocity
                else:
                    aim = track.center
            else:
                aim = None
                if track is None:
                    previous_velocity = None
            proj = camera_to_projector(aim, mapper) if aim is not None and mapper is not None else None
            mask = dot_mask(proj, args.radius) if proj is not None else None
            if output is not None:
                output.submit(mask if mask is not None else blank)
            if track is None:
                status = "draw a box" if not selected else "lost; beam off"
            elif not track.measured:
                status = f"coast score={track.score:.2f}; beam off"
            elif mapper is not None and mask is None:
                status = (
                    f"({track.center[0]:.0f},{track.center[1]:.0f}) "
                    f"score={track.score:.2f} outside light field; beam off"
                )
            else:
                aimed = ""
                if proj is not None:
                    aimed = f" -> proj ({proj[0]:.0f},{proj[1]:.0f})"
                lead = f" lead={args.lead:.1f}s" if args.lead else ""
                status = (
                    f"({track.center[0]:.0f},{track.center[1]:.0f}) "
                    f"v=({track.velocity[0]:.0f},{track.velocity[1]:.0f}) "
                    f"score={track.score:.2f}{lead}{aimed}"
                )
            if not args.no_preview:
                vis = _draw(frame, track, drag.box(), status, aim)
                if track is not None and frames % 20 == 0:
                    os.makedirs("tmp", exist_ok=True)
                    cv2.imwrite(os.path.join("tmp", "sot_live.png"), vis)
                    print(f"  {status}", flush=True)
                if args.scale != 1.0:
                    vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale)
                cv2.imshow(window, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    tracker.reset()
                    drag.ready = None
                    track = None
                    selected = False
                    previous_velocity = None
                    if output is not None:
                        output.submit(blank)
            frames += 1
            if args.max_frames and frames >= args.max_frames:
                break
        uploads = output.uploads if output is not None else 0
        print(f"  frames={frames} uploads={uploads}", flush=True)
        return 0
    finally:
        if not args.no_preview:
            cv2.destroyAllWindows()
        if close is not None:
            close()
        if output is not None:
            try:
                await output.finish()
            finally:
                try:
                    await _project(opened.client, blank, settle=0.2)
                finally:
                    await opened.close()
        elif opened is not None:
            await opened.close()


def run(args) -> int:
    if args.demo:
        return run_demo(args)
    try:
        return asyncio.run(run_live(args))
    except (RuntimeError, OSError) as exc:
        print(f"[error] {exc}", flush=True)
        return 2
    except Exception as exc:
        if exc.__class__.__module__.startswith("bleak"):
            print(f"[error] {exc}", flush=True)
            return 2
        raise


def main() -> None:
    parser = argparse.ArgumentParser(prog="track_sot", description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None, help="cuda or cpu; default is cuda when present")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cam-width", type=int, default=1280)
    parser.add_argument("--cam-height", type=int, default=720)
    parser.add_argument("--video", default=None)
    parser.add_argument("--hik", action="store_true")
    parser.add_argument("--mvs-dir", default=None)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--gain", type=float, default=8.0)
    parser.add_argument("--exposure-us", type=float, default=10000.0)
    parser.add_argument("--scale", type=float, default=0.7)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--demo", action="store_true",
        help="synthetic motions with known boxes, no camera and no projector",
    )
    parser.add_argument("--skip-project", action="store_true", help="camera and tracker only")
    parser.add_argument("--address", default=None, help="IrisGreen MAC; omit when only one advertises")
    parser.add_argument("--calibration", default=DEFAULT_OUT, help="flat-wall homography, used with --flat")
    parser.add_argument(
        "--procalib",
        default=os.path.join("captures", "procalib", "run", "procalib.npz"),
        help="ray-field calibration used for the dot",
    )
    parser.add_argument("--flat", action="store_true", help="use the flat-wall homography instead of the ray field")
    parser.add_argument("--radius", type=int, default=12, help="dot radius in projector pixels")
    parser.add_argument(
        "--lead", type=float, default=None,
        help="seconds to aim ahead of the yellow cross; default is the yellow cross itself",
    )
    parser.add_argument(
        "--max-lead-px", type=float, default=120.0,
        help="cap the extrapolated lead in camera pixels (0 disables the cap)",
    )
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = default_checkpoint()
    if args.radius < 1:
        parser.error("--radius must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
