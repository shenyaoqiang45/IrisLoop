"""Draw a freehand region on the camera view and fill it with projector light.

Same gesture as the old 2D lasso (tools/play_video_at.py): hold, trace any
closed shape, release. Textured selections follow in-plane object motion.
The mask is warped with a homography fitted at the reference depth; depth
changes and tilt are not supported. BLE uploads lag behind camera tracking.

    python tools/aim3d.py
    python tools/aim3d.py --at 640 512
    python tools/aim3d.py --static
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import cv2
import numpy as np

from irisloop.ble_client import IrisBleClient
from irisloop.hik_camera import HikCamera
from irisloop.image_pack import HEIGHT, WIDTH
from irisloop.region_tracker import RegionTracker, TrackingLost
from irisloop.procalib import (
    _black_image,
    _dirs_toward_scene,
    _project,
    detect_grid_dots,
    plane_homography_from_rays,
    proj_pixel_for_camera,
    ray_field_lookup,
    reference_depth,
    shape_light_mask,
    triangulate_rayfield,
)

async def open_projector(address: str | None):
    """Connect through the session layer and clear the boot slot once."""
    from irisloop.session import open_slot

    return await open_slot(address)


CALIB = os.path.join("captures", "procalib", "run", "procalib.npz")
DOT_RADIUS = 14
SCALE = 0.75


def load_calib(path: str) -> tuple[np.ndarray, np.ndarray, dict, float]:
    z = np.load(path)
    field = {
        "grid_pts": z["ray_grid_pts"],
        "origins": z["ray_origins"],
        "dirs": _dirs_toward_scene(z["ray_dirs"]),
    }
    return z["K_c"], z["dist_c"], field, reference_depth(field)


def _clamp_proj(uv: np.ndarray, field: dict) -> np.ndarray:
    xs = field["grid_pts"][:, 0]
    ys = field["grid_pts"][:, 1]
    return np.array([
        float(np.clip(uv[0], xs.min(), xs.max())),
        float(np.clip(uv[1], ys.min(), ys.max())),
    ])


def _camera_of_proj(proj_xy, plane_z: float, K, dist, field) -> np.ndarray | None:
    """Where a projector pixel lands on the fronto-parallel plane z=plane_z."""
    o, d = ray_field_lookup(field, np.asarray(proj_xy, dtype=np.float64).reshape(1, 2))
    if np.isnan(d).any() or abs(float(d[0, 2])) < 1e-8:
        return None
    d = _dirs_toward_scene(d[0])
    t = (plane_z - float(o[0, 2])) / float(d[2])
    Q = o[0] + t * d
    pts, _ = cv2.projectPoints(Q.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return pts.reshape(2)


def _correct_proj(proj, target, observed, plane_z, K, dist, field, step: float = 6.0):
    """One Jacobian step so the landing camera pixel moves toward the click."""
    p = np.asarray(proj, dtype=np.float64)
    c0 = _camera_of_proj(p, plane_z, K, dist, field)
    cx = _camera_of_proj(p + (step, 0), plane_z, K, dist, field)
    cy = _camera_of_proj(p + (0, step), plane_z, K, dist, field)
    if c0 is None or cx is None or cy is None:
        return p
    J = np.column_stack([(cx - c0) / step, (cy - c0) / step])
    if abs(np.linalg.det(J)) < 1e-6:
        return p
    dp = np.linalg.solve(J, np.asarray(target, dtype=np.float64) - np.asarray(observed))
    n = float(np.linalg.norm(dp))
    if n > 40.0:
        dp *= 40.0 / n
    return _clamp_proj(p + dp, field)


def _dot_image(proj_xy) -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.circle(img, (int(round(proj_xy[0])), int(round(proj_xy[1]))), DOT_RADIUS, 1, -1)
    return img


def _nearest_dot(frame, bg, hint) -> tuple[float, float] | None:
    cts = detect_grid_dots(frame, bg, thresh=40)
    if not cts:
        cts = detect_grid_dots(frame, None, raw_thresh=200)
    if not cts:
        return None
    hint = np.asarray(hint, dtype=np.float64)
    return min(cts, key=lambda p: float(np.hypot(p[0] - hint[0], p[1] - hint[1])))


class ClickUI:
    """Freehand closed shape, same gesture as the old 2D lasso picker."""

    def __init__(self):
        self.drawing = False
        self.points: list[tuple[float, float]] = []
        self.polygon = None
        self.pending_poly = None
        self.message = "hold and draw a shape, release to fill it with light    q = quit"

    def on_mouse(self, event, x, y, _flags, _param):
        px, py = x / SCALE, y / SCALE
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.points = [(px, py)]
            self.polygon = None
            self.pending_poly = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if not self.points or math.hypot(px - self.points[-1][0], py - self.points[-1][1]) >= 3:
                self.points.append((px, py))
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if not self.points or self.points[-1] != (px, py):
                self.points.append((px, py))
            contour = np.asarray(self.points, dtype=np.int32)
            if len(contour) >= 3 and abs(cv2.contourArea(contour)) >= 100:
                epsilon = max(1.0, 0.003 * cv2.arcLength(contour, True))
                self.polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
                self.pending_poly = self.polygon.astype(np.float64)
                self.points = []
            else:
                self.points = []
                self.message = "selection too small; draw again"

    def render(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        if self.polygon is not None:
            cv2.polylines(vis, [self.polygon.astype(np.int32)], True, (0, 255, 0), 2)
        if len(self.points) >= 2:
            cv2.polylines(vis, [np.asarray(self.points, dtype=np.int32)], False, (0, 255, 255), 2)
        cv2.putText(vis, self.message, (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return cv2.resize(vis, (int(vis.shape[1] * SCALE), int(vis.shape[0] * SCALE)))


class LatestProjection:
    """Serialize BLE uploads, replacing pending images instead of queuing them."""

    def __init__(self, cli: IrisBleClient) -> None:
        self.cli = cli
        self.pending: np.ndarray | None = None
        self.last = _black_image()
        self.task: asyncio.Task[None] | None = None

    def submit(self, mask: np.ndarray) -> None:
        self.pending = mask.copy()

    async def _send(self, mask: np.ndarray) -> None:
        started = time.perf_counter()
        await _project(self.cli, mask, settle=0.0)
        self.last = mask
        print(f"  projection upload={time.perf_counter() - started:.2f}s "
              "(not camera-to-light latency)", flush=True)

    def pump(self) -> None:
        if self.task is not None:
            if not self.task.done():
                return
            self.task.result()
            self.task = None
        if self.pending is not None:
            mask, self.pending = self.pending, None
            if not np.array_equal(mask, self.last):
                self.task = asyncio.create_task(self._send(mask))

    async def finish(self) -> None:
        self.pending = None
        # Do not interrupt a file transfer and then interleave a black upload.
        if self.task is not None:
            await self.task


async def _grab(cam: HikCamera):
    cam.flush(1)
    ok, frame = cam.read(1500)
    return frame if ok else None


async def aim_once(cli, cam, K, dist, field, z_ref, target, bg) -> tuple[np.ndarray, tuple | None, float | None]:
    proj = proj_pixel_for_camera(target, K, dist, field, z=z_ref)
    if proj is None:
        raise RuntimeError("click is outside the calibrated ray field")
    proj = np.array(proj, dtype=np.float64)
    print(f"  open-loop proj=({proj[0]:.1f},{proj[1]:.1f})", flush=True)
    measured = None
    plane_z = z_ref
    for step in range(2):
        await _project(cli, _dot_image(proj), settle=0.6)
        frame = await _grab(cam)
        if frame is None:
            break
        measured = _nearest_dot(frame, bg, target)
        if measured is None:
            print("  landing dot not detected", flush=True)
            break
        err = float(np.hypot(measured[0] - target[0], measured[1] - target[1]))
        print(f"  land ({measured[0]:.0f},{measured[1]:.0f}) err={err:.1f}px", flush=True)
        if err < 8 or step == 1:
            break
        X = triangulate_rayfield(
            np.array([measured]), np.array([proj]), K, dist, field)[0]
        if not np.isnan(X).any() and X[2] > 50:
            plane_z = float(X[2])
        proj = _correct_proj(proj, target, measured, plane_z, K, dist, field)
        print(f"  correct -> proj=({proj[0]:.1f},{proj[1]:.1f})", flush=True)
    err = None
    if measured is not None:
        err = float(np.hypot(measured[0] - target[0], measured[1] - target[1]))
    return proj, measured, err


async def run(args) -> int:
    K, dist, field, z_ref = load_calib(args.calib)
    H = plane_homography_from_rays(K, dist, field, z_ref)
    print(f"  rays={len(field['grid_pts'])}  depth_ref={z_ref:.0f}mm", flush=True)
    opened = await open_projector(args.address)
    cli = opened.client
    print(f"=== CONNECT {opened.address} ===", flush=True)
    cam = HikCamera(args.mvs_dir)
    cam.enumerate()
    cam.open(serial=args.serial)
    cam.set_gain(args.gain)
    cam.set_exposure_us(args.exposure)
    cam.flush(3)
    print(f"  camera: {cam.info()}", flush=True)
    ui = ClickUI()
    tracker = RegionTracker()
    output = LatestProjection(cli)
    cv2.namedWindow("aim3d")
    cv2.setMouseCallback("aim3d", ui.on_mouse)
    try:
        await _project(cli, _black_image(), settle=0.4)
        if args.at:
            bg = await _grab(cam)
            if bg is None:
                print("[error] no camera frame")
                return 2
            _proj, meas, err = await aim_once(
                cli, cam, K, dist, field, z_ref, (args.at[0], args.at[1]), bg)
            if meas is None:
                print("  landing blob not detected")
                return 2
            print(f"  target=({args.at[0]:.0f},{args.at[1]:.0f}) "
                  f"measured=({meas[0]:.1f},{meas[1]:.1f}) err={err:.1f}px")
            return 0

        print("window 'aim3d': draw a textured region, r = clear, q/Esc = quit", flush=True)
        if not args.static:
            print("  planar tracking only; BLE projection lags. "
                  "Loss requests black after any in-flight upload, NOT an emergency stop.",
                  flush=True)
        frame = None
        while True:
            output.pump()
            ok, captured = await asyncio.to_thread(cam.read, 100)
            if not ok or captured is None:
                if tracker.active or ui.polygon is not None:
                    print("[warning] camera frame unavailable; requesting black", flush=True)
                tracker.reset()
                ui.polygon = None
                ui.pending_poly = None
                ui.points = []
                ui.drawing = False
                ui.message = "camera frame unavailable; projection blank requested"
                output.submit(_black_image())
                if frame is not None:
                    cv2.imshow("aim3d", ui.render(frame))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue
            frame = captured
            if tracker.active:
                try:
                    ui.polygon = tracker.update(frame)
                except TrackingLost as exc:
                    ui.polygon = None
                    ui.message = f"LOST: {exc}; blank requested"
                    print(f"[warning] {ui.message}", flush=True)
                    output.submit(_black_image())
            cv2.imshow("aim3d", ui.render(frame))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r") or ui.drawing:
                tracker.reset()
                ui.polygon = None
                ui.pending_poly = None
                output.submit(_black_image())
                ui.message = "draw on a stationary textured object; release to track"
                if key == ord("r"):
                    ui.points = []
                    ui.drawing = False
                continue
            selected = ui.pending_poly is not None
            if selected:
                poly = ui.pending_poly
                ui.pending_poly = None
                if not args.static:
                    try:
                        ui.polygon = tracker.start(frame, poly)
                    except TrackingLost as exc:
                        ui.polygon = None
                        ui.message = str(exc)
                        print(f"[warning] {exc}", flush=True)
                        output.submit(_black_image())
                        continue
            if ui.polygon is None or (args.static and not selected):
                continue
            bw = shape_light_mask(ui.polygon, (frame.shape[1], frame.shape[0]), H)
            lit = int(bw.sum())
            if lit < 10:
                tracker.reset()
                ui.polygon = None
                ui.message = "outside projection field; blank requested, draw again"
                print(f"[warning] {ui.message}", flush=True)
                output.submit(_black_image())
                continue
            output.submit(bw)
            ui.message = ("static selection" if args.static else "tracking; BLE light lags")
            ui.message += "    r = clear, draw again, q = quit"
        return 0
    finally:
        cv2.destroyAllWindows()
        cam.close()
        try:
            try:
                await output.finish()
            finally:
                await _project(cli, _black_image(), settle=0.2)
        finally:
            await opened.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aim3d")
    ap.add_argument("--calib", default=CALIB)
    ap.add_argument("--address", default=None,
                    help="projector MAC; omit when only one IrisGreen is advertising")
    ap.add_argument("--serial", default="K65564865")
    ap.add_argument("--mvs-dir", default=None)
    ap.add_argument("--exposure", type=float, default=40000)
    ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"),
                    help="camera pixel, non-interactive single dot")
    ap.add_argument("--static", action="store_true",
                    help="disable object tracking; keep the original fixed selection")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.calib):
        print(f"[error] missing {args.calib}")
        return 2
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
