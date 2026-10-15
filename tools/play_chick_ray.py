"""Box-select a camera region and project the dancing chick there.

The destination rectangle is mapped with the ray-field calibration
(captures/procalib/run/procalib.npz), not the flat-wall homography.

    python tools/play_chick_ray.py
        1. drag around one chick in the video (Esc = the left chick)
        2. drag the target rectangle on the live camera view
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import cv2
import numpy as np

from irisloop.hik_camera import HikCamera
from irisloop.image_pack import HEIGHT, WIDTH, to_1bpp
from irisloop.procalib import _black_image, _project, proj_pixel_for_camera
from tools.aim3d import CALIB, load_calib
from tools.play_video_at import extract_frames, pick_region, read_preview_frame, warp_to_quad

SCALE = 0.75
LEFT_CHICK = (0, 0, 480, 720)  # left half of data/小猫跳舞.mp4 (960x720)


def cam_quad_to_proj(rect, K, dist, field, z):
    """Camera rect (x0,y0,x1,y1) -> projector quad TL,TR,BR,BL.

    Shrinks toward the center until every corner hits the calibrated bundle.
    """
    x0, y0, x1, y1 = [float(v) for v in rect]
    for _ in range(10):
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        proj = [proj_pixel_for_camera(c, K, dist, field, z=z) for c in corners]
        if all(p is not None for p in proj):
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], proj
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1 = cx + 0.85 * (x0 - cx), cx + 0.85 * (x1 - cx)
        y0, y1 = cy + 0.85 * (y0 - cy), cy + 0.85 * (y1 - cy)
    return None


class _Drag:
    def __init__(self, scale: float):
        self.scale = scale
        self.start = None
        self.cur = None
        self.rect = None

    def on_mouse(self, event, x, y, _flags, _param):
        px, py = x / self.scale, y / self.scale
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (px, py)
            self.cur = (px, py)
        elif event == cv2.EVENT_MOUSEMOVE and self.start:
            self.cur = (px, py)
        elif event == cv2.EVENT_LBUTTONUP and self.start:
            x0, y0 = self.start
            self.start = None
            if abs(px - x0) >= 16 and abs(py - y0) >= 16:
                self.rect = (min(x0, px), min(y0, py), max(x0, px), max(y0, py))


async def pick_target(cam) -> tuple[float, float, float, float] | None:
    drag = _Drag(SCALE)
    cv2.namedWindow("pick target")
    cv2.setMouseCallback("pick target", drag.on_mouse)
    print("window 'pick target': drag the rectangle to project the chick into. Esc = cancel",
          flush=True)
    try:
        while drag.rect is None:
            ok, frame = cam.read(1000)
            if not ok or frame is None:
                continue
            vis = frame.copy()
            if drag.start and drag.cur:
                a = tuple(int(v) for v in drag.start)
                b = tuple(int(v) for v in drag.cur)
                cv2.rectangle(vis, a, b, (0, 255, 255), 2)
            cv2.putText(vis, "drag a box for the chick   Esc=cancel", (16, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("pick target", cv2.resize(
                vis, (int(vis.shape[1] * SCALE), int(vis.shape[0] * SCALE))))
            if (cv2.waitKey(1) & 0xFF) == 27:
                return None
            await asyncio.sleep(0.01)
    finally:
        cv2.destroyWindow("pick target")
    return drag.rect


async def run(args) -> int:
    preview = read_preview_frame(args.video)
    crop = await pick_region(
        preview, "pick chick",
        "drag around ONE chick (Esc = left chick)",
        required=False)
    if crop is None:
        crop = LEFT_CHICK
    print(f"  content crop: {crop}", flush=True)

    K, dist, field, z_ref = load_calib(args.calib)
    print(f"  rays={len(field['grid_pts'])} depth_ref={z_ref:.0f}mm", flush=True)
    frames = extract_frames(args.video, args.frames, crop)
    print(f"  {len(frames)} frames", flush=True)

    from irisloop.session import open_slot

    opened = await open_slot(args.address)
    cli = opened.client
    print(f"=== CONNECT {opened.address} ===", flush=True)
    cam = HikCamera(args.mvs_dir)
    cam.enumerate()
    cam.open(serial=args.serial)
    cam.set_gain(args.gain)
    cam.set_exposure_us(args.exposure)
    cam.flush(3)
    print(f"  camera: {cam.info()}", flush=True)
    try:
        await _project(cli, _black_image(), settle=0.4)
        rect = await pick_target(cam)
        if rect is None:
            print("  cancelled", flush=True)
            return 1
        mapped = cam_quad_to_proj(rect, K, dist, field, z_ref)
        if mapped is None:
            print("[error] that box is outside the calibrated projection field", flush=True)
            return 2
        quad_cam, quad_proj = mapped
        print(f"  cam box: {[tuple(round(v) for v in p) for p in quad_cam]}", flush=True)
        print(f"  proj quad: {[tuple(round(v) for v in p) for p in quad_proj]}", flush=True)

        masks = [warp_to_quad(to_1bpp(gray), quad_proj) for gray in frames]
        frame_ms = max(100, int(round(1000 / args.fps)))
        shown = await opened.show_animation(masks, frame_ms=frame_ms, seconds=3600)
        print(
            f"=== PLAY {shown.frames} frames @ {shown.frame_ms}ms, "
            f"replaced {shown.replaced} ===",
            flush=True,
        )
        print("chick is looping. q/Esc closes and turns the projector off.", flush=True)
        while True:
            ok, frame = cam.read(1000)
            if ok and frame is not None:
                cv2.imshow("playing", cv2.resize(
                    frame, (int(frame.shape[1] * SCALE), int(frame.shape[0] * SCALE))))
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
        return 0
    finally:
        cv2.destroyAllWindows()
        cam.close()
        try:
            await _project(cli, _black_image(), settle=0.2)
        finally:
            await opened.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="play_chick_ray")
    ap.add_argument("--video", default=os.path.join("data", "小猫跳舞.mp4"))
    ap.add_argument("--calib", default=CALIB)
    ap.add_argument("--address", default=None,
                    help="projector MAC; omit when only one IrisGreen is advertising")
    ap.add_argument("--serial", default="K65564865")
    ap.add_argument("--mvs-dir", default=None)
    ap.add_argument("--exposure", type=float, default=40000)
    ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--fps", type=float, default=5.0)
    args = ap.parse_args(argv)
    if args.frames > 10:
        args.frames = 10
    if not os.path.isfile(args.video):
        print(f"[error] missing {args.video}")
        return 2
    if not os.path.isfile(args.calib):
        print(f"[error] missing {args.calib}")
        return 2
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
