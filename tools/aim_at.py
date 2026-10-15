#!/usr/bin/env python
"""Aim-and-project demo: click a spot in the camera view, IrisGreen lights it up.

Uses the saved flat-wall homography (captures/calibration/homography.npz) to
map camera pixels back to projector coordinates, renders content only there
(rest of the 640x480 bitmap stays black = no light), projects, then captures
to measure the landing error.

Interactive:
    python tools/aim_at.py
        click        project a dot marker at that point
        drag         fill the dragged rectangle with a checkerboard
        q / Esc      quit (device goes dark)

Scriptable single shot (also used for automated tests):
    python tools/aim_at.py --at 600 300
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

if sys.platform == "win32":
    try:  # sharp 1:1 window pixels; without this Windows bitmap-scales the window
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

import cv2
import numpy as np

from irisloop.calibrate import (
    CAL_DIR,
    DEFAULT_OUT,
    Calibration,
    _black_image,
    _capture_background,
    _grab_frame,
    _open_camera,
    _project,
    detect_dot,
)
from irisloop.image_pack import HEIGHT, WIDTH
MARK_RADIUS = 24          # projector-px radius of the dot marker
CHECKER_CELLS = (8, 6)    # checkerboard cells for the drag-rectangle fill


# ---------------- content builders ----------------


def in_bounds(pt, margin: float = 0.0) -> bool:
    x, y = pt
    return margin <= x <= WIDTH - margin and margin <= y <= HEIGHT - margin


def build_dot_marker(proj_pt, radius: int = MARK_RADIUS) -> np.ndarray:
    bw = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.circle(bw, (int(round(proj_pt[0])), int(round(proj_pt[1]))), radius, 1, -1)
    return bw


def build_checker_quad(proj_quad) -> np.ndarray:
    """Checkerboard warped into the quad (TL, TR, BR, BL in projector coords)."""
    cw, ch = 80, 60  # pattern resolution
    pat = np.zeros((ch, cw), dtype=np.uint8)
    cells_x, cells_y = CHECKER_CELLS
    for r in range(cells_y):
        for c in range(cells_x):
            if (r + c) % 2 == 0:
                pat[r * ch // cells_y:(r + 1) * ch // cells_y,
                    c * cw // cells_x:(c + 1) * cw // cells_x] = 1
    src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    dst = np.float32(proj_quad)
    M = cv2.getPerspectiveTransform(src, dst)
    bw = cv2.warpPerspective(pat.astype(np.float32), M, (WIDTH, HEIGHT))
    return (bw > 0.5).astype(np.uint8)


# ---------------- aim + verify ----------------


async def aim_at(cli, cam, bg, calib: Calibration, target_cam,
                 shape: str = "point", settle: float = 0.8):
    """Project a marker at target_cam (point) or target_cam=(x0,y0,x1,y1) rect.

    Returns (bw, measured_cam_pt, err_px). measured/err are None when the
    landing blob could not be detected. Raises ValueError if out of bounds.
    """
    if shape == "point":
        proj_pts = calib.cam_to_proj([target_cam])
        center_proj = proj_pts[0]
        if not in_bounds(center_proj, margin=MARK_RADIUS):
            raise ValueError(
                f"target maps outside the projection field "
                f"(proj={center_proj[0]:.0f},{center_proj[1]:.0f}); move the device or pick closer"
            )
        bw = build_dot_marker(center_proj)
        ref_cam = np.asarray(target_cam, dtype=np.float64)
    else:
        x0, y0, x1, y1 = target_cam
        quad_cam = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        quad_proj = calib.cam_to_proj(quad_cam)
        bad = [tuple(p) for p in quad_proj if not in_bounds(p)]
        if bad:
            raise ValueError(f"rectangle corners outside the projection field: {bad}")
        bw = build_checker_quad(quad_proj)
        ref_cam = np.array([(x0 + x1) / 2, (y0 + y1) / 2])  # centroid sanity check

    await _project(cli, bw, settle)
    frame = _grab_frame(cam)
    if frame is None:
        return bw, None, None
    meas, _area = detect_dot(frame, bg)
    if meas is None:
        return bw, None, None
    err = float(np.hypot(meas[0] - ref_cam[0], meas[1] - ref_cam[1]))
    return bw, meas, err


# ---------------- interactive UI ----------------


class AimUI:
    def __init__(self, bg_frame):
        self.bg = bg_frame
        self.drag_start = None     # (x, y) while dragging
        self.drag_cur = None
        self.pending = None        # ("point", (x,y)) or ("rect", (x0,y0,x1,y1))
        self.last_target = None    # camera point that was aimed at
        self.last_meas = None      # measured landing point
        self.last_err = None
        self.message = "click = dot, drag = checker box, q = quit"

    def on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_cur = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x0, y0 = self.drag_start
            self.drag_start = None
            if abs(x - x0) < 8 and abs(y - y0) < 8:
                self.pending = ("point", (x, y))
            else:
                self.pending = ("rect", (min(x0, x), min(y0, y), max(x0, x), max(y0, y)))

    def render(self) -> np.ndarray:
        disp = self.bg.copy()
        if self.drag_start and self.drag_cur:
            cv2.rectangle(disp, self.drag_start, self.drag_cur, (0, 255, 255), 1)
        if self.last_target:
            cv2.drawMarker(disp, tuple(int(v) for v in self.last_target),
                           (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        if self.last_meas:
            cv2.drawMarker(disp, tuple(int(v) for v in self.last_meas),
                           (0, 0, 255), cv2.MARKER_TILTED_CROSS, 24, 2)
        if self.last_err is not None:
            cv2.putText(disp, f"err={self.last_err:.1f}px", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(disp, self.message, (20, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return disp


async def run_interactive(args) -> int:
    calib = Calibration.load(args.calibration)
    from irisloop.session import open_slot

    opened = await open_slot(args.address)
    cli = opened.client
    print(f"=== CONNECT {opened.address} ===")
    cam = _open_camera(args.camera, args.cam_width, args.cam_height)
    print(f"  camera: {cam.info()}")
    await asyncio.sleep(1.0)
    try:
        bg = await _capture_background(cli, cam, CAL_DIR)  # device dark, clean wall frame
        ui = AimUI(bg)
        cv2.namedWindow("aim", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("aim", ui.on_mouse)
        cv2.resizeWindow("aim", bg.shape[1], bg.shape[0])  # 1:1 image pixels
        print("window 'aim': click = dot, drag = checker box, q/Esc = quit")
        while True:
            cv2.imshow("aim", ui.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            if ui.pending:
                shape, target = ui.pending
                ui.pending = None
                try:
                    _bw, meas, err = await aim_at(cli, cam, bg, calib, target, shape)
                except ValueError as e:
                    ui.message = f"cannot project there: {e}"
                    ui.last_target = None
                    continue
                center = target if shape == "point" else (
                    (target[0] + target[2]) / 2, (target[1] + target[3]) / 2)
                ui.last_target = center
                ui.last_meas = meas
                ui.last_err = err
                if meas is None:
                    ui.message = "projected, but landing blob not detected"
                else:
                    ui.message = f"landed at ({meas[0]:.0f},{meas[1]:.0f}), err={err:.1f}px"
                print(f"  aim {target} -> {ui.message}")
            await asyncio.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        cam.release()
        try:
            await _project(cli, _black_image(), settle=0.3)  # leave the device dark
        finally:
            await opened.close()
    return 0


async def run_single(args) -> int:
    calib = Calibration.load(args.calibration)
    from irisloop.session import open_slot

    opened = await open_slot(args.address)
    cli = opened.client
    print(f"=== CONNECT {opened.address} ===")
    cam = _open_camera(args.camera, args.cam_width, args.cam_height)
    print(f"  camera: {cam.info()}")
    await asyncio.sleep(1.0)
    try:
        bg = await _capture_background(cli, cam, CAL_DIR)
        bw, meas, err = await aim_at(cli, cam, bg, calib, (args.at[0], args.at[1]))
        cv2.imwrite(os.path.join(CAL_DIR, "aim_last.png"),
                    (bw * 255).astype(np.uint8))
        frame = _grab_frame(cam)
        if frame is not None:
            cv2.imwrite(os.path.join(CAL_DIR, "aim_last_capture.png"), frame)
        if meas is None:
            print("  landing blob not detected")
            return 2
        print(f"  target cam=({args.at[0]},{args.at[1]})  "
              f"measured=({meas[0]:.1f},{meas[1]:.1f})  err={err:.1f}px")
        return 0
    finally:
        cam.release()
        try:
            await _project(cli, _black_image(), settle=0.3)
        finally:
            await opened.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aim_at")
    ap.add_argument("--address", default=None,
                    help="projector MAC; omit when only one IrisGreen is advertising")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--calibration", default=DEFAULT_OUT)
    ap.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"),
                    help="camera pixel to aim at (non-interactive single shot)")
    args = ap.parse_args(argv)
    if not os.path.exists(args.calibration):
        print(f"[error] no calibration at {args.calibration} — run tools/run_calibration.py first")
        return 2
    if args.at:
        return asyncio.run(run_single(args))
    return asyncio.run(run_interactive(args))


if __name__ == "__main__":
    sys.exit(main())
