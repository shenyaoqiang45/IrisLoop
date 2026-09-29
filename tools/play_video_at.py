#!/usr/bin/env python
"""Play a video as a laser animation at a chosen spot in the camera view.

Extracts up to 10 frames (group-1 capacity), converts each to 640x480 1bpp,
warps them into the projector quad that corresponds to the target rectangle
in camera space (via the saved homography), pushes them as 1_1..1_N.bmp and
plays group 1 on a loop.

    python tools/play_video_at.py data/小猫跳舞.mp4
        -> window "pick content": drag around the subject in the video
           (e.g. just the left kitten); Esc = use the full frame
        -> window "pick target": drag the target rectangle on the wall frame
    python tools/play_video_at.py data/小猫跳舞.mp4 --at 650 300 --width 320 --fps 5
        -> non-interactive: center + width in camera pixels, full frame
    python tools/play_video_at.py data/小猫跳舞.mp4 --crop 0 150 480 600 --at 650 300
        -> non-interactive with a content crop in video pixels
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

from irisloop import protocol as P
from irisloop.ble_client import IrisBleClient
from irisloop.calibrate import (
    CAL_DIR,
    DEFAULT_OUT,
    Calibration,
    _black_image,
    _capture_background,
    _grab_frame,
    _greenness,
    _open_camera,
    _project,
)
from irisloop.image_pack import HEIGHT, WIDTH, build_stream, to_1bpp
from irisloop.loop import DEFAULT_ADDR, push_image

MAX_GROUP_FRAMES = 10


def read_preview_frame(video_path: str) -> np.ndarray:
    """Middle frame of the video, BGR, original resolution (for crop picking)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, f = cap.read()
    cap.release()
    if not ok or f is None:
        raise ValueError("cannot read a preview frame")
    return f


def extract_frames(video_path: str, n: int, crop=None) -> list[np.ndarray]:
    """n evenly spaced grayscale frames, optionally cropped to
    (x0, y0, x1, y1) in original video pixels, then resized to 640x480."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, min(n, total)).round().astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok and f is not None:
            if crop is not None:
                x0, y0, x1, y1 = crop
                f = f[y0:y1, x0:x1]
            frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (WIDTH, HEIGHT)))
    cap.release()
    if not frames:
        raise ValueError("no frames extracted")
    return frames


def target_quad_cam(calib: Calibration, at, width_cam: float | None,
                    aspect: float = 0.75):
    """Target rect (TL,TR,BR,BL) in camera coords.

    Default (no --at/--width): a centered rect spanning 70% of the PROJECTOR
    field, mapped to camera space — guaranteed to be in bounds. A camera-space
    default is not safe: the camera squeezes the field vertically, so a 4:3
    camera rect maps to a taller-than-field projector quad.
    """
    if at is None and width_cam is None:
        quad_proj = [(0.15 * WIDTH, 0.15 * HEIGHT), (0.85 * WIDTH, 0.15 * HEIGHT),
                     (0.85 * WIDTH, 0.85 * HEIGHT), (0.15 * WIDTH, 0.85 * HEIGHT)]
        return [tuple(p) for p in calib.proj_to_cam(quad_proj)]
    if at is None:
        at = calib.proj_to_cam([(WIDTH / 2, HEIGHT / 2)])[0]
    if width_cam is None:
        left = calib.proj_to_cam([(0, HEIGHT / 2)])[0]
        right = calib.proj_to_cam([(WIDTH, HEIGHT / 2)])[0]
        width_cam = 0.5 * float(np.hypot(*(right - left)))
    cx, cy = float(at[0]), float(at[1])
    w2, h2 = width_cam / 2, width_cam * aspect / 2
    return [(cx - w2, cy - h2), (cx + w2, cy - h2), (cx + w2, cy + h2), (cx - w2, cy + h2)]


def fit_quad(calib: Calibration, quad_cam, max_iter: int = 12):
    """Shrink the camera-space rect toward its center until the mapped
    projector quad fits the 640x480 field. Returns (quad_cam, quad_proj)."""
    for _ in range(max_iter):
        quad_proj = calib.cam_to_proj(quad_cam)
        if all(0 <= p[0] <= WIDTH and 0 <= p[1] <= HEIGHT for p in quad_proj):
            return quad_cam, quad_proj
        cx = sum(p[0] for p in quad_cam) / 4
        cy = sum(p[1] for p in quad_cam) / 4
        quad_cam = [(cx + 0.85 * (p[0] - cx), cy + 0.85 * (p[1] - cy)) for p in quad_cam]
    raise ValueError("target rectangle cannot fit inside the projection field")


def warp_to_quad(content_bw: np.ndarray, quad_proj) -> np.ndarray:
    src = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    M = cv2.getPerspectiveTransform(src, np.float32(quad_proj))
    return (cv2.warpPerspective(content_bw.astype(np.float32), M, (WIDTH, HEIGHT)) > 0.5).astype(np.uint8)


def aspect_fit(quad_cam, content_aspect: float):
    """Shrink an axis-aligned camera rect (TL,TR,BR,BL) to the content's
    aspect ratio (w/h), keeping center and the limiting dimension — otherwise
    the laser content stretches to fill whatever shape was picked."""
    (x0, y0), (x1, y1) = quad_cam[0], quad_cam[2]
    w, h = x1 - x0, y1 - y0
    if h <= 0 or w <= 0:
        return quad_cam
    if w / h > content_aspect:  # too wide: shrink width
        w2 = h * content_aspect
        x0, x1 = x0 + (w - w2) / 2, x1 - (w - w2) / 2
    else:                        # too tall: shrink height
        h2 = w / content_aspect
        y0, y1 = y0 + (h - h2) / 2, y1 - (h - h2) / 2
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# ---------------- interactive region picking ----------------


class _RegionPicker:
    def __init__(self):
        self.start = None
        self.cur = None
        self.rect = None  # (x0, y0, x1, y1)

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.cur = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start:
            self.cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start:
            x0, y0 = self.start
            self.start = None
            if abs(x - x0) >= 12 and abs(y - y0) >= 12:
                self.rect = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))


async def pick_region(frame: np.ndarray, title: str, prompt: str,
                      required: bool = True) -> tuple[int, int, int, int] | None:
    """Show a frame; user drags a rectangle. Esc: cancel (required) or skip
    (optional -> returns None, meaning 'no selection')."""
    picker = _RegionPicker()
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, picker.on_mouse)
    h, w = frame.shape[:2]
    cv2.resizeWindow(title, w, h)  # 1:1 image pixels, exact picking
    print(f"window '{title}': {prompt}")
    try:
        while picker.rect is None:
            disp = frame.copy()
            if picker.start and picker.cur:
                cv2.rectangle(disp, picker.start, picker.cur, (0, 255, 255), 2)
            cv2.putText(disp, prompt, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(title, disp)
            if (cv2.waitKey(20) & 0xFF) == 27:
                return None
            await asyncio.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
    return picker.rect


# ---------------- interactive region picking end ----------------


def field_overlay(calib: Calibration, frame: np.ndarray) -> np.ndarray:
    """Wall frame with the reachable projection field outlined."""
    quad = calib.proj_to_cam(
        [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]
    ).astype(np.int32)
    out = frame.copy()
    cv2.polylines(out, [quad], True, (0, 200, 255), 2)
    cv2.putText(out, "projection field", tuple(quad[3] + [10, 30]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return out


async def run(args) -> int:
    calib = Calibration.load(args.calibration)
    interactive = args.at is None and args.width is None

    crop = tuple(args.crop) if args.crop else None
    if interactive and crop is None:
        # stage 1: pick WHAT to project (a subject inside the video frame)
        preview = read_preview_frame(args.video)
        picked = await pick_region(preview, "pick content",
                                   "drag around the subject to project (Esc = full frame)",
                                   required=False)
        if picked is not None:
            crop = picked
        print(f"  content crop (video px): {crop or 'full frame'}")
    quad_cam = None if interactive else target_quad_cam(calib, args.at, args.width)

    if crop:
        content_aspect = (crop[2] - crop[0]) / (crop[3] - crop[1])
    else:
        cap = cv2.VideoCapture(args.video)
        content_aspect = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / max(1.0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    print(f"=== extract {args.frames} frames from {args.video} ===")
    frames = extract_frames(args.video, args.frames, crop)
    print(f"  got {len(frames)} frames, mode={args.mode}")

    cli = IrisBleClient(args.address)
    print(f"=== CONNECT {args.address} ===")
    await cli.connect()
    cam = _open_camera(args.camera, args.cam_width, args.cam_height)
    print(f"  camera: {cam.info()}")
    await asyncio.sleep(1.0)
    try:
        r = await cli.send_command(P.cmd_delete_group(1))
        print(f"  clear group 1: {'ok' if r.ok else r.error}")
        await asyncio.sleep(0.5)
        bg = await _capture_background(cli, cam, CAL_DIR)

        if interactive:
            # stage 2: pick WHERE to project it (on the clean wall frame,
            # with the reachable projection field outlined; retry until it fits)
            wall = field_overlay(calib, bg)
            while True:
                rect = await pick_region(wall, "pick target",
                                         "drag inside the orange field outline, Esc = cancel")
                if rect is None:
                    print("  cancelled")
                    await _project(cli, _black_image(), settle=0.3)
                    return 1
                x0, y0, x1, y1 = rect
                quad_cam = aspect_fit([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                                      content_aspect)
                try:
                    quad_cam, quad_proj = fit_quad(calib, quad_cam)
                    break
                except ValueError:
                    print("  region outside the projection field — pick again")
        else:
            quad_cam = aspect_fit(quad_cam, content_aspect)
            try:
                quad_cam, quad_proj = fit_quad(calib, quad_cam)
            except ValueError as e:
                print(f"[error] {e} — pick a region inside the projection field")
                await _project(cli, _black_image(), settle=0.3)
                return 2
        print(f"  target cam rect: {[tuple(round(v) for v in p) for p in quad_cam]}")

        for i, gray in enumerate(frames, 1):
            bw = warp_to_quad(to_1bpp(gray, mode=args.mode), quad_proj)
            name = f"1_{i}.bmp"
            print(f"  pushing frame {i}/{len(frames)} -> {name} ({bw.sum()} lit px)")
            await push_image(cli, build_stream(bw), name)
            await asyncio.sleep(0.2)
            if i == 1:
                cv2.imwrite(os.path.join(CAL_DIR, "video_frame1.png"),
                            (bw * 255).astype(np.uint8))

        interval = max(1, round(10 / args.fps))
        await cli.stop()
        await asyncio.sleep(0.3)
        r = await cli.play(group_id=1, loop=True, total_100ms=36000, interval_100ms=interval)
        print(f"=== PLAY group 1, {len(frames)} frames @ {args.fps}fps: "
              f"{'ok' if r.ok else r.error} ===")
        await asyncio.sleep(1.2)

        frame = _grab_frame(cam)
        if frame is not None:
            cv2.imwrite(os.path.join(CAL_DIR, "video_play_capture.png"), frame)
            score = _greenness(frame) - _greenness(bg)
            dg = frame.astype(np.int16)[..., 1] - bg.astype(np.int16)[..., 1]
            mask = ((score + np.clip(dg, 0, None)) > 40).astype(np.uint8)
            # only the projection field can contain laser light — ambient
            # drift elsewhere (window ledge, plants) is not our content
            field = calib.proj_to_cam(
                [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]).astype(np.int32)
            field_mask = np.zeros(mask.shape, np.uint8)
            cv2.fillConvexPoly(field_mask, field, 1)
            field_mask = cv2.dilate(field_mask, np.ones((21, 21), np.uint8))
            mask = mask * field_mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            # keep blobs comparable to the largest one: video content is dimmer
            # per pixel than a calibration dot, and small blobs here are ambient
            # drift (e.g. the window ledge), not laser
            areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
            floor = max(20, int(0.05 * max(areas))) if areas else 20
            keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= floor]
            if keep:
                xs = np.concatenate([np.nonzero(labels == i)[1] for i in keep])
                ys = np.concatenate([np.nonzero(labels == i)[0] for i in keep])
                xs, ys = xs.astype(float), ys.astype(float)
                x0, x1 = np.percentile(xs, 2), np.percentile(xs, 98)
                y0, y1 = np.percentile(ys, 2), np.percentile(ys, 98)
                tx = sorted(p[0] for p in quad_cam)
                ty = sorted(p[1] for p in quad_cam)
                print(f"  landed bbox cam=({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f}) "
                      f"(target ({tx[0]:.0f},{ty[0]:.0f})-({tx[-1]:.0f},{ty[-1]:.0f}))")
            else:
                print("  [warn] no bright region detected in verification capture")
        print(f"  frames keep looping for 1h; device stays on. Ctrl-C already done.")
        return 0
    finally:
        cam.release()
        await cli.disconnect()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="play_video_at")
    ap.add_argument("video", help="source video path")
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--calibration", default=DEFAULT_OUT)
    ap.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"),
                    help="target center in camera pixels; omit for interactive region picking")
    ap.add_argument("--width", type=float,
                    help="target width in camera pixels; omit for interactive region picking")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="content crop in video pixels (skip interactive crop picking)")
    ap.add_argument("--frames", type=int, default=MAX_GROUP_FRAMES,
                    help="frames to extract (group capacity 10)")
    ap.add_argument("--fps", type=float, default=5.0, help="playback frame rate")
    ap.add_argument("--mode", default="threshold",
                    choices=["threshold", "otsu", "dither", "dither_flat", "edges"],
                    help="1bpp conversion mode")
    args = ap.parse_args(argv)
    if args.frames > MAX_GROUP_FRAMES:
        print(f"[warn] group 1 holds at most {MAX_GROUP_FRAMES} frames; clamping")
        args.frames = MAX_GROUP_FRAMES
    if not os.path.exists(args.calibration):
        print(f"[error] no calibration at {args.calibration} — run tools/run_calibration.py first")
        return 2
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
