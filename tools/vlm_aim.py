#!/usr/bin/env python
"""VLM-aimed projection: say where, Qwen3-VL picks the region, IrisGreen lights it.

The upgrade of the manual "draw where to project" flow: instead of dragging a
mouse, the user states an intent; Qwen3-VL-32B-Instruct looks at the wall frame
(with the orange projection-field overlay) and returns the target region in
0-1000 normalized coordinates. Everything downstream (homography mapping,
target path, BLE push, loop playback) reuses the existing pipeline.

    python tools/vlm_aim.py "把光投到海报人物的头显上"
    python tools/vlm_aim.py "照亮海报" --draw fill
    python tools/vlm_aim.py "给海报描个边" --draw stroke --blink --interval 2
    python tools/vlm_aim.py "投到 E7 标志上" --judge   # ask the VLM to verify the landing
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np

from irisloop import protocol as P
from irisloop import qwen_client as qwen
from irisloop.ble_client import IrisBleClient
from irisloop.calibrate import (
    CAL_DIR,
    DEFAULT_OUT,
    Calibration,
    _capture_background,
    _grab_frame,
    _greenness,
    _open_camera,
)
from irisloop.image_pack import HEIGHT, WIDTH, build_stream
from irisloop.loop import DEFAULT_ADDR, push_image
from irisloop.target_path import TargetPath, render_target
from irisloop.vlm_ground import GROUND_PROMPT, refine_outline, regions_from_reply
from play_video_at import field_overlay

MAX_GROUP_FRAMES = 10


def fit_to_field(calib: Calibration, pts, max_iter: int = 12):
    """Shrink camera-space points toward their centroid until all of them map
    inside the 640x480 projection field."""
    pts = [tuple(map(float, p)) for p in pts]
    for _ in range(max_iter):
        proj = calib.cam_to_proj(pts)
        if np.isfinite(proj).all() and all(
            0 <= x < WIDTH and 0 <= y < HEIGHT for x, y in proj
        ):
            return pts
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        pts = [(cx + 0.85 * (x - cx), cy + 0.85 * (y - cy)) for x, y in pts]
    raise ValueError(
        "target cannot fit inside the projection field — pick something within the orange quad"
    )


def annotate(frame, pts, label: str):
    out = frame.copy()
    poly = np.rint(np.asarray(pts)).astype(np.int32)
    cv2.polylines(out, [poly], True, (0, 255, 0), 2)
    # cv2.putText renders no CJK glyphs — keep the overlay ASCII
    label = label.encode("ascii", "ignore").decode()
    if label:
        cv2.putText(out, label, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
    return out


def _field_limited_mask(calib: Calibration, frame, bg) -> np.ndarray:
    """Laser pixels in a frame vs the dark background, restricted to the
    projection field (ambient drift elsewhere is not our light)."""
    score = _greenness(frame) - _greenness(bg)
    dg = frame.astype(np.int16)[..., 1] - bg.astype(np.int16)[..., 1]
    mask = ((score + np.clip(dg, 0, None)) > 40).astype(np.uint8)
    field = calib.proj_to_cam(
        [(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)]
    ).astype(np.int32)
    field_mask = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(field_mask, field, 1)
    field_mask = cv2.dilate(field_mask, np.ones((21, 21), np.uint8))
    mask = mask * field_mask
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


async def run(args) -> int:
    calib = Calibration.load(args.calibration)
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
        wall = field_overlay(calib, bg)
        cv2.imwrite(os.path.join(CAL_DIR, "vlm_query.png"), wall)

        model = args.model or qwen.default_model()
        print(f'=== GROUND "{args.intent}" via {model} ===', flush=True)
        prompt = GROUND_PROMPT.format(intent=args.intent)
        resp = qwen.chat(
            qwen.vision_message(prompt, qwen.b64_data_url_bgr(wall)),
            model=args.model,
        )
        reply = qwen.message_text(resp)
        print(f"  reply: {reply[:300]}")
        h, w = wall.shape[:2]
        try:
            regions = regions_from_reply(reply, w, h)
        except ValueError as e:
            print(f"[error] {e}")
            return 2
        region = regions[0]
        label = region["label"] or args.intent
        print(f"  picked: {label} bbox={tuple(round(v) for v in region['bbox'])}")

        pts = None
        if args.refine:
            try:
                pts = refine_outline(bg, region["bbox"])
                print(f"  refined outline: {len(pts)} pts (GrabCut on clean frame)")
            except ValueError as e:
                print(f"  [warn] refine failed: {e} — falling back to box")
        if pts is None:
            if args.shape == "outline" and region["polygon"]:
                pts = region["polygon"]
            else:
                x0, y0, x1, y1 = region["bbox"]
                pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        try:
            pts = fit_to_field(calib, pts)
        except ValueError as e:
            print(f"[error] {e}")
            return 2
        cv2.imwrite(os.path.join(CAL_DIR, "vlm_pick.png"), annotate(wall, pts, label))

        if args.draw == "fill":
            mode, paths = "shape", [pts]
        else:
            mode, paths = "stroke", [pts + [pts[0]]]  # close the outline
        target = TargetPath.create(
            paths,
            scene=args.scene,
            mode=mode,
            camera_size=(w, h),
            calibration_path=args.calibration,
            note=f"vlm_aim: {args.intent}",
        )
        rendered = render_target(target, calib, args.brush_width)
        target.save(args.path_out)
        cv2.imwrite(os.path.join(CAL_DIR, "vlm_mask.png"), rendered.mask * 255)
        print(f"  target: {rendered.mask_pixels} lit px -> {args.path_out}")

        pattern = rendered.mask
        black = np.zeros_like(pattern)
        if args.blink:
            frames = [pattern if i % 2 == 0 else black for i in range(MAX_GROUP_FRAMES)]
        else:
            frames = [pattern] * MAX_GROUP_FRAMES
        for i, bw in enumerate(frames, 1):
            print(f"  pushing frame {i}/{len(frames)} -> 1_{i}.bmp ({bw.sum()} lit px)")
            await push_image(cli, build_stream(bw), f"1_{i}.bmp")
            await asyncio.sleep(0.2)
        await cli.stop()
        await asyncio.sleep(0.3)
        interval = args.interval if args.blink else 50
        r = await cli.play(group_id=1, loop=True,
                           total_100ms=36000, interval_100ms=interval)
        print(f"=== PLAY {'blink ' if args.blink else ''}group 1 "
              f"@ {interval * 100}ms/frame: {'ok' if r.ok else r.error} ===")
        if not r.ok:
            return 2

        # verification capture; with blink, sample twice and keep the lit frame
        await asyncio.sleep(1.2)
        best = None
        for _ in range(2):
            frame = _grab_frame(cam)
            if frame is None:
                continue
            mask = _field_limited_mask(calib, frame, bg)
            lit = int(mask.sum())
            if best is None or lit > best[1]:
                best = (frame, lit, mask)
            await asyncio.sleep(0.4)
        if best is None:
            print("  [warn] verification capture failed")
            return 0
        frame, lit, mask = best
        cv2.imwrite(os.path.join(CAL_DIR, "vlm_capture.png"), frame)
        ys, xs = np.nonzero(mask)
        if len(xs):
            lx0, lx1 = np.percentile(xs, 2), np.percentile(xs, 98)
            ly0, ly1 = np.percentile(ys, 2), np.percentile(ys, 98)
            tx = [p[0] for p in pts]
            ty = [p[1] for p in pts]
            print(f"  landed bbox cam=({lx0:.0f},{ly0:.0f})-({lx1:.0f},{ly1:.0f}) "
                  f"(target ({min(tx):.0f},{min(ty):.0f})-({max(tx):.0f},{max(ty):.0f}))")
        else:
            print("  [warn] no laser pixels detected in verification capture")

        if args.judge:
            judge_prompt = (
                f"这是激光投射后的相机实拍图。亮绿色激光应该落在「{label}」上"
                f"（用户意图：{args.intent}）。请判断激光是否命中目标，只输出 JSON："
                '{"hit": true或false, "note": "一句话说明"}'
            )
            jresp = qwen.chat(
                qwen.vision_message(judge_prompt, qwen.b64_data_url_bgr(frame)),
                model=args.model,
            )
            print(f"  judge: {qwen.message_text(jresp)[:200]}")

        print("  frames keep looping for 1h; device stays on.")
        return 0
    finally:
        cam.release()
        await cli.disconnect()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vlm_aim")
    ap.add_argument("intent", help="natural-language target, e.g. '把光投到海报人物的头显上'")
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--calibration", default=DEFAULT_OUT)
    ap.add_argument("--model", default=None,
                    help="override SILICONFLOW_VL_MODEL (default Qwen/Qwen3-VL-32B-Instruct)")
    ap.add_argument("--shape", choices=["box", "outline"], default="box",
                    help="use the model's bbox, or its outline polygon when provided")
    ap.add_argument("--refine", action="store_true",
                    help="refine the VLM bbox into a true silhouette via GrabCut")
    ap.add_argument("--draw", choices=["fill", "stroke"], default="fill",
                    help="fill the region, or trace its boundary with the brush")
    ap.add_argument("--brush-width", type=int, default=16,
                    help="camera-pixel width for --draw stroke")
    ap.add_argument("--blink", action="store_true",
                    help="alternate pattern/black frames for a blinking effect")
    ap.add_argument("--interval", type=int, default=2,
                    help="blink frame time in 100ms units (2 = 200ms)")
    ap.add_argument("--judge", action="store_true",
                    help="ask the VLM to verify the landing on the capture")
    ap.add_argument("--scene", default="current_wall")
    ap.add_argument("--path-out",
                    default=os.path.join(CAL_DIR, "target_path.json"),
                    help="where to save the reusable camera-space target")
    args = ap.parse_args(argv)
    if args.brush_width < 1:
        ap.error("--brush-width must be >= 1")
    if args.interval < 1:
        ap.error("--interval must be >= 1")
    if not os.path.exists(args.calibration):
        print(f"[error] no calibration at {args.calibration} — run python -m irisloop.calibrate run first")
        return 2
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
