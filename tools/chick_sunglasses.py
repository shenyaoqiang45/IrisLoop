"""Fixed-pose toy sunglasses experiment. See docs/chick-sunglasses-projection-pipeline.md."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.aim_five import DEFAULT_BRIDGE, DEFAULT_SERIAL, grab, open_bridge, save_image, transform
from irisloop.calibrate import make_dot_image
from irisloop.hik_camera import HikCamera
from irisloop.procalib import detect_grid_dots

FACE = (375, 410, 758, 556)
SEED = ROOT / "captures" / "procalib" / "five_circles_20261027_100519" / "live_homography.npz"


def sunglasses():
    image = np.zeros((1024, 1280), np.uint8)
    left = np.array([(427, 458), (552, 458), (548, 482), (539, 501),
                     (525, 507), (452, 507), (439, 498)], np.int32)
    right = np.array([(594, 455), (716, 455), (713, 485), (703, 500),
                      (690, 505), (620, 505), (606, 496)], np.int32)
    cv2.fillPoly(image, [left, right], 1)
    cv2.polylines(image, [np.array([(546, 472), (558, 466), (578, 465), (601, 471)])],
                  False, 1, 9)
    cv2.line(image, (409, 452), (434, 465), 1, 8)
    cv2.line(image, (710, 462), (733, 450), 1, 8)
    return image


def find_spots(frame, bg):
    dots = detect_grid_dots(frame, bg, thresh=12, min_area=6, max_area=1000)
    return np.array([p for p in dots if FACE[0] < p[0] < FACE[2] and FACE[1] < p[1] < FACE[3]])


def basis(points):
    p = np.asarray(points, np.float64)
    x, y = (p[:, 0] - 320) / 320, (p[:, 1] - 240) / 240
    return np.column_stack([np.ones(len(p)), x, y, x*x, x*y, y*y])


def render(coefficients, proj_grid, desired):
    yy, xx = np.indices((480, 640))
    mapped = (basis(np.column_stack([xx.ravel(), yy.ravel()])) @ coefficients).reshape(480, 640, 2)
    # Measured feedback from the 2026-10-27 fixed-pose experiment, not a universal correction.
    right_weight = np.clip((mapped[:, :, 0] - 565) / 60, 0, 1)
    mapped[:, :, 0] -= 4.5 * right_weight
    mapped[:, :, 1] += right_weight
    mask = cv2.remap(desired, mapped[:, :, 0].astype(np.float32), mapped[:, :, 1].astype(np.float32),
                     cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    mask[(xx < proj_grid[:, 0].min() - 20) | (xx > proj_grid[:, 0].max() + 15) |
         (yy < proj_grid[:, 1].min() - 10) | (yy > proj_grid[:, 1].max() + 10)] = 0
    if mask.sum() < 1000:
        raise RuntimeError("Sunglasses mask unexpectedly small")
    return mask


def evaluate(frame, bg, desired):
    gain = frame.mean(axis=2).astype(float) - bg.mean(axis=2).astype(float)
    visible = (gain > 15).astype(np.uint8)
    roi = np.zeros_like(visible)
    roi[FACE[1]:FACE[3], FACE[0]:FACE[2]] = 1
    visible &= roi
    visible = cv2.morphologyEx(visible, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(visible, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    observed = np.zeros_like(visible)
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            cv2.drawContours(observed, [contour], -1, 1, -1)
    a, b = desired.astype(bool), observed.astype(bool)
    union = int((a | b).sum())
    iou = int((a & b).sum()) / union if union else 0
    expanded = cv2.dilate(desired, np.ones((15, 15), np.uint8)).astype(bool)
    total = int(b.sum())
    spill = int((b & ~expanded).sum()) / total if total else 1
    shifts, coverages = [], []
    for x0, x1 in ((420, 556), (592, 719)):
        region = np.zeros_like(a)
        region[448:515, x0:x1] = True
        yt, xt = np.nonzero(a & region)
        yo, xo = np.nonzero(b & region)
        if len(xo) < 100 or len(xt) == 0:
            raise RuntimeError(f"Lens at {x0} has no substantial target or measured light")
        shifts.append(float(np.hypot(xt.mean() - xo.mean(), yt.mean() - yo.mean())))
        coverages.append(float((a & b & region).sum() / (a & region).sum()))
    board_spill = float((gain[380:570, 775:1180] > 15).sum() / max(1, desired.sum()))
    result = dict(iou=iou, spill_outside_7px_margin=spill, lens_centroid_error_px=shifts,
                  lens_coverage=coverages, board_spill_fraction=board_spill)
    result["passed"] = (iou >= 0.75 and spill <= 0.1 and max(shifts) <= 5
                        and min(coverages) >= 0.8 and board_spill <= 0.03)
    return result, observed


async def run(args):
    reference = cv2.imread(str(args.reference))
    if reference is None or reference.shape != (1024, 1280, 3):
        raise ValueError("Reference must be a readable 1280x1024 scene image of the fixed toy pose")
    with np.load(args.seed) as seed:
        H = seed["H"].copy()
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("Invalid seed homography")
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "address": args.address, "target": "green sunglasses on toy eyes"}
    cam = HikCamera()
    bridge = None
    passed = False
    black = np.zeros((480, 640), np.uint8)
    try:
        bridge, state = await open_bridge(args.bridge, args.address)
        report["address"] = state["address"]

        async def project(mask):
            await bridge.push_current(mask)
            await asyncio.sleep(2.5)

        def capture():
            return grab(cam, args.serial, exposure=15000, gain=8, frames=10)

        await project(black)
        bg = capture()
        save_image(out, "background.png", bg)
        desired = sunglasses()
        save_image(out, "desired_camera_mask.png", desired * 255)
        ref_crop = cv2.cvtColor(reference[420:535, 395:752], cv2.COLOR_BGR2GRAY)
        bg_crop = cv2.cvtColor(bg[400:555, 375:772], cv2.COLOR_BGR2GRAY)
        scores = cv2.matchTemplate(bg_crop, ref_crop, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(scores)
        displacement = np.array(location) - (20, 20)
        if score < 0.85 or np.linalg.norm(displacement) > 2:
            raise RuntimeError(f"Toy moved: match={score:.3f}, shift={displacement}")
        report["initial_pose_check"] = dict(correlation=score, displacement=displacement.tolist())

        center_proj = np.rint(transform(np.linalg.inv(H), [[575., 450.]])[0])
        if ((center_proj < (7, 7)) | (center_proj > (632, 472))).any():
            raise RuntimeError("Center probe is outside the projection field")
        await project(make_dot_image(*center_proj, radius=5))
        frame = capture()
        save_image(out, "center_probe.png", frame)
        dots = find_spots(frame, bg)
        if len(dots) != 1:
            raise RuntimeError(f"Expected one center probe, got {dots.tolist()}")
        translation = dots[0] - transform(H, [center_proj])[0]
        H = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]]) @ H
        report["face_parallax_shift_px"] = translation.tolist()
        camera_grid = np.array([(x, y) for y in (440, 480, 523) for x in (421, 510, 623, 680)], float)
        proj_grid = np.rint(transform(np.linalg.inv(H), camera_grid))
        if ((proj_grid < (7, 7)) | (proj_grid > (632, 472))).any():
            raise RuntimeError("Face calibration grid is outside the projection field")
        grid_image = black.copy()
        for p in proj_grid:
            grid_image |= make_dot_image(*p, radius=4)
        await project(grid_image)
        grid = capture()
        save_image(out, "face_grid_raw.png", grid)
        dots = find_spots(grid, bg)
        if len(dots) != 12:
            raise RuntimeError(f"Expected 12 surface probes, got {len(dots)}")
        distances = np.linalg.norm(camera_grid[:, None] - dots[None, :], axis=2)
        indices = distances.argmin(axis=1)
        if len(set(indices)) != 12 or distances[np.arange(12), indices].max() > 35:
            raise RuntimeError("Surface probes cannot be matched uniquely")
        measured = dots[indices]
        coefficients, _, rank, _ = np.linalg.lstsq(basis(proj_grid), measured, rcond=None)
        residual = np.linalg.norm(basis(proj_grid) @ coefficients - measured, axis=1)
        if rank != 6 or not np.isfinite(coefficients).all() or residual.max() > 4:
            raise RuntimeError(f"Curved-face fit inadequate: rank={rank}, residual={residual}")
        report["surface_fit"] = dict(projector=proj_grid.tolist(), camera=measured.tolist(),
                                     residual_px=residual.tolist())
        report["right_lens_feedback_correction_px"] = [4.5, -1.0]
        mask = render(coefficients, proj_grid, desired)
        save_image(out, "projector_mask.png", mask * 255)
        np.savez(out / "face_mapping.npz", coefficients=coefficients, projector=proj_grid, camera=measured)
        await project(black)
        final_bg = capture()
        save_image(out, "final_background.png", final_bg)
        await project(mask)
        report["verification"] = []
        for i in range(3):
            frame = capture()
            save_image(out, f"verification_{i+1}_raw.png", frame)
            result, observed = evaluate(frame, final_bg, desired)
            save_image(out, f"verification_{i+1}_observed_mask.png", observed * 255)
            report["verification"].append(result)
            print("VERIFY", i+1, result, flush=True)
            if not result["passed"]:
                raise RuntimeError("Actual sunglasses failed optical alignment thresholds")
            await asyncio.sleep(1)
        await bridge.close()
        bridge = None
        frame = capture()
        save_image(out, "after_disconnect_raw.png", frame)
        result, _ = evaluate(frame, final_bg, desired)
        report["after_disconnect_verification"] = result
        if not result["passed"]:
            raise RuntimeError("Sunglasses failed post-disconnect verification")
        passed = True
        report.update(status="passed", final_state="sunglasses projecting", play_seconds=3600)
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        if bridge is None:
            print("[warn] No active BLE session; projection state cannot be changed.", flush=True)
        raise
    finally:
        try:
            if bridge is not None:
                try:
                    if not passed and bridge.connected:
                        await bridge.push_current(black)
                        report["final_state"] = "black requested"
                finally:
                    await bridge.close()
        finally:
            cam.close()
            (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("REPORT", out / "report.json", report["status"], flush=True)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--address", help="omit only when exactly one projector advertises")
    ap.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    ap.add_argument("--serial", default=DEFAULT_SERIAL)
    ap.add_argument("--seed", type=Path, default=SEED)
    ap.add_argument("--reference", type=Path, required=True, help="local scene image; never upload bystanders")
    ap.add_argument("--out", type=Path, default=ROOT / "captures" / "procalib" /
                    datetime.now().strftime("chick_sunglasses_%Y%m%d_%H%M%S"))
    return ap


if __name__ == "__main__":
    asyncio.run(run(parser().parse_args()))
