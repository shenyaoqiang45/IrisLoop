"""Camera-projector homography calibration for a flat projection wall.

Scenario: the IrisGreen projector and the camera are both fixed, pointing at
the same flat wall. A single 3x3 homography H then maps projector pixels
(640x480) onto camera pixels, and its inverse maps camera pixels back to
projector coordinates — the inverse is what lets us aim content at a spot
picked in the camera view.

Flow (run subcommand):
    1. connect BLE, open camera, capture a background frame (projection off)
    2. project a grid of dots one at a time (known projector coordinates)
       and detect the green centroid in each camera frame
    3. solve H with RANSAC; save homography.npz + raw point pairs (JSON)
    4. verify: project fresh test points, report predicted-vs-measured error

Other subcommands:
    solve   re-solve H from a saved points JSON (no hardware needed)
    map     convert coordinates: camera pixel -> projector pixel (--reverse flips)

Usage:
    python -m irisloop.calibrate run --address F4:12:FA:B6:B7:CA
    python -m irisloop.calibrate run --cols 6 --rows 5 --no-verify
    python -m irisloop.calibrate solve --points captures/calibration/calib_points.json
    python -m irisloop.calibrate map --x 640 --y 360
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from irisloop.ble_client import IrisBleClient
from irisloop.camera import UsbCamera
from irisloop.image_pack import HEIGHT, WIDTH

CAL_DIR = "captures/calibration"
DEFAULT_OUT = os.path.join(CAL_DIR, "homography.npz")
POINTS_JSON = os.path.join(CAL_DIR, "calib_points.json")

# keep the group on continuously while capturing (same params as irisloop.loop)


# ---------------- point grid & dot image ----------------


def grid_points(cols: int, rows: int, margin: float = 0.12) -> list[tuple[float, float]]:
    """Row-major grid of projector coordinates, inset by `margin` (fraction)."""
    xs = np.linspace(WIDTH * margin, WIDTH * (1 - margin), cols)
    ys = np.linspace(HEIGHT * margin, HEIGHT * (1 - margin), rows)
    return [(float(x), float(y)) for y in ys for x in xs]


def make_dot_image(x: float, y: float, radius: int = 12) -> np.ndarray:
    """640x480 1bpp image with a single filled dot at (x, y)."""
    img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.circle(img, (int(round(x)), int(round(y))), radius, 1, -1)
    return img


# ---------------- green-spot detection ----------------


def _greenness(frame: np.ndarray) -> np.ndarray:
    """How much greener a pixel is than red/blue. Laser green saturates G."""
    b, g, r = cv2.split(frame.astype(np.int16))
    return g - np.maximum(b, r)


def detect_dot(
    frame: np.ndarray,
    bg: np.ndarray | None = None,
    threshold: int = 40,
    min_area: int = 6,
    border: int = 24,
) -> tuple[tuple[float, float] | None, int]:
    """Locate the laser spot.

    Returns ((cx, cy), area) with a score-weighted subpixel centroid,
    or (None, area_of_largest_candidate) when nothing clears the thresholds.

    Score combines two cues against the background (projection off):
      - greenness gain, for unsaturated green light
      - green-channel brightness gain, because a bright laser dot saturates
        the sensor to white, where greenness (G - max(R,B)) collapses to ~0
    and its centroid would skew toward the green fringe.
    """
    if bg is not None:
        score = _greenness(frame) - _greenness(bg)
        dg = frame.astype(np.int16)[..., 1] - bg.astype(np.int16)[..., 1]
        score = score + np.clip(dg, 0, None)
    else:
        score = _greenness(frame)
    mask = (score > threshold).astype(np.uint8)
    # Camera-edge exposure drift can create a large horizontal component that
    # wins "largest blob" despite not being projected light. Calibration dots
    # are intentionally inset, so this border cannot contain a valid sample.
    if border > 0:
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None, 0
    largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        fill = area / max(1, width * height)
        elongation = max(width, height) / max(1, min(width, height))
        # A projected calibration dot remains compact. Reflections from the
        # glossy target and camera-edge drift form long strips or sparse blobs.
        if area >= min_area and fill >= 0.35 and elongation <= 3.0:
            candidates.append((area, i))
    if not candidates:
        return None, largest_area
    area, idx = max(candidates)
    comp = labels == idx
    w = np.clip(score, 0, None).astype(np.float64) * comp
    total = float(w.sum())
    if total <= 0:
        return (float(cents[idx][0]), float(cents[idx][1])), area
    ys, xs = np.nonzero(comp)
    cx = float((xs * w[comp]).sum() / total)
    cy = float((ys * w[comp]).sum() / total)
    return (cx, cy), area


# ---------------- camera helpers ----------------


def _open_camera(index: int, width: int, height: int) -> UsbCamera:
    cam = UsbCamera(index, width, height, 30)
    cam.open()
    # manual exposure: EV=-7 balances stripe suppression vs brightness (from loop.py)
    try:
        cam.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cam.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
    except Exception:
        pass
    return cam


def _grab_frame(cam: UsbCamera, flush: int = 8) -> np.ndarray | None:
    frame = None
    for _ in range(flush):
        ok, f = cam.read()
        if ok and f is not None:
            frame = f
    return frame


# ---------------- BLE projection ----------------


async def _project(cli: IrisBleClient, bw: np.ndarray, settle: float) -> None:
    """Overwrite 1_1.bmp and keep group 1 playing. Does not delete the group.

    stop is required before play: a re-issued play does NOT restart playback
    while already playing. Safe against the boot fallback only because run()
    clears group 1 first — the stop fallback then shows our own frame.
    """
    from irisloop.session import deliver_mask

    await deliver_mask(cli, bw, settle=settle)


# ---------------- collect ----------------


def _black_image() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


async def _capture_background(
    cli: IrisBleClient, cam: UsbCamera, out_dir: str | None = None
) -> np.ndarray | None:
    """Truly dark background frame.

    Do NOT use cli.stop() for this: stop (0xA1) only halts the commanded
    playback, and the device falls back to its built-in idle animation
    (the elephant) — which is green laser light too. An all-black bitmap
    played as group 1 is the only way to guarantee zero laser output.
    """
    await _project(cli, _black_image(), settle=1.0)
    frame = _grab_frame(cam)
    if frame is not None and out_dir:
        cv2.imwrite(os.path.join(out_dir, "bg.png"), frame)
    return frame


async def collect(
    cli: IrisBleClient,
    cam: UsbCamera,
    points: list[tuple[float, float]],
    out_dir: str,
    settle: float = 0.8,
    dot_radius: int = 12,
    threshold: int = 40,
) -> list[dict]:
    """Project each point, capture, detect. Returns one record per point."""
    bg = await _capture_background(cli, cam, out_dir)

    records: list[dict] = []
    for i, (px, py) in enumerate(points, 1):
        print(f"  [{i}/{len(points)}] proj=({px:.0f},{py:.0f}) ->", end=" ", flush=True)
        await _project(cli, make_dot_image(px, py, dot_radius), settle)
        frame = _grab_frame(cam)
        if frame is None:
            print("CAPTURE FAILED")
            records.append({"proj": [px, py], "cam": None, "area": 0, "shot": None})
            continue
        shot = os.path.join(out_dir, f"pt{i:02d}.png")
        cv2.imwrite(shot, frame)
        cam_pt, area = detect_dot(frame, bg, threshold=threshold)
        if cam_pt is None:
            print(f"NOT DETECTED (largest blob area={area})")
        else:
            print(f"cam=({cam_pt[0]:.1f},{cam_pt[1]:.1f}) area={area}")
        records.append({
            "proj": [px, py],
            "cam": [cam_pt[0], cam_pt[1]] if cam_pt else None,
            "area": area,
            "shot": shot,
        })
    await _project(cli, _black_image(), settle=0.3)  # go dark instead of idle animation
    return records


# ---------------- solve ----------------


def solve_homography(
    records: list[dict],
    ransac_thresh: float = 3.0,
) -> tuple[np.ndarray, dict]:
    """Solve projector->camera homography from collected point pairs."""
    pairs = [(r["proj"], r["cam"]) for r in records if r.get("cam")]
    if len(pairs) < 4:
        raise ValueError(f"need >= 4 valid point pairs, got {len(pairs)}")
    proj = np.array([p for p, _ in pairs], dtype=np.float64)
    cam = np.array([c for _, c in pairs], dtype=np.float64)

    H, mask = cv2.findHomography(
        proj.reshape(-1, 1, 2), cam.reshape(-1, 1, 2),
        cv2.RANSAC, ransac_thresh,
    )
    if H is None:
        raise ValueError("findHomography failed (points degenerate?)")
    inliers = mask.ravel().astype(bool)
    # least-squares refit on inliers only
    if int(inliers.sum()) >= 4:
        H = cv2.findHomography(
            proj[inliers].reshape(-1, 1, 2), cam[inliers].reshape(-1, 1, 2), 0
        )[0]

    pred = cv2.perspectiveTransform(proj.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(pred - cam, axis=1)
    stats = {
        "total": len(pairs),
        "inliers": int(inliers.sum()),
        "outliers": [i for i, ok in enumerate(inliers) if not ok],
        "err_mean": float(err[inliers].mean()) if inliers.any() else float(err.mean()),
        "err_max": float(err[inliers].max()) if inliers.any() else float(err.max()),
        "err_all_mean": float(err.mean()),
    }
    return H, stats


@dataclass
class Calibration:
    """A solved projector<->camera homography."""

    H: np.ndarray  # projector pixel -> camera pixel

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    def _map(self, pts, M: np.ndarray) -> np.ndarray:
        a = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(a, M).reshape(-1, 2)

    def proj_to_cam(self, pts) -> np.ndarray:
        return self._map(pts, self.H)

    def cam_to_proj(self, pts) -> np.ndarray:
        return self._map(pts, self.H_inv)

    def save(self, path: str, records: list[dict] | None = None, stats: dict | None = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "proj_size": [WIDTH, HEIGHT],
            "stats": stats or {},
        }
        valid = [r for r in (records or []) if r.get("cam")]
        np.savez(
            path,
            H=self.H,
            proj=np.array([r["proj"] for r in valid], dtype=np.float64),
            cam=np.array([r["cam"] for r in valid], dtype=np.float64),
            meta=np.array(json.dumps(meta, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: str) -> "Calibration":
        d = np.load(path, allow_pickle=False)
        return cls(H=d["H"].astype(np.float64))


def place_in_camera_rect(
    bw: np.ndarray,
    calib: Calibration,
    rect: tuple[float, float, float, float],
    padding: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Place 1bpp content inside a camera-view rectangle.

    The source aspect ratio is preserved (contain fit), then the fitted camera
    rectangle is mapped back into projector coordinates with the calibration.
    Returns the 640x480 projector bitmap and its projector-space quad.
    """
    source = np.asarray(bw, dtype=np.uint8)
    if source.ndim != 2 or source.shape[0] < 2 or source.shape[1] < 2:
        raise ValueError(f"expected a 2D image at least 2x2, got {source.shape}")
    x0, y0, x1, y1 = (float(v) for v in rect)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid camera rectangle: {rect}")
    if not 0.0 <= padding < 0.5:
        raise ValueError(f"padding must be in [0, 0.5), got {padding}")

    inset_x = (x1 - x0) * padding
    inset_y = (y1 - y0) * padding
    x0, x1 = x0 + inset_x, x1 - inset_x
    y0, y1 = y0 + inset_y, y1 - inset_y
    target_w, target_h = x1 - x0, y1 - y0
    source_aspect = source.shape[1] / source.shape[0]
    if target_w / target_h > source_aspect:
        fitted_w = target_h * source_aspect
        x0 += (target_w - fitted_w) / 2
        x1 = x0 + fitted_w
    else:
        fitted_h = target_w / source_aspect
        y0 += (target_h - fitted_h) / 2
        y1 = y0 + fitted_h

    camera_quad = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
    )
    projector_quad = calib.cam_to_proj(camera_quad)
    if not np.isfinite(projector_quad).all():
        raise ValueError("camera rectangle maps to non-finite projector coordinates")
    if not all(0 <= x < WIDTH and 0 <= y < HEIGHT for x, y in projector_quad):
        raise ValueError(
            "camera rectangle extends outside the calibrated projection field: "
            f"{projector_quad.round(1).tolist()}"
        )

    source_quad = np.float32(
        [[0, 0], [source.shape[1] - 1, 0],
         [source.shape[1] - 1, source.shape[0] - 1], [0, source.shape[0] - 1]]
    )
    transform = cv2.getPerspectiveTransform(source_quad, projector_quad.astype(np.float32))
    placed = cv2.warpPerspective(
        source,
        transform,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (placed > 0).astype(np.uint8), projector_quad


# ---------------- verify ----------------


def default_test_points() -> list[tuple[float, float]]:
    """Five fresh points (quarter-grid corners + center), disjoint from a coarse calib grid."""
    return [
        (0.25 * WIDTH, 0.25 * HEIGHT),
        (0.75 * WIDTH, 0.25 * HEIGHT),
        (0.50 * WIDTH, 0.50 * HEIGHT),
        (0.25 * WIDTH, 0.75 * HEIGHT),
        (0.75 * WIDTH, 0.75 * HEIGHT),
    ]


async def verify(
    cli: IrisBleClient,
    cam: UsbCamera,
    calib: Calibration,
    settle: float = 0.8,
    dot_radius: int = 12,
    threshold: int = 40,
) -> list[float]:
    """Project test points and compare measured vs H-predicted camera positions."""
    bg = await _capture_background(cli, cam, CAL_DIR)

    errors: list[float] = []
    for i, (px, py) in enumerate(default_test_points(), 1):
        await _project(cli, make_dot_image(px, py, dot_radius), settle)
        frame = _grab_frame(cam)
        if frame is None:
            print(f"  [verify {i}] proj=({px:.0f},{py:.0f}) capture failed")
            continue
        meas, area = detect_dot(frame, bg, threshold=threshold)
        pred = calib.proj_to_cam([(px, py)])[0]
        if meas is None:
            print(f"  [verify {i}] proj=({px:.0f},{py:.0f}) NOT DETECTED "
                  f"(predicted cam=({pred[0]:.0f},{pred[1]:.0f}))")
            continue
        err = float(np.hypot(pred[0] - meas[0], pred[1] - meas[1]))
        errors.append(err)
        print(f"  [verify {i}] proj=({px:.0f},{py:.0f}) predicted=({pred[0]:.0f},{pred[1]:.0f}) "
              f"measured=({meas[0]:.0f},{meas[1]:.0f}) err={err:.1f}px")
    await _project(cli, _black_image(), settle=0.3)  # go dark instead of idle animation
    return errors


# ---------------- pipeline ----------------


def _print_stats(stats: dict) -> None:
    print(f"  points: {stats['inliers']}/{stats['total']} inliers"
          + (f" (outlier idx: {stats['outliers']})" if stats["outliers"] else ""))
    print(f"  reprojection error: mean={stats['err_mean']:.2f}px max={stats['err_max']:.2f}px "
          f"(in-camera pixels)")


async def run(args: argparse.Namespace) -> int:
    os.makedirs(CAL_DIR, exist_ok=True)
    from irisloop.session import open_slot

    opened = await open_slot(args.address)
    cli = opened.client
    print(f"=== CONNECT {opened.address} ===")
    cam = _open_camera(args.camera, args.cam_width, args.cam_height)
    print(f"  camera: {cam.info()}")
    await asyncio.sleep(1.0)  # exposure settle
    try:
        points = grid_points(args.cols, args.rows, args.margin)
        print(f"=== COLLECT {len(points)} points ({args.cols}x{args.rows}, margin={args.margin}) ===")
        records = await collect(cli, cam, points, CAL_DIR,
                                settle=args.settle, dot_radius=args.dot_radius,
                                threshold=args.threshold)
        with open(POINTS_JSON, "w", encoding="utf-8") as f:
            json.dump({"records": records}, f, ensure_ascii=False, indent=2)
        print(f"  raw points -> {POINTS_JSON}")

        try:
            H, stats = solve_homography(records, ransac_thresh=args.ransac)
        except ValueError as e:
            print(f"[error] {e}")
            return 2
        print("=== SOLVE ===")
        _print_stats(stats)
        Calibration(H).save(args.out, records, stats)
        print(f"  homography -> {args.out}")

        if args.verify:
            print("=== VERIFY ===")
            errors = await verify(cli, cam, Calibration(H), settle=args.settle,
                                  dot_radius=args.dot_radius, threshold=args.threshold)
            if errors:
                print(f"  verify error: mean={np.mean(errors):.1f}px max={np.max(errors):.1f}px")
                if np.mean(errors) > 20:
                    print("  (large error — wall may not be flat, or rig moved during calibration)")
        return 0
    finally:
        cam.release()
        await cli.disconnect()


def main(argv=None) -> int:
    argv = list(argv) if argv is not None else os.sys.argv[1:]
    if not argv or argv[0] not in ("run", "solve", "map"):
        argv.insert(0, "run")  # bare invocation = full pipeline

    ap = argparse.ArgumentParser(prog="irisloop.calibrate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="full pipeline: collect -> solve -> save -> verify")
    pr.add_argument("--address", default=None,
                    help="projector MAC; omit when only one IrisGreen is advertising")
    pr.add_argument("--camera", type=int, default=0)
    pr.add_argument("--cam-width", type=int, default=1280)
    pr.add_argument("--cam-height", type=int, default=720)
    pr.add_argument("--cols", type=int, default=5)
    pr.add_argument("--rows", type=int, default=4)
    pr.add_argument("--margin", type=float, default=0.12,
                    help="grid inset as a fraction of the 640x480 field")
    pr.add_argument("--dot-radius", type=int, default=12)
    pr.add_argument("--settle", type=float, default=0.8,
                    help="seconds to wait after play before capturing")
    pr.add_argument("--threshold", type=int, default=40,
                    help="green-detection threshold (lower if dots are missed)")
    pr.add_argument("--ransac", type=float, default=3.0)
    pr.add_argument("--out", default=DEFAULT_OUT)
    pr.add_argument("--no-verify", dest="verify", action="store_false")

    ps = sub.add_parser("solve", help="re-solve H from a saved points JSON (offline)")
    ps.add_argument("--points", default=POINTS_JSON)
    ps.add_argument("--ransac", type=float, default=3.0)
    ps.add_argument("--out", default=DEFAULT_OUT)

    pm = sub.add_parser("map", help="convert a coordinate using a saved calibration")
    pm.add_argument("--calibration", default=DEFAULT_OUT)
    pm.add_argument("--x", type=float, required=True)
    pm.add_argument("--y", type=float, required=True)
    pm.add_argument("--reverse", action="store_true",
                    help="projector -> camera instead of camera -> projector")

    args = ap.parse_args(argv)

    if args.cmd == "run":
        return asyncio.run(run(args))

    if args.cmd == "solve":
        with open(args.points, encoding="utf-8") as f:
            records = json.load(f)["records"]
        H, stats = solve_homography(records, ransac_thresh=args.ransac)
        _print_stats(stats)
        Calibration(H).save(args.out, records, stats)
        print(f"  homography -> {args.out}")
        return 0

    # map
    calib = Calibration.load(args.calibration)
    if args.reverse:
        out = calib.proj_to_cam([(args.x, args.y)])[0]
        print(f"proj({args.x:.1f},{args.y:.1f}) -> cam({out[0]:.1f},{out[1]:.1f})")
    else:
        out = calib.cam_to_proj([(args.x, args.y)])[0]
        print(f"cam({args.x:.1f},{args.y:.1f}) -> proj({out[0]:.1f},{out[1]:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
