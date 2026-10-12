"""Camera-projector inverse-camera-model calibration (海康相机 × IrisGreen 激光投影).

Method: the projector is calibrated as an inverse camera. A dot-plate
(CB-0300-W-001, 11x9 dots @25mm, 5 large fiducials fixing orientation) provides
both the camera intrinsics target and the per-pose plane on which projected
laser dots are localized in 3D.

Pipeline (subcommands):
    camint   capture board poses (projection off) -> camera intrinsics npz
    collect  per pose: black frame (board pose) + laser dot-grid frame ->
             per-pose (projector pixel <-> camera pixel <-> 3D) records
    solve    offline: projector intrinsics + stereo extrinsics -> procalib.npz
    verify   fresh poses: triangulation + plane-fit RMSE report

Board frames use long exposure; laser frames use short exposure + multi-frame
averaging (suppresses laser scan stripes). The board must stay fixed during
each pose's two captures (use the vice), reposition between poses.

Usage:
    python -m irisloop.procalib camint --dir captures/procalib/run1
    python -m irisloop.procalib collect --dir captures/procalib/run1 --poses 10
    python -m irisloop.procalib solve --dir captures/procalib/run1
    python -m irisloop.procalib verify --dir captures/procalib/run1 --poses 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import time
from dataclasses import dataclass

import cv2
import numpy as np

from irisloop import protocol as P
from irisloop.ble_client import IrisBleClient
from irisloop.hik_camera import HikCamera
from irisloop.image_pack import WIDTH, HEIGHT, build_stream
from irisloop.loop import DEFAULT_ADDR, push_image

DEFAULT_LAYOUT = "cbinfo/CB-0300-W-001.layout.json"
DEFAULT_DIR = os.path.join("captures", "procalib", time.strftime("run_%Y%m%d_%H%M%S"))
CAM_INT_NPZ = "camera_intrinsics.npz"
PROCALIB_NPZ = "procalib.npz"
REPORT_JSON = "report.json"

# keep the group on continuously while capturing (same as irisloop.calibrate)
PLAY_TOTAL_100MS = 36000
PLAY_INTERVAL_100MS = 50


# ---------------- board model ----------------


@dataclass
class BoardModel:
    cols: int
    rows: int
    pitch_mm: float
    fiducials: set[tuple[int, int]]  # canonical (col,row) of the large dots

    @classmethod
    def load(cls, path: str = DEFAULT_LAYOUT) -> "BoardModel":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            cols=int(d["cols"]),
            rows=int(d["rows"]),
            pitch_mm=float(d["pitch_mm"]),
            fiducials={(f["col"], f["row"]) for f in d["fiducials"]},
        )

    def object_points(self) -> dict[tuple[int, int], tuple[float, float, float]]:
        """Board-frame 3D points (z=0), keyed by (col,row). Origin at corner (0,0);
        absolute pose of the origin is irrelevant (fixed by fiducial convention)."""
        return {
            (c, r): (c * self.pitch_mm, r * self.pitch_mm, 0.0)
            for c in range(self.cols)
            for r in range(self.rows)
        }


# ---------------- board dot detection & lattice matching ----------------


def detect_board_dots(
    frame: np.ndarray,
    block: int = 51,
    cval: int = 8,
    min_area: float = 30,
    max_area: float = 8000,
    min_circularity: float = 0.55,
) -> list[dict]:
    """Detect dark circular dots (the board's printed dots) in a BGR frame.

    Adaptive threshold: a dot is darker than its local surround by `cval` gray
    levels — robust to global exposure changes (fixed thresholds fail when the
    whole board goes dim). `block` should be ~2x the dot pitch in pixels.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    block |= 1  # must be odd
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, block, cval)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circ = 4 * math.pi * area / (peri * peri)
        if circ < min_circularity:
            continue
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        dots.append({"cx": m["m10"] / m["m00"], "cy": m["m01"] / m["m00"], "area": area})
    return dots


def _rot90k(c: int, r: int, k: int, cols: int, rows: int) -> tuple[int, int]:
    """Rotate canonical (col,row) by k*90 deg. Odd k swaps the frame dims —
    the relative BFS frame can be oriented along either physical axis."""
    if k == 0:
        return c, r
    if k == 1:
        return rows - 1 - r, c
    if k == 2:
        return cols - 1 - c, rows - 1 - r
    return r, cols - 1 - c


def _rot90k_inv(c: int, r: int, k: int, cols: int, rows: int) -> tuple[int, int]:
    """Inverse of _rot90k (maps rotated frame back to the canonical 11x9)."""
    return _rot90k(c, r, (4 - k) % 4, rows, cols) if k % 2 else _rot90k(c, r, (4 - k) % 4, cols, rows)


def match_lattice(
    dots: list[dict],
    board: BoardModel,
    max_resid_frac: float = 0.35,
    min_matched: int = 40,
    big_area_ratio: float = 1.8,
    min_fiducials: int = 4,
) -> dict[tuple[int, int], tuple[float, float]] | None:
    """Assign absolute board (col,row) to detected dots.

    Local graph walk (robust to perspective, unlike a global affine fit):
    BFS over nearest-neighbor edges (~pitch), stepping in grid directions
    classified against the dominant basis. Absolute anchoring via the fiducial
    set (0/180-degree search + translation). Final homography residual filter
    drops mis-assignments. Returns {(col,row): (cx,cy)} or None.
    """
    n = len(dots)
    if n < 20:
        return None
    pts = np.array([[d["cx"], d["cy"]] for d in dots], dtype=np.float64)
    areas = np.array([d["area"] for d in dots], dtype=np.float64)

    dmat = np.linalg.norm(pts[None, :, :] - pts[:, None, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    nn_idx = np.argmin(dmat, axis=1)
    nn_dist = dmat[np.arange(n), nn_idx]
    pitch = float(np.median(nn_dist))

    vecs = pts[nn_idx] - pts
    good = nn_dist < 1.4 * pitch
    ang = np.arctan2(vecs[good, 1], vecs[good, 0]) % math.pi
    hist, edges = np.histogram(ang, bins=72, range=(0, math.pi))
    a1 = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    v1 = np.array([math.cos(a1), math.sin(a1)])
    v2 = np.array([-v1[1], v1[0]])

    # adjacency: edges at ~pitch (diagonals at 1.41x pitch mostly excluded)
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    lo, hi = 0.6 * pitch, 1.35 * pitch
    ii, jj = np.nonzero((dmat > lo) & (dmat < hi))
    for a, b in zip(ii.tolist(), jj.tolist()):
        adj[a].append(b)

    # BFS grid-coordinate walk from the dot nearest the centroid
    seed = int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    assign: dict[int, tuple[int, int]] = {seed: (0, 0)}
    queue = [seed]
    while queue:
        i = queue.pop()
        ci, ri = assign[i]
        for j in adj[i]:
            if j in assign:
                continue
            v = pts[j] - pts[i]
            s1, s2 = float(v @ v1), float(v @ v2)
            step = (int(math.copysign(1, s1)), 0) if abs(s1) > abs(s2) else (0, int(math.copysign(1, s2)))
            assign[j] = (ci + step[0], ri + step[1])
            queue.append(j)
    if len(assign) < min_matched:
        return None

    idxs = np.array(sorted(assign.keys()))
    ij = np.array([assign[i] for i in idxs], dtype=np.int64)

    # drop gross mis-assignments: RANSAC homography (models perspective)
    Hm, hm_mask = cv2.findHomography(
        ij.astype(np.float64), pts[idxs], cv2.RANSAC, 0.25 * pitch)
    if Hm is None or hm_mask is None or int(hm_mask.sum()) < min_matched:
        return None

    # Re-assign EVERY dot through the homography: BFS coords can contain slips
    # and collisions where spurious blobs short-circuit the walk; the fitted H
    # is the consistent majority model.
    Hi = np.linalg.inv(Hm)
    grid_xy = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hi).reshape(-1, 2)
    ij_all = np.round(grid_xy).astype(int)
    pred = cv2.perspectiveTransform(
        ij_all.astype(np.float64).reshape(-1, 1, 2), Hm).reshape(-1, 2)
    resid = np.linalg.norm(pred - pts, axis=1)
    inl = resid < 0.3 * pitch
    # dedupe: one dot per lattice cell (closest to prediction wins)
    best_per_cell: dict[tuple[int, int], int] = {}
    for i in np.nonzero(inl)[0]:
        cell = (int(ij_all[i][0]), int(ij_all[i][1]))
        if cell not in best_per_cell or resid[i] < resid[best_per_cell[cell]]:
            best_per_cell[cell] = i
    if len(best_per_cell) < min_matched:
        return None
    sel = np.array(sorted(best_per_cell.values()))
    ij2_all = ij_all[sel]
    sel_pts = pts[sel]
    sel_areas = areas[sel]

    med_area = float(np.median(sel_areas))
    big = sel_areas > big_area_ratio * med_area
    if int(big.sum()) < min_fiducials:
        return None

    # anchor: 4 rotations (the relative BFS frame can start on either physical
    # axis, so 90-degree rotations are possible too) + translation maximizing
    # fiducial matches. Hypothesis: q = R_k(p) + t  (q relative, p canonical).
    best = None
    bigs_rel = [tuple(x) for x in ij2_all[big]]
    for k in range(4):
        ref_k = {_rot90k(c, r, k, board.cols, board.rows) for c, r in board.fiducials}
        for bc, br in bigs_rel:
            for fc, fr in ref_k:
                t = (fc - bc, fr - br)
                score = sum((b[0] + t[0], b[1] + t[1]) in ref_k for b in bigs_rel)
                if best is None or score > best[0]:
                    best = (score, k, t)
    if best is None or best[0] < min_fiducials:
        return None
    _, k, t = best

    out: dict[tuple[int, int], tuple[float, float]] = {}
    for (c, r), p in zip(ij2_all, sel_pts):
        cc, rr = _rot90k_inv(int(c) + t[0], int(r) + t[1], k, board.cols, board.rows)
        if 0 <= cc < board.cols and 0 <= rr < board.rows:
            out[(cc, rr)] = (float(p[0]), float(p[1]))
    return out if len(out) >= min_matched else None


def solve_camera_intrinsics(
    views: list[dict[tuple[int, int], tuple[float, float]]],
    board: BoardModel,
    img_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """views: matched {(col,row): (cx,cy)} per pose. Returns K, dist, RMS px."""
    obj_all, img_all = [], []
    obj_map = board.object_points()
    for v in views:
        keys = sorted(v.keys())
        obj_all.append(np.array([obj_map[k] for k in keys], dtype=np.float32))
        img_all.append(np.array([v[k] for k in keys], dtype=np.float32))
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_all, img_all, img_size, None, None,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-8))
    return K, dist, float(rms)


def board_pose(
    matched: dict[tuple[int, int], tuple[float, float]],
    board: BoardModel,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Board pose in camera frame via planar solvePnP."""
    obj_map = board.object_points()
    keys = sorted(matched.keys())
    obj = np.array([obj_map[k] for k in keys], dtype=np.float64)
    img = np.array([matched[k] for k in keys], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP failed on board dots")
    return rvec, tvec


# ---------------- laser dot grid ----------------


def make_grid_image(
    cols: int = 10, rows: int = 8, margin: float = 0.12, radius: int = 6
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Dot-grid 1bpp image + projector coords (row-major)."""
    xs = np.linspace(WIDTH * margin, WIDTH * (1 - margin), cols)
    ys = np.linspace(HEIGHT * margin, HEIGHT * (1 - margin), rows)
    pts = [(float(x), float(y)) for y in ys for x in xs]
    img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for x, y in pts:
        cv2.circle(img, (int(round(x)), int(round(y))), radius, 1, -1)
    return img, pts


def detect_grid_dots(
    frame: np.ndarray,
    bg: np.ndarray | None,
    thresh: int = 30,
    min_area: int = 6,
    max_area: int = 2000,
    raw_thresh: int = 200,
) -> list[tuple[float, float]]:
    """Detect laser dot centroids.

    Two modes:
      - bg given: brightness gain vs the dark field (any exposure).
      - bg None: raw bright blobs above raw_thresh (use at board exposure where
        laser dots saturate at 255 against a ~100-gray board).
    Compactness filters reject scan-stripe / reflection artifacts.
    """
    gray = frame.astype(np.int16).mean(axis=2)
    if bg is not None:
        score = gray - bg.astype(np.int16).mean(axis=2)
        mask = (score > thresh).astype(np.uint8)
    else:
        score = gray
        mask = (gray > raw_thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if not (min_area <= area <= max_area):
            continue
        fill = area / max(1, w * h)
        elong = max(w, h) / max(1, min(w, h))
        if fill < 0.35 or elong > 3.0:
            continue
        comp = labels == i
        wt = np.clip(score, 0, None).astype(np.float64) * comp
        total = float(wt.sum())
        if total <= 0:
            out.append((float(cents[i][0]), float(cents[i][1])))
        else:
            ys, xs = np.nonzero(comp)
            out.append((float((xs * wt[comp]).sum() / total),
                        float((ys * wt[comp]).sum() / total)))
    return out


def bright_on_board(
    frame: np.ndarray,
    board: BoardModel,
    block: int,
    c: int,
    raw_thresh: int = 200,
    margin: float = 40.0,
    min_board: int = 30,
) -> tuple[dict | None, list[tuple[float, float]]]:
    """Laser-dot centroids that fall inside the detected calibration board.

    Ceiling lamps and monitor LEDs saturate the same way a laser dot does.
    Restricting to the board bounding box keeps anchor/black/grid checks
    from treating those fixtures as projected content.
    """
    matched = match_lattice(detect_board_dots(frame, block, c), board)
    cts = detect_grid_dots(frame, None, raw_thresh=raw_thresh)
    if not matched or len(matched) < min_board:
        return None, []
    pts = np.array(list(matched.values()), dtype=np.float64)
    x0, y0 = pts.min(axis=0) - margin
    x1, y1 = pts.max(axis=0) + margin
    kept = [p for p in cts if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
    return matched, kept


def board_roi(matched: dict, margin: float = 40.0) -> tuple[float, float, float, float]:
    pts = np.array(list(matched.values()), dtype=np.float64)
    x0, y0 = pts.min(axis=0) - margin
    x1, y1 = pts.max(axis=0) + margin
    return float(x0), float(y0), float(x1), float(y1)


def dots_in_roi(
    frame: np.ndarray,
    roi: tuple[float, float, float, float] | None,
    raw_thresh: int = 200,
) -> list[tuple[float, float]]:
    cts = detect_grid_dots(frame, None, raw_thresh=raw_thresh)
    if roi is None:
        return cts
    x0, y0, x1, y1 = roi
    return [p for p in cts if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]


def assign_grid_indices(
    centroids: list[tuple[float, float]],
    H_boot: np.ndarray,
    grid_pts: list[tuple[float, float]],
    tol_px: float = 25.0,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Match detected centroids to known projector grid points via H_boot.

    Two passes: coarse nearest-neighbor to estimate the median global offset
    (board depth changes parallax-shift the rough bootstrap homography), then
    offset-corrected nearest neighbor within tol_px.
    Returns [(proj_xy, cam_xy)].
    """
    if not centroids:
        return []
    pred = cv2.perspectiveTransform(
        np.array(grid_pts, dtype=np.float64).reshape(-1, 1, 2), H_boot
    ).reshape(-1, 2)
    det = np.array(centroids, dtype=np.float64)

    def nn_matches(offset: np.ndarray, tol: float):
        pairs = {}
        used = set()
        for gi, p in enumerate(pred + offset):
            d = np.linalg.norm(det - p, axis=1)
            j = int(np.argmin(d))
            if d[j] <= tol and j not in used:
                pairs[gi] = j
                used.add(j)
        return pairs

    coarse = nn_matches(np.zeros(2), tol_px * 2.5)
    offset = np.zeros(2)
    if len(coarse) >= 3:
        shifts = np.array([det[j] - pred[gi] for gi, j in coarse.items()])
        offset = np.median(shifts, axis=0)
    fine = nn_matches(offset, tol_px)
    return [(grid_pts[gi], tuple(det[j])) for gi, j in sorted(fine.items())]


# ---------------- geometry ----------------


def rays_to_board(
    cam_xy: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """Camera pixels -> 3D points on the board plane (camera frame)."""
    pts = np.asarray(cam_xy, dtype=np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    dirs = np.hstack([norm, np.ones((len(norm), 1))])  # ray directions, cam frame
    R, _ = cv2.Rodrigues(rvec)
    n = R @ np.array([0.0, 0.0, 1.0])
    p0 = tvec.reshape(3)
    t = (p0 @ n) / (dirs @ n)
    return dirs * t[:, None]


def to_board_frame(
    X_cam: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    return (R.T @ (np.asarray(X_cam, dtype=np.float64) - tvec.reshape(3)).T).T


def solve_projector(
    obj_views: list[np.ndarray],
    proj_px_views: list[np.ndarray],
    proj_size: tuple[int, int] = (WIDTH, HEIGHT),
) -> tuple[np.ndarray, np.ndarray, float, list[np.ndarray], list[np.ndarray]]:
    """Projector intrinsics: object points = board-frame 3D of the laser dots,
    image points = their projector pixels. Returns K_p, dist_p, RMS, and the
    per-view board poses in the projector frame (for diagnostics)."""
    obj = [np.asarray(o, dtype=np.float32) for o in obj_views]
    img = [np.asarray(i, dtype=np.float32) for i in proj_px_views]
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-8)
    rms, K_p, dist_p, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, proj_size, None, None, criteria=criteria)
    return K_p, dist_p, float(rms), rvecs, tvecs


def per_view_reproj_errors(
    obj_views: list[np.ndarray],
    px_views: list[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
) -> np.ndarray:
    """Per-view mean reprojection error (px)."""
    errs = []
    for obj, px, rv, tv in zip(obj_views, px_views, rvecs, tvecs):
        proj, _ = cv2.projectPoints(np.asarray(obj, dtype=np.float64), rv, tv, K, dist)
        errs.append(float(np.linalg.norm(proj.reshape(-1, 2) - np.asarray(px).reshape(-1, 2), axis=1).mean()))
    return np.array(errs)


def prune_views(
    errs: np.ndarray, hard_cap: float = 2.5
) -> np.ndarray:
    """Indices to keep: adaptive threshold min(hard_cap, max(0.8, median*1.8))
    (from harness-calibration's validated strategy)."""
    thr = min(hard_cap, max(0.8, float(np.median(errs)) * 1.8))
    return np.nonzero(errs <= thr)[0]


def solve_stereo(
    obj_views: list[np.ndarray],
    cam_px_views: list[np.ndarray],
    proj_px_views: list[np.ndarray],
    K_c: np.ndarray,
    dist_c: np.ndarray,
    K_p: np.ndarray,
    dist_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Stereo extrinsics: X_proj = R @ X_cam + T (OpenCV stereoCalibrate semantic)."""
    obj = [np.asarray(o, dtype=np.float32) for o in obj_views]
    img1 = [np.asarray(i, dtype=np.float32) for i in cam_px_views]
    img2 = [np.asarray(i, dtype=np.float32) for i in proj_px_views]
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-9)
    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        obj, img1, img2, K_c, dist_c, K_p, dist_p, None,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    return R, T.reshape(3), float(rms)


def triangulate(
    cam_xy: np.ndarray,
    proj_xy: np.ndarray,
    K_c: np.ndarray,
    dist_c: np.ndarray,
    K_p: np.ndarray,
    dist_p: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Midpoint triangulation of camera-pixel / projector-pixel pairs."""
    nc = cv2.undistortPoints(
        np.asarray(cam_xy, dtype=np.float64).reshape(-1, 1, 2), K_c, dist_c
    ).reshape(-1, 2)
    np_ = cv2.undistortPoints(
        np.asarray(proj_xy, dtype=np.float64).reshape(-1, 1, 2), K_p, dist_p
    ).reshape(-1, 2)
    d1 = np.hstack([nc, np.ones((len(nc), 1))])
    d1 = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    d2p = np.hstack([np_, np.ones((len(np_), 1))])
    d2 = (R.T @ d2p.T).T  # projector ray dirs in camera frame
    d2 = d2 / np.linalg.norm(d2, axis=1, keepdims=True)
    C2 = (-R.T @ T.reshape(3)).reshape(1, 3)  # projector center in camera frame

    out = np.zeros((len(d1), 3))
    for i in range(len(d1)):
        a, b = d1[i], d2[i]
        w0 = -C2[0]  # camera center is origin: p1 = 0
        A = np.array([[a @ a, -(a @ b)], [a @ b, -(b @ b)]])
        rhs = np.array([-(a @ w0), -(b @ w0)])
        s, t = np.linalg.lstsq(A, rhs, rcond=None)[0]
        p1, p2 = s * a, C2[0] + t * b
        out[i] = 0.5 * (p1 + p2)
    return out


# ---------------- hardware capture helpers ----------------


async def _project(cli: IrisBleClient, bw: np.ndarray, settle: float = 0.8) -> None:
    """Push a 1bpp image as 1_1.bmp and keep group 1 playing (see calibrate.py)."""
    await push_image(cli, build_stream(bw), "1_1.bmp")
    await cli.stop()
    await asyncio.sleep(0.3)
    r = await cli.play(group_id=1, loop=True,
                       total_100ms=PLAY_TOTAL_100MS, interval_100ms=PLAY_INTERVAL_100MS)
    if not r.ok:
        raise RuntimeError(f"play failed: {r.error}")
    await asyncio.sleep(settle)


def _black_image() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


async def _project_verified(
    cli: IrisBleClient,
    cam: HikCamera,
    bw: np.ndarray,
    kind: str,
    board: BoardModel,
    block: int,
    c: int,
    settle: float = 0.8,
    timeout: float = 30.0,
    raw_thresh: int = 200,
    roi: tuple[float, float, float, float] | None = None,
    bg: np.ndarray | None = None,
) -> None:
    """Push+play, then verify via the camera that the display really shows it.

    The IrisGreen has an opaque UI state machine (connection-status screens can
    overlay or swallow playback for seconds) — never trust that play took
    effect. kind: "black" (no bright dots on the board) or "grid" (many bright
    dots on the board, position-stable across two consecutive frames).
    Grid checks use the board extent measured on the black frame (`roi`):
    laser dots wash out the black-dot detector, so the board cannot be
    re-found on the grid frame itself.
    """
    t0 = time.time()
    attempt = 0
    n1 = -1
    while True:
        attempt += 1
        await _project(cli, bw, settle=settle)
        cam.flush(2)
        ok1, f1 = cam.read(1000)
        if not ok1 or f1 is None:
            continue
        if kind == "black" and bg is not None:
            # Specular highlights of room lights sit on the board and look like
            # laser dots in absolute threshold. Subtract the pre-capture black
            # frame so only content that newly appeared (the grid) counts.
            gained = detect_grid_dots(f1, bg, thresh=50)
            if roi is not None:
                x0, y0, x1, y1 = roi
                gained = [p for p in gained if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
            n1 = len(gained)
            if n1 <= 2:
                return
        elif kind == "grid" and roi is not None:
            c1 = dots_in_roi(f1, roi, raw_thresh)
            n1 = len(c1)
            ok_frame = n1 >= 40
            if ok_frame:
                ok2, f2 = cam.read(1000)
                c2 = dots_in_roi(f2, roi, raw_thresh) if ok2 and f2 is not None else []
                ok_frame = len(c2) >= 40
                if ok_frame:
                    a, b = np.array(c1), np.array(c2)
                    d = np.linalg.norm(a[None] - b[:, None], axis=-1).min(axis=1)
                    ok_frame = float(np.median(d)) < 1.5
            if ok_frame:
                return
        else:
            matched_b, c1 = bright_on_board(f1, board, block, c, raw_thresh=raw_thresh)
            n1 = len(c1) if matched_b is not None else -1
            if kind == "black" and matched_b is not None and n1 <= 12:
                return
            if kind == "grid" and matched_b is not None and n1 >= 40:
                ok2, f2 = cam.read(1000)
                matched2, c2 = (
                    bright_on_board(f2, board, block, c, raw_thresh=raw_thresh)
                    if ok2 and f2 is not None else (None, [])
                )
                if matched2 is not None and len(c2) >= 40:
                    a, b = np.array(c1), np.array(c2)
                    d = np.linalg.norm(a[None] - b[:, None], axis=-1).min(axis=1)
                    if float(np.median(d)) < 1.5:
                        return
        if time.time() - t0 > timeout:
            raise RuntimeError(f"投影状态验证超时 kind={kind} (dots={n1})")
        print(f"  [verify] 投影未就绪 kind={kind} dots={n1}，重试 ({attempt})")


async def _project_one_dot(
    cli: IrisBleClient,
    cam: HikCamera,
    bw: np.ndarray,
    board: BoardModel,
    block: int,
    c: int,
    settle: float = 0.8,
    timeout: float = 30.0,
    raw_thresh: int = 200,
) -> tuple[float, float] | None:
    """Re-push a single-dot frame until the board shows exactly one bright dot.

    A one-shot play is not enough: the connection-status screen can overlay or
    swallow the first playback for many seconds. Room lights outside the board
    are ignored.
    """
    t0 = time.time()
    attempt = 0
    last_n = -1
    while time.time() - t0 < timeout:
        attempt += 1
        await _project(cli, bw, settle=settle)
        deadline = time.time() + 4.0
        while time.time() < deadline and time.time() - t0 < timeout:
            cam.flush(1)
            ok, frame = cam.read(1000)
            if not ok or frame is None:
                continue
            _, cts = bright_on_board(frame, board, block, c, raw_thresh=raw_thresh)
            last_n = len(cts)
            if last_n == 1:
                return cts[0]
        print(f"  [anchor] 未见到单点 dots={last_n}，重试 ({attempt})")
    return None


def _open_hik(args) -> HikCamera:
    cam = HikCamera(getattr(args, "mvs_dir", None))
    devices = cam.enumerate()
    if not devices:
        raise RuntimeError("未枚举到海康相机")
    print("  devices: " + "; ".join(f"[{d['index']}]{d['interface']} {d['model']}" for d in devices))
    serial = getattr(args, "serial", None)
    if serial is None:
        usb = [d for d in devices if d["interface"] == "USB3"]
        if usb:
            serial = usb[0]["serial"]
    cam.open(serial=serial)
    print(f"  camera: {cam.info()}")
    return cam


def _cam_params(cam: HikCamera, exp_us: float, gain_db: float) -> None:
    """Switch exposure/gain and flush frames captured with the old parameters."""
    cam.set_gain(gain_db)
    cam.set_exposure_us(exp_us)
    cam.flush(3)


# ---------------- camint ----------------


def cmd_camint(args) -> int:
    board = BoardModel.load(args.layout)
    os.makedirs(args.dir, exist_ok=True)
    save_dir = os.path.join(args.dir, "intrinsics")
    os.makedirs(save_dir, exist_ok=True)

    cam = _open_hik(args)
    _cam_params(cam, args.exp_board, args.gain_board)
    print(f"=== CAMINT: 空格采集 / Esc 结束求解 (目标 {args.views} 姿态) ===")
    views: list[dict] = []
    last = None
    try:
        while True:
            ok, frame = cam.read(1000)
            if not ok or frame is None:
                continue
            matched = match_lattice(
                detect_board_dots(frame, args.board_block, args.board_c), board)
            vis = frame.copy()
            if matched:
                for (c, r), (x, y) in matched.items():
                    cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 0), 2)
            n = len(matched) if matched else 0
            cv2.putText(vis, f"dots={n} views={len(views)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imshow("camint", cv2.resize(vis, (960, 768)))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc
                break
            if key == 32 and matched and len(matched) >= args.min_dots:  # Space
                views.append(matched)
                last = os.path.join(save_dir, f"view{len(views):02d}.png")
                cv2.imwrite(last, frame)
                print(f"  view {len(views)}: {len(matched)} dots -> {last}")
                if len(views) >= args.views:
                    break
    finally:
        cv2.destroyAllWindows()
        cam.close()

    if len(views) < 8:
        print(f"[error] 姿态太少 ({len(views)}), 至少 8")
        return 2
    K, dist, rms = solve_camera_intrinsics(views, board, cam.actual_size)
    print(f"=== SOLVE: RMS={rms:.3f}px ===")
    print(f"  K=\n{K}\n  dist={dist.ravel()}")
    np.savez(
        os.path.join(args.dir, CAM_INT_NPZ),
        K=K, dist=dist, rms=rms, img_size=np.array(cam.actual_size),
        meta=np.array(json.dumps({"views": len(views), "board": board.cols * board.rows})),
    )
    print(f"  -> {os.path.join(args.dir, CAM_INT_NPZ)}")
    return 0


# ---------------- collect ----------------


async def _capture_pose(
    cli: IrisBleClient,
    cam: HikCamera,
    board: BoardModel,
    K: np.ndarray,
    dist: np.ndarray,
    args,
    H_boot: np.ndarray | None,
    grid_img: np.ndarray,
    grid_pts: list[tuple[float, float]],
    pose_dir: str,
    board_frame: np.ndarray,
) -> dict | None:
    """One pose at a single exposure (40ms/gain8, no parameter switching):
    board frame comes from the preview (projection verified black); then push
    the grid with camera-verified display, capture a single laser frame, and
    restore black (verified). The board must not move during the sequence."""
    os.makedirs(pose_dir, exist_ok=True)

    matched = match_lattice(
        detect_board_dots(board_frame, args.board_block, args.board_c), board)
    if not matched or len(matched) < args.min_dots:
        print(f"  board dots insufficient ({len(matched) if matched else 0})")
        return None
    rvec, tvec = board_pose(matched, board, K, dist)
    cv2.imwrite(os.path.join(pose_dir, "board.png"), board_frame)
    roi = board_roi(matched)

    await _project_verified(
        cli, cam, grid_img, "grid", board, args.board_block, args.board_c,
        settle=args.settle, raw_thresh=args.dot_raw_thresh, roi=roi)
    cam.flush(2)
    ok, laser_frame = cam.read(1500)
    if not ok or laser_frame is None:
        print("  laser frame capture failed")
        return None
    cv2.imwrite(os.path.join(pose_dir, "grid.png"), laser_frame)
    try:
        await _project_verified(
            cli, cam, _black_image(), "black", board, args.board_block, args.board_c,
            settle=0.4, raw_thresh=args.dot_raw_thresh, roi=roi, bg=board_frame)
    except RuntimeError as e:
        print(f"  [warn] 恢复全黑未确认：{e}")
        await _project(cli, _black_image(), settle=0.3)

    centroids = dots_in_roi(laser_frame, roi, args.dot_raw_thresh)
    if H_boot is None:
        print("  no bootstrap homography yet")
        return None
    pairs = assign_grid_indices(centroids, H_boot, grid_pts, tol_px=args.assign_tol)
    print(f"  board dots={len(matched)} laser dots={len(centroids)} matched={len(pairs)}")
    if len(pairs) < 8:
        print("  too few laser matches")
        return None

    cam_xy = np.array([c for _, c in pairs])
    X_cam = rays_to_board(cam_xy, K, dist, rvec, tvec)
    X_board = to_board_frame(X_cam, rvec, tvec)
    record = {
        "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(),
        "pairs": [{"proj": list(p), "cam": list(c)} for p, c in pairs],
        "X_board": X_board.tolist(),
    }
    with open(os.path.join(pose_dir, "record.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    return record


async def run_collect(args) -> int:
    board = BoardModel.load(args.layout)
    os.makedirs(args.dir, exist_ok=True)
    intr = np.load(os.path.join(args.dir, CAM_INT_NPZ))
    K, dist = intr["K"], intr["dist"]

    cli = IrisBleClient(args.address)
    print(f"=== CONNECT {args.address} ===")
    await cli.connect()
    r = await cli.send_command(P.cmd_delete_group(1))
    print(f"  clear group 1: {'ok' if r.ok else r.error}")
    cam = _open_hik(args)
    # one exposure for the whole session: board dots + saturating laser dots
    _cam_params(cam, args.exp_board, args.gain_board)
    grid_img, grid_pts = make_grid_image(args.grid_cols, args.grid_rows, args.grid_margin, args.grid_radius)

    saved: list[dict] = []
    poses_root = os.path.join(args.dir, "poses")
    if os.path.isdir(poses_root):
        for name in sorted(os.listdir(poses_root)):
            rec_path = os.path.join(poses_root, name, "record.json")
            if not os.path.isfile(rec_path):
                continue
            with open(rec_path, encoding="utf-8") as f:
                saved.append(json.load(f))

    H_boot = None
    try:
        if saved:
            proj = np.array([p["proj"] for p in saved[-1]["pairs"]], dtype=np.float64)
            cam_xy = np.array([p["cam"] for p in saved[-1]["pairs"]], dtype=np.float64)
            if len(proj) >= 8:
                H_boot, _ = cv2.findHomography(
                    proj.reshape(-1, 1, 2), cam_xy.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
            if H_boot is not None:
                print(f"=== RESUME: H_boot from last saved pose ({len(proj)} pairs) ===")
        if H_boot is None:
            # bootstrap: 5 anchor dots, each camera-verified (device UI state
            # machine can overlay/swallow playback — poll until exactly 1 dot)
            print("=== BOOTSTRAP (5 anchor dots, camera-verified) ===")
            await _project(cli, _black_image(), settle=0.6)
            anchors = [(WIDTH * 0.2, HEIGHT * 0.2), (WIDTH * 0.8, HEIGHT * 0.2),
                       (WIDTH * 0.5, HEIGHT * 0.5), (WIDTH * 0.2, HEIGHT * 0.8),
                       (WIDTH * 0.8, HEIGHT * 0.8)]
            ap, ac = [], []
            for i, (px, py) in enumerate(anchors, 1):
                img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                cv2.circle(img, (int(px), int(py)), 12, 1, -1)
                found = await _project_one_dot(
                    cli, cam, img, board, args.board_block, args.board_c,
                    settle=args.settle, raw_thresh=args.dot_raw_thresh)
                if found is not None:
                    ap.append((px, py))
                    ac.append(found)
                    print(f"  anchor {i}: proj=({px:.0f},{py:.0f}) -> cam=({found[0]:.1f},{found[1]:.1f})")
                else:
                    print(f"  anchor {i}: no single dot within 30s")
            if len(ap) >= 4:
                H_boot, _ = cv2.findHomography(
                    np.array(ap).reshape(-1, 1, 2), np.array(ac).reshape(-1, 1, 2), 0)
                print(f"  H_boot from {len(ap)} anchors")
            else:
                print("[error] 锚点不足，检查投影后重试")
                return 2

        print("=== COLLECT: 空格=采集当前姿态 / r=清空重采 / Esc=结束 ===")
        try:
            await _project_verified(
                cli, cam, _black_image(), "black", board, args.board_block, args.board_c,
                settle=0.4, raw_thresh=args.dot_raw_thresh, timeout=12)
        except RuntimeError as e:
            print(f"  [warn] {e}，进入预览")
            await _project(cli, _black_image(), settle=0.3)
        n_done = 0
        ref_X: dict[tuple[float, float], list[np.ndarray]] = {}
        for prev in saved:
            Rb, _ = cv2.Rodrigues(np.array(prev["rvec"], dtype=np.float64))
            tvec = np.array(prev["tvec"], dtype=np.float64)
            X_cam = (Rb @ np.array(prev["X_board"]).T).T + tvec
            for p, X in zip(prev["pairs"], X_cam):
                ref_X.setdefault(tuple(p["proj"]), []).append(X)
            n_done += 1
        if n_done:
            print(f"  resume from {n_done} saved poses")

        def _line_check(rec: dict) -> float | None:
            """Median distance of the new pose's 3D points to per-pixel rays
            fitted through previously accepted poses (the true rigidity
            invariant: projector rays are fixed in the camera frame)."""
            Rb, _ = cv2.Rodrigues(np.array(rec["rvec"], dtype=np.float64))
            tvec = np.array(rec["tvec"], dtype=np.float64)
            X_cam = (Rb @ np.array(rec["X_board"]).T).T + tvec
            errs = []
            for p, X in zip(rec["pairs"], X_cam):
                prev = ref_X.get(tuple(p["proj"]))
                if not prev or len(prev) < 2:
                    continue
                Pp = np.array(prev)
                c = Pp.mean(axis=0)
                d = Pp[1] - Pp[0] if len(Pp) == 2 else \
                    np.linalg.svd(Pp - c, full_matrices=False)[2][0]
                d = d / np.linalg.norm(d)
                errs.append(float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - c))))
            return float(np.median(errs)) if len(errs) >= 8 else None

        while n_done < args.poses:
            ok, frame = cam.read(1000)
            if not ok or frame is None:
                continue
            matched = match_lattice(
                detect_board_dots(frame, args.board_block, args.board_c), board)
            vis = frame.copy()
            if matched:
                for x, y in matched.values():
                    cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 0), 2)
            n_dots = len(matched) if matched else 0
            cv2.putText(vis, f"dots={n_dots} poses={n_done}/{args.poses} "
                        "[SPACE]=capture [r]=restart [Esc]=done", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("collect", cv2.resize(vis, (960, 768)))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("r"):
                shutil.rmtree(os.path.join(args.dir, "poses"), ignore_errors=True)
                ref_X.clear()
                n_done = 0
                print("  [restart] 已清空所有姿态，重新采集")
                continue
            if key == 32 and matched and n_dots >= args.min_dots:
                pose_dir = os.path.join(args.dir, "poses", f"pose{n_done + 1:02d}")
                cv2.putText(vis, "CAPTURING...", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("collect", cv2.resize(vis, (960, 768)))
                cv2.waitKey(1)
                try:
                    rec = await _capture_pose(cli, cam, board, K, dist, args,
                                              H_boot, grid_img, grid_pts, pose_dir, frame)
                except RuntimeError as e:
                    print(f"  [error] 采集异常：{e} —— 调整后可继续按空格重试")
                    await _project(cli, _black_image(), settle=0.3)
                    continue
                if rec is None:
                    print("  本姿态无效，调整板后重试")
                    continue
                line_err = _line_check(rec)
                if line_err is not None and line_err > 3.0:
                    shutil.rmtree(pose_dir, ignore_errors=True)
                    print(f"  [警告] 3D 线一致性失败 ({line_err:.2f}mm > 3mm)——"
                          "装置被碰动或投影状态异常，本姿态作废。恢复后空格重试；按 r 清空重采")
                    continue
                if line_err is not None:
                    print(f"  line-consistency ok ({line_err:.2f}mm)")
                Rb, _ = cv2.Rodrigues(np.array(rec["rvec"], dtype=np.float64))
                tvec = np.array(rec["tvec"], dtype=np.float64)
                X_cam = (Rb @ np.array(rec["X_board"]).T).T + tvec
                for p, X in zip(rec["pairs"], X_cam):
                    ref_X.setdefault(tuple(p["proj"]), []).append(X)
                n_done += 1
                print(f"  pose {n_done}/{args.poses} saved")
        cv2.destroyAllWindows()
        await _project(cli, _black_image(), settle=0.3)
        print(f"=== COLLECT done: {n_done} poses ===")
        return 0 if n_done >= 6 else 2
    finally:
        cam.close()
        await cli.disconnect()


# ---------------- ray field (non-pinhole projector model) ----------------


def fit_ray_field(
    records: list[dict],
    min_samples: int = 5,
    outlier_rounds: int = 3,
) -> dict:
    """Per-pixel empirical rays in the CAMERA frame (rig must be rigid).

    Pixel j's laser dot lies on one physical ray across poses; fit that line by
    PCA with adaptive outlier rejection. Returns {"grid_pts", "origins", "dirs",
    "rms", "samples", "center"} — origins/dirs are (N,3) arrays.
    """
    from collections import defaultdict

    per_px: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for r in records:
        Rb, _ = cv2.Rodrigues(np.array(r["rvec"], dtype=np.float64))
        tvec = np.array(r["tvec"], dtype=np.float64)
        Xb = np.array(r["X_board"], dtype=np.float64)
        X_cam = (Rb @ Xb.T).T + tvec
        for p, X in zip(r["pairs"], X_cam):
            per_px[tuple(p["proj"])].append(X)

    keys, origins, dirs, rms_list, ns = [], [], [], [], []
    for key, Xs in sorted(per_px.items()):
        X = np.array(Xs)
        if len(X) < min_samples:
            continue
        for _ in range(outlier_rounds):
            c = X.mean(axis=0)
            _, _, vt = np.linalg.svd(X - c, full_matrices=False)
            d = vt[0]
            dd = np.linalg.norm((X - c) - ((X - c) @ d)[:, None] * d[None, :], axis=1)
            keep = dd <= max(0.5, float(np.median(dd)) * 2.5)
            if keep.all() or int(keep.sum()) < 4:
                break
            X = X[keep]
        c = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - c, full_matrices=False)
        d = vt[0] / np.linalg.norm(vt[0])
        dd = np.linalg.norm((X - c) - ((X - c) @ d)[:, None] * d[None, :], axis=1)
        keys.append(key)
        origins.append(c)
        dirs.append(d)
        rms_list.append(float(np.sqrt((dd ** 2).mean())))
        ns.append(len(X))

    origins = np.array(origins)
    dirs = np.array(dirs)
    # concurrency reference center (least squares)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(origins, dirs):
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ c
    center = np.linalg.solve(A, b)
    return {
        "grid_pts": np.array(keys, dtype=np.float64),
        "origins": origins,
        "dirs": dirs,
        "rms": np.array(rms_list),
        "samples": np.array(ns),
        "center": center,
    }


def ray_field_lookup(field: dict, proj_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate origins/dirs at arbitrary projector pixels.

    The field is sampled on a regular grid (grid_pts). Bilinear over the
    enclosing cell; nearest fallback outside. Directions renormalized.
    """
    gp = field["grid_pts"]
    xs = np.sort(np.unique(gp[:, 0]))
    ys = np.sort(np.unique(gp[:, 1]))
    org_grid = np.full((len(ys), len(xs), 3), np.nan)
    dir_grid = np.full((len(ys), len(xs), 3), np.nan)
    xi = {x: i for i, x in enumerate(xs)}
    yi = {y: i for i, y in enumerate(ys)}
    for (x, y), o, d in zip(gp, field["origins"], field["dirs"]):
        org_grid[yi[y], xi[x]] = o
        dir_grid[yi[y], xi[x]] = d

    def interp(grid, u, v):
        fx = np.interp(u, xs, np.arange(len(xs)))
        fy = np.interp(v, ys, np.arange(len(ys)))
        x0, y0 = int(np.floor(fx)), int(np.floor(fy))
        x1, y1 = min(x0 + 1, len(xs) - 1), min(y0 + 1, len(ys) - 1)
        tx, ty = fx - x0, fy - y0
        out = np.zeros(3)
        wsum = 0.0
        for xx, wx in ((x0, 1 - tx), (x1, tx)):
            for yy, wy in ((y0, 1 - ty), (y1, ty)):
                g = grid[yy, xx]
                if not np.isnan(g).any():
                    out += wx * wy * g
                    wsum += wx * wy
        return out / wsum if wsum > 0 else None

    origins_out, dirs_out = [], []
    for u, v in np.asarray(proj_xy, dtype=np.float64):
        o = interp(org_grid, u, v)
        d = interp(dir_grid, u, v)
        if o is None or d is None:
            o, d = np.full(3, np.nan), np.full(3, np.nan)
        else:
            d = d / np.linalg.norm(d)
        origins_out.append(o)
        dirs_out.append(d)
    return np.array(origins_out), np.array(dirs_out)


def triangulate_rayfield(
    cam_xy: np.ndarray,
    proj_xy: np.ndarray,
    K_c: np.ndarray,
    dist_c: np.ndarray,
    field: dict,
) -> np.ndarray:
    """Midpoint triangulation: camera ray (pinhole) x projector ray (field)."""
    nc = cv2.undistortPoints(
        np.asarray(cam_xy, dtype=np.float64).reshape(-1, 1, 2), K_c, dist_c
    ).reshape(-1, 2)
    d1 = np.hstack([nc, np.ones((len(nc), 1))])
    d1 = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    origins, dirs = ray_field_lookup(field, np.asarray(proj_xy, dtype=np.float64))

    out = np.full((len(d1), 3), np.nan)
    for i in range(len(d1)):
        a, b = d1[i], dirs[i]
        if np.isnan(b).any():
            continue
        w0 = -origins[i]  # camera center is origin
        A = np.array([[a @ a, -(a @ b)], [a @ b, -(b @ b)]])
        rhs = np.array([-(a @ w0), -(b @ w0)])
        s, t = np.linalg.lstsq(A, rhs, rcond=None)[0]
        p1, p2 = s * a, origins[i] + t * b
        out[i] = 0.5 * (p1 + p2)
    return out


def _dirs_toward_scene(dirs: np.ndarray) -> np.ndarray:
    """Flip ray directions so they point away from the camera (+Z)."""
    d = np.array(dirs, dtype=np.float64, copy=True)
    if d.ndim == 1:
        return -d if d[2] < 0 else d
    d[d[:, 2] < 0] *= -1
    return d


def camera_rays(cam_xy: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Unit rays in the camera frame for image points. Camera center is the origin."""
    nc = cv2.undistortPoints(
        np.asarray(cam_xy, dtype=np.float64).reshape(-1, 1, 2), K, dist
    ).reshape(-1, 2)
    d = np.hstack([nc, np.ones((len(nc), 1))])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def point_on_camera_ray(
    cam_xy: tuple[float, float] | np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    z: float,
) -> np.ndarray:
    """3D point on the camera ray at camera-frame depth z (mm)."""
    d = camera_rays(np.asarray(cam_xy, dtype=np.float64).reshape(1, 2), K, dist)[0]
    return d * (float(z) / d[2])


def reference_depth(field: dict) -> float:
    """Typical surface depth (mm) from the ray samples used to fit the field."""
    return float(np.median(field["origins"][:, 2]))


def _point_ray_distance(origins: np.ndarray, dirs: np.ndarray, X: np.ndarray) -> np.ndarray:
    w = np.asarray(X, dtype=np.float64).reshape(1, 3) - origins
    along = np.sum(w * dirs, axis=1, keepdims=True)
    return np.linalg.norm(w - along * dirs, axis=1)


def proj_pixel_for_point(X: np.ndarray, field: dict) -> tuple[float, float] | None:
    """Projector pixel whose ray passes closest to a camera-frame 3D point.

    Searches the fitted grid, then refines inside the cell. Returns None when
    the point is far from every ray (outside the calibrated bundle).
    """
    dirs = _dirs_toward_scene(field["dirs"])
    dist = _point_ray_distance(field["origins"], dirs, X)
    i = int(np.argmin(dist))
    if float(dist[i]) > 40.0:
        return None
    p = field["grid_pts"][i].astype(np.float64).copy()
    xs = np.sort(np.unique(field["grid_pts"][:, 0]))
    ys = np.sort(np.unique(field["grid_pts"][:, 1]))
    step = float(min(np.diff(xs).min(), np.diff(ys).min()))
    x0, x1 = float(xs[0]), float(xs[-1])
    y0, y1 = float(ys[0]), float(ys[-1])

    def score(uv: np.ndarray) -> float:
        o, d = ray_field_lookup(field, uv.reshape(1, 2))
        if np.isnan(d).any():
            return 1e9
        d = _dirs_toward_scene(d)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        return float(_point_ray_distance(o, d, X)[0])

    best = score(p)
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for _ in range(12):
        improved = False
        for du, dv in offsets:
            q = np.array([
                float(np.clip(p[0] + du * step, x0, x1)),
                float(np.clip(p[1] + dv * step, y0, y1)),
            ])
            s = score(q)
            if s + 1e-6 < best:
                best, p, improved = s, q, True
        if not improved:
            step *= 0.5
            if step < 0.25:
                break
    return float(p[0]), float(p[1])


def proj_pixel_for_camera(
    cam_xy: tuple[float, float],
    K: np.ndarray,
    dist: np.ndarray,
    field: dict,
    z: float | None = None,
) -> tuple[float, float] | None:
    """Open-loop aim: camera pixel -> projector pixel at a reference depth."""
    if z is None:
        z = reference_depth(field)
    X = point_on_camera_ray(cam_xy, K, dist, z)
    return proj_pixel_for_point(X, field)


def plane_homography_from_rays(
    K: np.ndarray,
    dist: np.ndarray,
    field: dict,
    z: float,
) -> np.ndarray:
    """Camera-pixel -> projector-pixel homography on the plane at depth z.

    Same role as the old wall homography: draw a mask in the camera image and
    warp it into the projector frame. The correspondences come from intersecting
    the calibrated ray field with that plane.
    """
    dirs = _dirs_toward_scene(field["dirs"])
    denom = dirs[:, 2].copy()
    denom[np.abs(denom) < 1e-8] = np.nan
    t = (float(z) - field["origins"][:, 2]) / denom
    ok = np.isfinite(t) & (t * dirs[:, 2] + field["origins"][:, 2] > 50)
    Q = field["origins"][ok] + t[ok, None] * dirs[ok]
    cam, _ = cv2.projectPoints(
        Q.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    cam = cam.reshape(-1, 2)
    proj = field["grid_pts"][ok]
    finite = np.isfinite(cam).all(axis=1)
    if int(finite.sum()) < 8:
        raise RuntimeError("ray field does not cover the reference plane")
    H, _ = cv2.findHomography(cam[finite], proj[finite], cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("could not fit a camera-to-projector homography")
    return H


def shape_light_mask(
    polygon: np.ndarray,
    camera_size: tuple[int, int],
    H_cam_to_proj: np.ndarray,
) -> np.ndarray:
    """Fill a freehand camera polygon and warp it to a 640x480 projector mask.

    Matches the old 2D shape mode (fillPoly + warpPerspective), with H taken
    from the ray field instead of the flat-wall calibration.
    """
    width, height = camera_size
    camera_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(camera_mask, [np.rint(polygon).astype(np.int32)], 1)
    warped = cv2.warpPerspective(
        camera_mask, H_cam_to_proj, (WIDTH, HEIGHT),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (warped > 0).astype(np.uint8)


# ---------------- solve / verify ----------------


def _load_poses(args) -> list[dict]:
    poses_dir = os.path.join(args.dir, "poses")
    out = []
    for name in sorted(os.listdir(poses_dir)):
        p = os.path.join(poses_dir, name, "record.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def cmd_solve(args) -> int:
    intr = np.load(os.path.join(args.dir, CAM_INT_NPZ))
    K_c, dist_c = intr["K"], intr["dist"]
    records = _load_poses(args)
    if len(records) < 6:
        print(f"[error] 有效姿态不足 ({len(records)} < 6)")
        return 2

    obj_views = [np.array(r["X_board"], dtype=np.float64) for r in records]
    proj_views = [np.array([p["proj"] for p in r["pairs"]], dtype=np.float64) for r in records]
    cam_views = [np.array([p["cam"] for p in r["pairs"]], dtype=np.float64) for r in records]

    # projector intrinsics with adaptive worst-view pruning (max 3 rounds, keep >= 6)
    dropped: list[int] = []
    for _ in range(3):
        K_p, dist_p, rms_p, rvecs, tvecs = solve_projector(obj_views, proj_views)
        errs = per_view_reproj_errors(obj_views, proj_views, K_p, dist_p, rvecs, tvecs)
        keep = prune_views(errs)
        if len(keep) == len(errs) or len(keep) < 6:
            break
        worst = int(np.argmax(errs))
        print(f"  prune pose{worst + 1:02d}: reproj {errs[worst]:.2f}px "
                  f"(median {np.median(errs):.2f})")
        dropped.append(worst)
        obj_views = [o for i, o in enumerate(obj_views) if i != worst]
        proj_views = [p for i, p in enumerate(proj_views) if i != worst]
        cam_views = [c for i, c in enumerate(cam_views) if i != worst]
    print(f"  projector per-view reproj: {np.round(errs, 3).tolist()}"
          + (f" (dropped {dropped})" if dropped else ""))

    R, T, rms_s = solve_stereo(obj_views, cam_views, proj_views, K_c, dist_c, K_p, dist_p)
    baseline = float(np.linalg.norm(T))

    # independent per-pose cross-check: board pose (camera side) -> X_board to
    # camera frame -> R,T into projector frame -> project -> vs projector pixels
    cross_errs = []
    for r, o, pv in zip(records, obj_views, proj_views):
        rvec = np.array(r["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.array(r["tvec"], dtype=np.float64).reshape(3, 1)
        Rb, _ = cv2.Rodrigues(rvec)
        X_cam = (Rb @ o.T).T + tvec.reshape(3)
        X_proj = (R @ X_cam.T).T + T.reshape(3)
        rp, _ = cv2.Rodrigues(np.eye(3))
        proj, _ = cv2.projectPoints(X_proj, rp, np.zeros(3), K_p, dist_p)
        cross_errs.append(float(np.linalg.norm(
            proj.reshape(-1, 2) - pv.reshape(-1, 2), axis=1).mean()))

    print("=== PROJECTOR ===")
    print(f"  K_p=\n{K_p}\n  dist_p={dist_p.ravel()}\n  RMS={rms_p:.3f}px (in 640x480)")
    print("=== STEREO ===")
    print(f"  R=\n{R}\n  T={T.round(2)} (|T|={baseline:.1f}mm)  RMS={rms_s:.3f}px")
    print(f"  cross-check per-pose (board-pose->proj px): {np.round(cross_errs, 3).tolist()}")

    # --- empirical ray field (primary model; the dual-mirror MEMS projector is
    # not a pinhole device, K_p is a rough approximation only) ---
    field = fit_ray_field(records)
    fr = field["rms"]
    dc = np.array([
        np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (field["center"] - c))
        for c, d in zip(field["origins"], field["dirs"])
    ])
    print("=== RAY FIELD ===")
    print(f"  rays={len(fr)}  line-fit RMS: mean={fr.mean():.2f}mm "
          f"median={np.median(fr):.2f} max={fr.max():.2f}mm")
    print(f"  concurrency center={field['center'].round(1)}mm  "
          f"ray-to-center mean={dc.mean():.2f}mm max={dc.max():.2f}mm")
    if np.median(fr) > 1.0:
        print("  [warn] 线拟合残差过大——装置在采集中很可能被碰动过，数据不可信，建议重采")

    np.savez(
        os.path.join(args.dir, PROCALIB_NPZ),
        K_c=K_c, dist_c=dist_c, K_p=K_p, dist_p=dist_p, R=R, T=T,
        ray_grid_pts=field["grid_pts"], ray_origins=field["origins"],
        ray_dirs=field["dirs"], ray_rms=field["rms"], ray_center=field["center"],
        meta=np.array(json.dumps({"poses": len(obj_views), "dropped": dropped,
                                  "rms_proj": rms_p, "rms_stereo": rms_s,
                                  "baseline_mm": baseline,
                                  "rayfield_rms_median": float(np.median(fr))})),
    )
    with open(os.path.join(args.dir, REPORT_JSON), "w", encoding="utf-8") as f:
        json.dump({"poses": len(obj_views), "dropped": dropped,
                   "rms_camera": float(intr["rms"]),
                   "rms_projector": rms_p, "rms_stereo": rms_s,
                   "proj_per_view_reproj": errs.tolist(),
                   "cross_check_per_pose": cross_errs,
                   "rayfield": {"rays": len(fr), "rms_mean": float(fr.mean()),
                                "rms_median": float(np.median(fr)),
                                "rms_max": float(fr.max()),
                                "center": field["center"].tolist(),
                                "ray_to_center_mean": float(dc.mean())},
                   "baseline_mm": baseline}, f, ensure_ascii=False, indent=2)
    print(f"  -> {os.path.join(args.dir, PROCALIB_NPZ)} / {REPORT_JSON}")
    return 0


async def run_verify(args) -> int:
    calib = np.load(os.path.join(args.dir, PROCALIB_NPZ))
    K_c, dist_c = calib["K_c"], calib["dist_c"]
    K_p, dist_p, R, T = calib["K_p"], calib["dist_p"], calib["R"], calib["T"]
    field = None
    if "ray_grid_pts" in calib.files:
        field = {"grid_pts": calib["ray_grid_pts"], "origins": calib["ray_origins"],
                 "dirs": calib["ray_dirs"]}
        print("  verify 使用光线场模型（逐像素光线插值）")
    else:
        print("  verify 使用针孔模型")
    board = BoardModel.load(args.layout)

    cli = IrisBleClient(args.address)
    await cli.connect()
    await cli.send_command(P.cmd_delete_group(1))
    cam = _open_hik(args)
    _cam_params(cam, args.exp_board, args.gain_board)  # one exposure throughout
    grid_img, grid_pts = make_grid_image(args.grid_cols, args.grid_rows, args.grid_margin, args.grid_radius)

    H_boot = None
    poses_root = os.path.join(args.dir, "poses")
    if os.path.isdir(poses_root):
        recs = []
        for name in sorted(os.listdir(poses_root)):
            rec_path = os.path.join(poses_root, name, "record.json")
            if os.path.isfile(rec_path):
                with open(rec_path, encoding="utf-8") as f:
                    recs.append(json.load(f))
        if recs and len(recs[-1]["pairs"]) >= 8:
            proj = np.array([p["proj"] for p in recs[-1]["pairs"]], dtype=np.float64)
            cam_xy = np.array([p["cam"] for p in recs[-1]["pairs"]], dtype=np.float64)
            H_boot, _ = cv2.findHomography(
                proj.reshape(-1, 1, 2), cam_xy.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
            if H_boot is not None:
                print(f"  H_boot from last pose ({len(proj)} pairs)")
    if H_boot is None:
        await _project(cli, _black_image(), settle=0.6)
        anchors = [(WIDTH * 0.2, HEIGHT * 0.2), (WIDTH * 0.8, HEIGHT * 0.2),
                   (WIDTH * 0.5, HEIGHT * 0.5), (WIDTH * 0.2, HEIGHT * 0.8),
                   (WIDTH * 0.8, HEIGHT * 0.8)]
        ap, ac = [], []
        for px, py in anchors:
            img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            cv2.circle(img, (int(px), int(py)), 12, 1, -1)
            found = await _project_one_dot(
                cli, cam, img, board, args.board_block, args.board_c,
                settle=args.settle, raw_thresh=args.dot_raw_thresh)
            if found is not None:
                ap.append((px, py))
                ac.append(found)
        if len(ap) < 4:
            print("[error] verify 锚点检测失败")
            return 2
        H_boot, _ = cv2.findHomography(
            np.array(ap).reshape(-1, 1, 2), np.array(ac).reshape(-1, 1, 2), 0)

    try:
        print("=== VERIFY: 空格=验证当前姿态 / Esc=结束 ===")
        try:
            await _project_verified(
                cli, cam, _black_image(), "black", board, args.board_block, args.board_c,
                settle=0.4, raw_thresh=args.dot_raw_thresh, timeout=12)
        except RuntimeError as e:
            print(f"  [warn] {e}，进入预览")
            await _project(cli, _black_image(), settle=0.3)
        v = 0
        while v < args.poses:
            ok, frame = cam.read(1000)
            if not ok or frame is None:
                continue
            matched = match_lattice(
                detect_board_dots(frame, args.board_block, args.board_c), board)
            vis = frame.copy()
            if matched:
                for x, y in matched.values():
                    cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 0), 2)
            cv2.putText(vis, f"verify {v}/{args.poses} [SPACE]=run [Esc]=done",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("verify", cv2.resize(vis, (960, 768)))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key != 32 or not matched:
                continue

            rvec, tvec = board_pose(matched, board, K_c, dist_c)
            roi = board_roi(matched)
            await _project_verified(
                cli, cam, grid_img, "grid", board, args.board_block, args.board_c,
                settle=args.settle, raw_thresh=args.dot_raw_thresh, roi=roi)
            cam.flush(2)
            ok, laser = cam.read(1500)
            if not ok or laser is None:
                continue
            await _project_verified(
                cli, cam, _black_image(), "black", board, args.board_block, args.board_c,
                settle=0.4, raw_thresh=args.dot_raw_thresh)
            centroids = dots_in_roi(laser, roi, args.dot_raw_thresh)
            pairs = assign_grid_indices(centroids, H_boot, grid_pts, tol_px=args.assign_tol)
            if len(pairs) < 8:
                print(f"  匹配点太少 ({len(pairs)})")
                continue
            cam_xy = np.array([c for _, c in pairs])
            proj_xy = np.array([p for p, _ in pairs])

            if field is not None:
                X_tri = triangulate_rayfield(cam_xy, proj_xy, K_c, dist_c, field)
            else:
                X_tri = triangulate(cam_xy, proj_xy, K_c, dist_c, K_p, dist_p, R, T)
            X_plane = rays_to_board(cam_xy, K_c, dist_c, rvec, tvec)
            valid = ~np.isnan(X_tri[:, 0])
            X_tri = X_tri[valid]
            X_plane = X_plane[valid]
            # plane fit on triangulated points
            c0 = X_tri.mean(axis=0)
            _, _, vt = np.linalg.svd(X_tri - c0)
            n = vt[-1]
            d_fit = np.abs((X_tri - c0) @ n)
            d_plane = np.linalg.norm(X_tri - X_plane, axis=1)
            print(f"  pose {v + 1}: {len(pairs)} pts | "
                  f"tri->plane-fit RMSE={np.sqrt((d_fit ** 2).mean()):.2f}mm "
                  f"max={d_fit.max():.2f}mm | "
                  f"tri vs board-plane diff mean={d_plane.mean():.2f}mm")
            v += 1
        cv2.destroyAllWindows()
        await _project(cli, _black_image(), settle=0.3)
        return 0
    finally:
        cam.close()
        await cli.disconnect()


# ---------------- CLI ----------------


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    ap.add_argument("--mvs-dir", default=None)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--exp-board", type=float, default=40000,
                    help="全程唯一曝光 us（≥激光扫描整帧周期，条纹积分消隐）")
    ap.add_argument("--gain-board", type=float, default=8.0, help="全程增益 dB")
    ap.add_argument("--board-block", type=int, default=51,
                    help="板点自适应阈值邻域块 px（约 2 倍点间距）")
    ap.add_argument("--board-c", type=int, default=8,
                    help="板点自适应阈值常数（点比邻域暗的灰度级数）")
    ap.add_argument("--dot-raw-thresh", type=int, default=200,
                    help="激光点原始亮度阈值（长曝光下激光点饱和 255）")
    ap.add_argument("--min-dots", type=int, default=60, help="板点最少匹配数")
    ap.add_argument("--grid-cols", type=int, default=10)
    ap.add_argument("--grid-rows", type=int, default=8)
    ap.add_argument("--grid-margin", type=float, default=0.12)
    ap.add_argument("--grid-radius", type=int, default=10, help="投影点半径 px")
    ap.add_argument("--assign-tol", type=float, default=25.0)
    ap.add_argument("--settle", type=float, default=0.8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="irisloop.procalib")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("camint", help="相机内参采集+求解")
    _add_common(pc)
    pc.add_argument("--views", type=int, default=18)

    pl = sub.add_parser("collect", help="投影点阵采集（每姿态两帧）")
    _add_common(pl)
    pl.add_argument("--poses", type=int, default=12,
                    help="12-16 姿态是精度平台期（仿真研究结论），最少 9")

    ps = sub.add_parser("solve", help="离线解算投影仪内外参")
    _add_common(ps)

    pv = sub.add_parser("verify", help="三角化/平面验证")
    _add_common(pv)
    pv.add_argument("--poses", type=int, default=2)

    args = ap.parse_args(argv)
    if args.cmd == "camint":
        return cmd_camint(args)
    if args.cmd == "collect":
        return asyncio.run(run_collect(args))
    if args.cmd == "solve":
        return cmd_solve(args)
    return asyncio.run(run_verify(args))


if __name__ == "__main__":
    raise SystemExit(main())
