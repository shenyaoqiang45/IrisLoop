"""Synthetic-data tests for irisloop.procalib (no hardware).

Builds a known camera-projector rig, forward-projects laser dots onto board
planes, and checks the solver recovers K_p / R / T. Also covers the lattice
matcher (fiducial anchoring, missing/outlier dots) and grid assignment.
"""

import math

import cv2
import numpy as np

from irisloop.image_pack import WIDTH, HEIGHT
from irisloop.procalib import (
    BoardModel,
    assign_grid_indices,
    make_grid_image,
    match_lattice,
    rays_to_board,
    solve_projector,
    solve_stereo,
    to_board_frame,
    triangulate,
)

BOARD = BoardModel(
    cols=11, rows=9, pitch_mm=25.0,
    fiducials={(2, 4), (5, 2), (5, 6), (6, 2), (8, 4)},
)

K_C = np.array([[1300.0, 0, 640.0], [0, 1300.0, 512.0], [0, 0, 1.0]])
DIST_C = np.array([-0.08, 0.05, 0.0003, -0.0002, 0.0])
K_P = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
DIST_P = np.array([-0.05, 0.02, 0.0, 0.0, 0.0])

# projector center 100mm to the right of the camera, yawed 7 degrees inward
_C_P = np.array([100.0, 0.0, 0.0])
_R_GT = cv2.Rodrigues(np.array([0.0, math.radians(7.0), 0.0]))[0]
T_GT = -_R_GT @ _C_P  # X_proj = _R_GT @ X_cam + T_GT
R_GT = _R_GT


def _projector_ray_to_board(proj_px, rvec_b, tvec_b):
    """proj pixel -> ray -> board plane intersection, all in camera frame."""
    norm = cv2.undistortPoints(
        np.array(proj_px, dtype=np.float64).reshape(-1, 1, 2), K_P, DIST_P
    ).reshape(-1, 2)
    d_p = np.hstack([norm, np.ones((len(norm), 1))])
    o = -R_GT.T @ T_GT                       # projector center, camera frame
    d = (R_GT.T @ d_p.T).T                   # ray dirs, camera frame
    R_b, _ = cv2.Rodrigues(rvec_b)
    n = R_b @ np.array([0.0, 0.0, 1.0])
    t = ((tvec_b - o) @ n) / (d @ n)
    return o + d * t[:, None]


def _synthetic_dataset(n_poses=12, noise_px=0.05, seed=7):
    rng = np.random.default_rng(seed)
    _, grid_pts = make_grid_image(cols=8, rows=6, margin=0.12, radius=4)
    obj_views, proj_views, cam_views, X_cam_views = [], [], [], []
    for i in range(n_poses):
        rvec_b = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5),
                           rng.uniform(-0.25, 0.25)])
        tvec_b = np.array([rng.uniform(-80, 80), rng.uniform(-60, 60),
                           rng.uniform(450, 700)])
        X_cam = _projector_ray_to_board(grid_pts, rvec_b, tvec_b)
        cam_px, _ = cv2.projectPoints(
            X_cam, np.zeros(3), np.zeros(3), K_C, DIST_C)
        cam_px = cam_px.reshape(-1, 2) + rng.normal(0, noise_px, (len(grid_pts), 2))
        inb = ((cam_px[:, 0] >= 0) & (cam_px[:, 0] < 1280)
               & (cam_px[:, 1] >= 0) & (cam_px[:, 1] < 1024))
        if inb.sum() < 20:
            continue
        X_board = to_board_frame(X_cam, rvec_b, tvec_b)
        obj_views.append(X_board[inb])
        proj_views.append(np.array(grid_pts)[inb])
        cam_views.append(cam_px[inb])
        X_cam_views.append(X_cam[inb])
    return obj_views, proj_views, cam_views, X_cam_views


def test_synthetic_rig_recovery():
    obj_views, proj_views, cam_views, X_cam_views = _synthetic_dataset()
    assert len(obj_views) >= 9

    # sanity: board-frame points are planar (z ~ 0)
    for o in obj_views:
        assert np.abs(o[:, 2]).max() < 1e-6

    K_p, dist_p, rms_p, _, _ = solve_projector(obj_views, proj_views)
    assert rms_p < 0.3
    assert abs(K_p[0, 0] - K_P[0, 0]) / K_P[0, 0] < 0.01
    assert abs(K_p[0, 2] - K_P[0, 2]) < 5

    R, T, rms_s = solve_stereo(obj_views, cam_views, proj_views,
                               K_C, DIST_C, K_p, dist_p)
    assert rms_s < 0.3
    rot_err = math.degrees(math.acos(
        min(1.0, max(-1.0, (np.trace(R @ R_GT.T) - 1) / 2))))
    assert rot_err < 0.5
    assert np.linalg.norm(T - T_GT) < 3.0

    X_tri = triangulate(cam_views[0], proj_views[0], K_C, DIST_C, K_p, dist_p, R, T)
    assert np.linalg.norm(X_tri - X_cam_views[0], axis=1).mean() < 2.0


def test_rays_to_board_roundtrip():
    rvec_b = np.array([0.3, -0.4, 0.1])
    tvec_b = np.array([20.0, -30.0, 550.0])
    _, grid_pts = make_grid_image(cols=6, rows=4, margin=0.2, radius=4)
    X_cam = _projector_ray_to_board(grid_pts[:10], rvec_b, tvec_b)
    cam_px, _ = cv2.projectPoints(X_cam, np.zeros(3), np.zeros(3), K_C, DIST_C)
    X_back = rays_to_board(cam_px.reshape(-1, 2), K_C, DIST_C, rvec_b, tvec_b)
    assert np.linalg.norm(X_back - X_cam, axis=1).max() < 1e-6


def test_match_lattice_with_missing_and_outliers():
    rng = np.random.default_rng(3)
    # affine map (col,row) -> image px: 20px/cell, rotated 25 degrees
    th = math.radians(25)
    A = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) * 20.0
    dots, truth = [], {}
    idxs = [(c, r) for c in range(11) for r in range(9)]
    rng.shuffle(idxs)
    dropped = set(idxs[:7])  # drop 7 dots
    for c, r in idxs:
        if (c, r) in dropped:
            continue
        p = A @ np.array([c, r], dtype=np.float64) + np.array([400.0, 300.0])
        p += rng.normal(0, 0.8, 2)
        area = 4000.0 if (c, r) in BOARD.fiducials else 800.0
        dots.append({"cx": p[0], "cy": p[1], "area": area})
        truth[(round(float(p[0]), 3), round(float(p[1]), 3))] = (c, r)
    # outlier blobs off the lattice
    for _ in range(4):
        dots.append({"cx": rng.uniform(0, 100), "cy": rng.uniform(0, 100),
                     "area": 800.0})

    matched = match_lattice(dots, BOARD, min_matched=40)
    assert matched is not None
    assert len(matched) >= 99 - 7 - 2
    # fiducials must be exactly right (orientation anchoring)
    for f in BOARD.fiducials - dropped:
        assert f in matched
        p = matched[f]
        assert truth[(round(p[0], 3), round(p[1], 3))] == f


def test_assign_grid_indices_shuffled_offset():
    rng = np.random.default_rng(11)
    _, grid_pts = make_grid_image(cols=8, rows=6, margin=0.12, radius=4)
    H = np.array([[1.6, 0.1, 200.0], [-0.05, 1.5, 150.0], [0.0002, -0.0001, 1.0]])
    pred = cv2.perspectiveTransform(
        np.array(grid_pts).reshape(-1, 1, 2), H).reshape(-1, 2)
    pred += 12.0  # global parallax-like offset
    keep = rng.random(len(pred)) > 0.1  # drop 10%
    cents = [tuple(p) for p in pred[keep]]
    cents += [(5.0, 5.0), (1200.0, 50.0)]  # outliers
    rng.shuffle(cents)

    pairs = assign_grid_indices(cents, H, grid_pts, tol_px=25.0)
    assert len(pairs) == int(keep.sum())
    for p, c in pairs:
        gi = grid_pts.index(p)
        assert np.linalg.norm(np.array(c) - pred[gi]) < 1e-6


def test_assign_grid_indices_empty():
    _, grid_pts = make_grid_image()
    assert assign_grid_indices([], np.eye(3), grid_pts) == []


# ---------------- ray field ----------------

from irisloop.procalib import (  # noqa: E402
    fit_ray_field, plane_homography_from_rays, point_on_camera_ray,
    proj_pixel_for_camera, proj_pixel_for_point, ray_field_lookup,
    shape_light_mask, triangulate_rayfield,
)


def _synthetic_records(n_poses=10, noise_px=0.05, seed=5):
    """Records in the on-disk collect format, from the known synthetic rig."""
    rng = np.random.default_rng(seed)
    _, grid_pts = make_grid_image(cols=8, rows=6, margin=0.12, radius=4)
    records = []
    for _ in range(n_poses):
        rvec_b = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5),
                           rng.uniform(-0.25, 0.25)])
        tvec_b = np.array([rng.uniform(-80, 80), rng.uniform(-60, 60),
                           rng.uniform(450, 700)])
        X_cam = _projector_ray_to_board(grid_pts, rvec_b, tvec_b)
        cam_px, _ = cv2.projectPoints(X_cam, np.zeros(3), np.zeros(3), K_C, DIST_C)
        cam_px = cam_px.reshape(-1, 2) + rng.normal(0, noise_px, (len(grid_pts), 2))
        inb = ((cam_px[:, 0] >= 0) & (cam_px[:, 0] < 1280)
               & (cam_px[:, 1] >= 0) & (cam_px[:, 1] < 1024))
        if inb.sum() < 20:
            continue
        X_board = to_board_frame(X_cam, rvec_b, tvec_b)
        records.append({
            "rvec": rvec_b.tolist(), "tvec": tvec_b.tolist(),
            "pairs": [{"proj": list(p), "cam": list(c)}
                      for p, c, ok in zip(grid_pts, cam_px, inb) if ok],
            "X_board": X_board[inb].tolist(),
        })
    return records, grid_pts


def test_ray_field_recovers_true_rays():
    records, grid_pts = _synthetic_records()
    assert len(records) >= 8
    field = fit_ray_field(records)
    assert len(field["rms"]) >= 30
    # line fit residuals should be at the noise level (well under 0.5mm)
    assert np.median(field["rms"]) < 0.5
    # direction of each fitted ray vs ground-truth projector ray (cam frame)
    gt_dirs = []
    for px in field["grid_pts"]:
        norm = cv2.undistortPoints(
            np.array(px, dtype=np.float64).reshape(1, 1, 2), K_P, DIST_P).reshape(2)
        d_p = np.array([norm[0], norm[1], 1.0])
        gt_dirs.append((R_GT.T @ d_p) / np.linalg.norm(d_p))
    gt_dirs = np.array(gt_dirs)
    cosang = np.abs(np.sum(field["dirs"] * gt_dirs, axis=1))
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    assert np.median(ang) < 0.05  # degrees


def test_ray_field_lookup_and_triangulation():
    records, grid_pts = _synthetic_records()
    field = fit_ray_field(records)
    # interpolate at a grid midpoint; compare with the true projector ray there
    xs = np.unique(field["grid_pts"][:, 0])
    ys = np.unique(field["grid_pts"][:, 1])
    mid = ((xs[2] + xs[3]) / 2, (ys[2] + ys[3]) / 2)
    o, d = ray_field_lookup(field, np.array([mid]))
    norm = cv2.undistortPoints(
        np.array(mid, dtype=np.float64).reshape(1, 1, 2), K_P, DIST_P).reshape(2)
    d_true = (R_GT.T @ np.array([norm[0], norm[1], 1.0]))
    d_true /= np.linalg.norm(d_true)
    cosang = float(np.clip(d[0] @ d_true, -1, 1))
    assert math.degrees(math.acos(cosang)) < 0.2

    # triangulate grid points of one pose and compare with ground truth 3D
    rec = records[0]
    cam_xy = np.array([p["cam"] for p in rec["pairs"]])
    proj_xy = np.array([p["proj"] for p in rec["pairs"]])
    rvec = np.array(rec["rvec"])
    tvec = np.array(rec["tvec"])
    X_cam_true = (cv2.Rodrigues(rvec)[0] @ np.array(rec["X_board"]).T).T + tvec
    X_tri = triangulate_rayfield(cam_xy, proj_xy, K_C, DIST_C, field)
    err = np.linalg.norm(X_tri - X_cam_true, axis=1)
    assert np.nanmean(err) < 3.0  # mm, dominated by camera-pixel noise at ~600mm


def test_proj_pixel_for_point_recovers_source_pixel():
    records, _grid_pts = _synthetic_records()
    field = fit_ray_field(records)
    rec = records[0]
    idx = len(rec["pairs"]) // 2
    rvec = np.array(rec["rvec"], dtype=np.float64)
    tvec = np.array(rec["tvec"], dtype=np.float64)
    X = cv2.Rodrigues(rvec)[0] @ np.array(rec["X_board"][idx]) + tvec
    got = proj_pixel_for_point(X, field)
    assert got is not None
    src = rec["pairs"][idx]["proj"]
    assert math.hypot(got[0] - src[0], got[1] - src[1]) < 8.0

    cam, _ = cv2.projectPoints(X.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K_C, DIST_C)
    back = point_on_camera_ray(cam.reshape(2), K_C, DIST_C, float(X[2]))
    assert np.linalg.norm(back - X) < 1.0
    aimed = proj_pixel_for_camera(tuple(cam.reshape(2)), K_C, DIST_C, field, z=float(X[2]))
    assert aimed is not None
    assert math.hypot(aimed[0] - src[0], aimed[1] - src[1]) < 8.0


def test_shape_mask_warps_through_rayfield_homography():
    records, _grid_pts = _synthetic_records()
    field = fit_ray_field(records)
    z = float(np.median(field["origins"][:, 2]))
    H = plane_homography_from_rays(K_C, DIST_C, field, z)
    # the filled camera blob is the ray's intersection with the reference plane,
    # so its warp must land on that projector pixel
    idx = len(field["grid_pts"]) // 2
    src = field["grid_pts"][idx]
    cam = proj_pixel_for_camera  # keep the open-loop aim as the reference
    X = field["origins"][idx].copy()
    d = field["dirs"][idx].copy()
    if d[2] < 0:
        d = -d
    X = X + ((z - X[2]) / d[2]) * d
    cx, cy = cv2.projectPoints(
        X.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K_C, DIST_C)[0].reshape(2)
    aimed = cam((float(cx), float(cy)), K_C, DIST_C, field, z=z)
    assert aimed is not None
    poly = np.array([
        [cx - 18, cy - 14], [cx + 16, cy - 12], [cx + 10, cy + 20], [cx - 14, cy + 12],
    ])
    mask = shape_light_mask(poly, (1280, 1024), H)
    assert mask.sum() > 0
    ys, xs = np.nonzero(mask)
    center = (float(xs.mean()), float(ys.mean()))
    assert math.hypot(center[0] - src[0], center[1] - src[1]) < 20.0
    assert math.hypot(center[0] - aimed[0], center[1] - aimed[1]) < 20.0
