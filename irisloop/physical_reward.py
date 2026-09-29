"""Objective, reproducible reward for projector -> camera captures.

The target is a 640x480 projector-space binary mask. The capture and a dark
background are camera frames. Laser pixels are detected by background
subtraction, warped back into projector space with the saved homography, then
compared with the target using strict and halo-tolerant geometry metrics.

Usage:
    python -m irisloop.physical_reward \
      captures/calibration/target_mask.png \
      captures/calibration/video_play_capture.png \
      --background captures/calibration/bg.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from irisloop.calibrate import Calibration, DEFAULT_OUT, _greenness
from irisloop.image_pack import HEIGHT, WIDTH


@dataclass
class StrictMetrics:
    iou: float
    precision: float
    recall: float
    target_pixels: int
    observed_pixels: int
    intersection_pixels: int
    spill_pixels: int


@dataclass
class TolerantMetrics:
    radius_px: int
    precision: float
    recall: float
    f1: float
    halo_pixels: int
    halo_ratio: float


@dataclass
class PhysicalReward:
    reward: float
    strict: StrictMetrics
    tolerant: list[TolerantMetrics]
    selected_tolerance_px: int
    halo_penalty: float
    centroid_error_px: float | None
    observed_width_median_px: float | None
    threshold: int
    threshold_sweep: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def read_image(path: str, flags: int) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buf, flags)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def laser_response(
    capture: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    if capture.shape != background.shape:
        raise ValueError(
            f"capture/background shape mismatch: {capture.shape} != {background.shape}"
        )
    score = _greenness(capture) - _greenness(background)
    dg = capture.astype(np.int16)[..., 1] - background.astype(np.int16)[..., 1]
    return np.clip(score + np.clip(dg, 0, None), 0, None).astype(np.float32)


def detect_laser(
    capture: np.ndarray,
    background: np.ndarray,
    threshold: int = 40,
) -> np.ndarray:
    score = laser_response(capture, background)
    mask = (score > threshold).astype(np.uint8)
    return cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def _strict_metrics(target: np.ndarray, observed: np.ndarray) -> StrictMetrics:
    intersection = target & observed
    union = target | observed
    target_count = int(target.sum())
    observed_count = int(observed.sum())
    intersection_count = int(intersection.sum())
    return StrictMetrics(
        iou=intersection_count / max(1, int(union.sum())),
        precision=intersection_count / max(1, observed_count),
        recall=intersection_count / max(1, target_count),
        target_pixels=target_count,
        observed_pixels=observed_count,
        intersection_pixels=intersection_count,
        spill_pixels=int((observed & ~target).sum()),
    )


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _tolerant_metrics(
    target: np.ndarray, observed: np.ndarray, radius: int
) -> TolerantMetrics:
    target_near = _dilate(target, radius)
    observed_near = _dilate(observed, radius)
    precision = float((observed & target_near).sum()) / max(1, int(observed.sum()))
    recall = float((target & observed_near).sum()) / max(1, int(target.sum()))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    halo = int((observed & ~target_near).sum())
    return TolerantMetrics(
        radius_px=radius,
        precision=precision,
        recall=recall,
        f1=f1,
        halo_pixels=halo,
        halo_ratio=halo / max(1, int(target.sum())),
    )


def _median_width(mask: np.ndarray) -> float | None:
    binary = mask.astype(np.uint8)
    if not binary.any():
        return None
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    local_max = distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)) - 1e-6
    widths = 2.0 * distance[local_max & (distance > 0)]
    return float(np.median(widths)) if widths.size else None


def _warp_response(
    response: np.ndarray, calibration: Calibration
) -> np.ndarray:
    camera_field = np.zeros(response.shape, dtype=np.uint8)
    field = calibration.proj_to_cam(
        [(0, 0), (WIDTH - 1, 0), (WIDTH - 1, HEIGHT - 1), (0, HEIGHT - 1)]
    )
    cv2.fillConvexPoly(camera_field, np.rint(field).astype(np.int32), 1)
    response = response * camera_field
    return cv2.warpPerspective(
        response,
        calibration.H_inv,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def assess_physical(
    target: np.ndarray,
    capture: np.ndarray,
    background: np.ndarray,
    calibration: Calibration,
    threshold: int = 40,
    tolerance_px: tuple[int, ...] = (0, 2, 4, 8),
    selected_tolerance_px: int = 4,
    halo_penalty: float = 0.25,
    threshold_sweep: tuple[int, ...] = (),
) -> tuple[PhysicalReward, np.ndarray]:
    target_mask = (
        cv2.resize(target, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST) > 127
    )
    warped_response = _warp_response(
        laser_response(capture, background), calibration
    )
    observed = (warped_response > threshold).astype(np.uint8)
    observed = cv2.morphologyEx(
        observed, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    ) > 0
    strict = _strict_metrics(target_mask, observed)
    radii = tuple(dict.fromkeys((*tolerance_px, selected_tolerance_px)))
    tolerant = [_tolerant_metrics(target_mask, observed, radius) for radius in radii]
    selected = next(item for item in tolerant if item.radius_px == selected_tolerance_px)
    reward = float(np.clip(
        selected.f1 - halo_penalty * selected.halo_ratio, 0.0, 1.0
    ))
    target_center = _centroid(target_mask)
    observed_center = _centroid(observed)
    centroid_error = (
        None
        if target_center is None or observed_center is None
        else float(np.linalg.norm(target_center - observed_center))
    )
    result = PhysicalReward(
        reward=reward,
        strict=strict,
        tolerant=tolerant,
        selected_tolerance_px=selected_tolerance_px,
        halo_penalty=halo_penalty,
        centroid_error_px=centroid_error,
        observed_width_median_px=_median_width(observed),
        threshold=threshold,
    )
    for value in threshold_sweep:
        sweep_observed = warped_response > value
        sweep_metrics = _tolerant_metrics(
            target_mask, sweep_observed, selected_tolerance_px
        )
        result.threshold_sweep[str(value)] = float(np.clip(
            sweep_metrics.f1 - halo_penalty * sweep_metrics.halo_ratio,
            0.0,
            1.0,
        ))
    return result, observed.astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="irisloop.physical_reward")
    ap.add_argument("target", help="640x480 projector-space target mask")
    ap.add_argument("capture", help="camera capture with projection on")
    ap.add_argument("--background", required=True, help="camera frame with projection off")
    ap.add_argument("--calibration", default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=int, default=40)
    ap.add_argument("--tolerance", type=int, nargs="+", default=[0, 2, 4, 8])
    ap.add_argument("--selected-tolerance", type=int, default=4)
    ap.add_argument("--halo-penalty", type=float, default=0.25)
    ap.add_argument("--threshold-sweep", type=int, nargs="*", default=[])
    ap.add_argument("--observed-out", help="optional warped observed mask output")
    args = ap.parse_args(argv)

    result, observed = assess_physical(
        read_image(args.target, cv2.IMREAD_GRAYSCALE),
        read_image(args.capture, cv2.IMREAD_COLOR),
        read_image(args.background, cv2.IMREAD_COLOR),
        Calibration.load(args.calibration),
        threshold=args.threshold,
        tolerance_px=tuple(args.tolerance),
        selected_tolerance_px=args.selected_tolerance,
        halo_penalty=args.halo_penalty,
        threshold_sweep=tuple(args.threshold_sweep),
    )
    if args.observed_out:
        cv2.imencode(".png", observed * 255)[1].tofile(args.observed_out)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
