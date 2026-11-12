"""Follow a moving object in camera frames and aim one laser dot at it.

Pure detection/geometry layer: no BLE, no camera, no GUI. ``tools/track_dot.py``
wires it to a projector session. Camera and projector pixels are in their own
raw image spaces; the ``mapper`` callable converts camera points to projector
points (``Calibration.cam_to_proj`` for a flat wall, or a ray-field wrapper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np

from irisloop.calibrate import make_dot_image
from irisloop.image_pack import HEIGHT, WIDTH

Point = tuple[float, float]
Mapper = Callable[[Sequence[Point]], np.ndarray]


@dataclass(frozen=True)
class Target:
    """One object observation in camera pixels."""

    center: Point
    bbox: tuple[float, float, float, float]
    area: float
    measured: bool = True  # False = coasting on the motion model, no detection


def _distance(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class MotionDetector:
    """Foreground blobs from MOG2, with the projector's own dot rejected.

    The laser dot is part of the scene and moves with the object, so a naive
    motion detector locks onto our own light. Small bright / strongly green
    components are discarded. ``candidates`` is kept separate from ``detect``
    so the selection policy can be tested without a camera.
    """

    def __init__(
        self,
        *,
        history: int = 90,
        var_threshold: float = 36.0,
        min_area: float = 300.0,
        min_fill: float = 0.20,
        roi: tuple[float, float, float, float] | None = None,
        max_jump: float = 180.0,
        laser_max_area: float = 900.0,
        laser_bright_gray: float = 215.0,
        laser_bright_fraction: float = 0.5,
        laser_green_excess: float = 28.0,
        laser_min_green: float = 120.0,
    ) -> None:
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False,
        )
        self.min_area = min_area
        self.min_fill = min_fill
        self.roi = roi
        self.max_jump = max_jump
        self.laser_max_area = laser_max_area
        self.laser_bright_gray = laser_bright_gray
        self.laser_bright_fraction = laser_bright_fraction
        self.laser_green_excess = laser_green_excess
        self.laser_min_green = laser_min_green

    def _laser_like(
        self,
        frame: np.ndarray,
        contour: np.ndarray,
        rect: tuple[int, int, int, int],
        area: float,
    ) -> bool:
        if area > self.laser_max_area:
            return False
        x, y, w, h = rect
        patch = frame[y:y + h, x:x + w]
        component = np.zeros((h, w), np.uint8)
        shifted = contour.reshape(-1, 1, 2) - np.array([x, y], dtype=np.int32)
        cv2.drawContours(component, [shifted], -1, 255, -1)
        inside = component > 0
        pixels = patch[inside]
        if pixels.size == 0:
            return False
        gray_patch = (
            cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else patch
        )
        bright_fraction = float(np.mean(gray_patch[inside] >= self.laser_bright_gray))
        if bright_fraction >= self.laser_bright_fraction:
            return True
        if frame.ndim == 3 and self.laser_green_excess > 0:
            b, g, r = (float(v) for v in pixels.mean(axis=0))
            if g - max(b, r) >= self.laser_green_excess and g >= self.laser_min_green:
                return True
        return False

    def candidates(self, frame: np.ndarray) -> list[Target]:
        """All foreground blobs that look like an object rather than our dot."""
        if frame is None or frame.ndim not in (2, 3):
            raise ValueError("frame must be a BGR or grayscale image")
        foreground = self.subtractor.apply(frame)
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
        )
        foreground = cv2.dilate(
            foreground, np.ones((5, 5), np.uint8), iterations=1,
        )
        if self.roi is not None:
            height, width = foreground.shape[:2]
            x0, y0 = int(max(0, self.roi[0])), int(max(0, self.roi[1]))
            x1 = int(min(width, self.roi[2]))
            y1 = int(min(height, self.roi[3]))
            gate = np.zeros_like(foreground)
            if x1 > x0 and y1 > y0:
                gate[y0:y1, x0:x1] = 255
            foreground = cv2.bitwise_and(foreground, gate)

        found: list[Target] = []
        contours, _ = cv2.findContours(
            foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area / max(1.0, w * h) < self.min_fill:
                continue
            if self._laser_like(frame, contour, (x, y, w, h), area):
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            center = (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
            found.append(Target(center, (x, y, x + w, y + h), area))
        return found

    def detect(self, frame: np.ndarray, hint: Point | None = None) -> Target | None:
        """Pick one blob. Without a hint take the largest; with a hint prefer
        the nearest one inside ``max_jump`` so a nearby distractor cannot steal
        the track. No candidate near the hint means a miss (coast), not a jump.
        """
        candidates = self.candidates(frame)
        if not candidates:
            return None
        if hint is None:
            return max(candidates, key=lambda item: item.area)
        near = [c for c in candidates if _distance(c.center, hint) <= self.max_jump]
        if not near:
            return None
        return min(near, key=lambda item: _distance(item.center, hint))


class MovingObjectTracker:
    """Alpha-beta style constant-velocity filter over MotionDetector hits.

    A full-frame BLE upload takes seconds, so ``lead()`` extrapolates where the
    object will be when the light actually lands. When detections stop we coast
    for ``max_lost`` frames and then declare loss, so the caller can blank.
    """

    def __init__(
        self,
        detector: MotionDetector | None = None,
        *,
        max_lost: int = 8,
        smoothing: float = 0.6,
        frame_dt: float = 1.0 / 30.0,
        max_speed: float = 2000.0,
    ) -> None:
        self.detector = detector or MotionDetector()
        self.max_lost = max_lost
        self.smoothing = smoothing
        self.frame_dt = frame_dt
        self.max_speed = max_speed
        self.center: Point | None = None
        self.velocity: Point = (0.0, 0.0)
        self.lost_frames = 0
        self.last_bbox: tuple[float, float, float, float] | None = None
        self.last_area = 0.0

    @property
    def active(self) -> bool:
        return self.center is not None

    def reset(self) -> None:
        self.center = None
        self.velocity = (0.0, 0.0)
        self.lost_frames = 0
        self.last_bbox = None
        self.last_area = 0.0

    def predict(self, dt: float = 0.0) -> Point | None:
        if self.center is None:
            return None
        x, y = self.center
        vx, vy = self.velocity
        return (x + vx * dt, y + vy * dt)

    def lead(self, seconds: float) -> Point | None:
        """Aim ahead by the expected projection latency."""
        return self.predict(max(0.0, seconds))

    def update(self, frame: np.ndarray, dt: float | None = None) -> Target | None:
        step = max(1e-3, self.frame_dt if dt is None else dt)
        predicted = self.predict(step)
        measured = self.detector.detect(
            frame, hint=predicted if self.active else None,
        )
        if measured is None:
            if not self.active or predicted is None:
                return None
            self.lost_frames += 1
            if self.lost_frames > self.max_lost:
                self.reset()
                return None
            previous = self.center
            self.center = predicted
            bbox = self.last_bbox
            if bbox is not None:
                dx, dy = predicted[0] - previous[0], predicted[1] - previous[1]
                bbox = (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)
            self.last_bbox = bbox
            if bbox is None:
                bbox = (predicted[0], predicted[1], predicted[0], predicted[1])
            return Target(predicted, bbox, self.last_area, measured=False)

        if not self.active or predicted is None:
            center = measured.center
            velocity: Point = (0.0, 0.0)
        else:
            mx, my = measured.center
            px, py = predicted
            center = (
                px + self.smoothing * (mx - px),
                py + self.smoothing * (my - py),
            )
            raw_vx = (center[0] - self.center[0]) / step
            raw_vy = (center[1] - self.center[1]) / step
            speed = float(np.hypot(raw_vx, raw_vy))
            scale = min(1.0, self.max_speed / speed) if speed > 0 else 1.0
            velocity = (raw_vx * scale, raw_vy * scale)
        self.center = center
        self.velocity = velocity
        self.lost_frames = 0
        self.last_bbox = measured.bbox
        self.last_area = measured.area
        return Target(center, measured.bbox, measured.area)


def camera_to_projector(center: Point, mapper: Mapper) -> Point | None:
    """Map one camera point; return None when the mapper cannot resolve it."""
    mapped = np.asarray(mapper([center]), dtype=np.float64).reshape(-1, 2)
    if mapped.shape[0] < 1:
        return None
    x, y = mapped[0]
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return (float(x), float(y))


def dot_mask(proj_xy: Point | None, radius: int = 12) -> np.ndarray | None:
    """640x480 1bpp mask with one dot, or None when outside the light field.

    Outside means no light at all: callers must push an all-black frame. It is
    the only way to guarantee zero laser output (``stop`` falls back to the
    built-in idle animation).
    """
    if proj_xy is None or radius < 1:
        return None
    x, y = proj_xy
    if not (radius <= x <= WIDTH - radius and radius <= y <= HEIGHT - radius):
        return None
    return make_dot_image(x, y, radius)
