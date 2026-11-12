"""Single-object track: one box measurement per frame, then a constant-velocity Kalman filter.

The measurer owns identity. It is initialized from one user box and then searches
around its own previous box. The filter does not choose a new target and does
not move the search window. It estimates position and velocity so a later
latency prediction can ask where the object will be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


Point = tuple[float, float]
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class BoxMeasurement:
    """One tracker output. ``bbox`` is ``(x, y, w, h)`` in camera pixels."""

    bbox: BBox
    score: float

    @property
    def center(self) -> Point:
        x, y, width, height = self.bbox
        return (x + 0.5 * width, y + 0.5 * height)


@dataclass(frozen=True)
class Track:
    """Filtered track plus the raw box that produced this update."""

    center: Point
    velocity: Point
    bbox: BBox
    raw_bbox: BBox
    score: float
    measured: bool


class Measurer(Protocol):
    def start(self, frame: np.ndarray, bbox: BBox) -> BoxMeasurement: ...

    def track(self, frame: np.ndarray) -> BoxMeasurement: ...

    def reset(self) -> None: ...


class ConstantVelocityKalman:
    """2D constant-velocity Kalman filter.

    State is ``(x, y, vx, vy)`` in pixels and pixels per second. A missed
    measurement predicts forward. After ``max_lost`` consecutive misses the
    filter forgets the track.
    """

    def __init__(
        self,
        *,
        meas_std: float = 4.0,
        accel_std: float = 300.0,
        max_lost: int = 8,
        initial_speed_std: float = 800.0,
    ) -> None:
        if meas_std <= 0 or accel_std < 0 or max_lost < 0 or initial_speed_std < 0:
            raise ValueError("noise scales must be non-negative and meas_std > 0")
        self.meas_std = float(meas_std)
        self.accel_std = float(accel_std)
        self.max_lost = int(max_lost)
        self.initial_speed_std = float(initial_speed_std)
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.lost = 0

    @property
    def active(self) -> bool:
        return self.x is not None

    def reset(self) -> None:
        self.x = None
        self.P = None
        self.lost = 0

    def start(self, center: Point) -> None:
        px, py = center
        if not np.isfinite(px) or not np.isfinite(py):
            raise ValueError("center must be finite")
        self.x = np.array([px, py, 0.0, 0.0], dtype=np.float64)
        speed_var = self.initial_speed_std ** 2
        meas_var = self.meas_std ** 2
        self.P = np.diag([meas_var, meas_var, speed_var, speed_var])
        self.lost = 0

    def lead(self, seconds: float) -> Point | None:
        """Position this many seconds ahead of the current filtered state."""
        if self.x is None:
            return None
        ahead = max(0.0, float(seconds))
        return (
            float(self.x[0] + self.x[2] * ahead),
            float(self.x[1] + self.x[3] * ahead),
        )

    def update(self, measurement: Point | None, dt: float) -> tuple[Point, Point, bool] | None:
        """Fold in one measurement. ``None`` coasts. Returns ``(center, velocity, measured)``."""
        if self.x is None:
            if measurement is None:
                return None
            self.start(measurement)
            return self._reading(True)
        step = max(1e-3, float(dt))
        self._predict(step)
        if measurement is None or not np.isfinite(measurement).all():
            self.lost += 1
            if self.lost > self.max_lost:
                self.reset()
                return None
            return self._reading(False)
        self._correct(measurement)
        self.lost = 0
        return self._reading(True)

    def _reading(self, measured: bool) -> tuple[Point, Point, bool]:
        assert self.x is not None
        return (
            (float(self.x[0]), float(self.x[1])),
            (float(self.x[2]), float(self.x[3])),
            measured,
        )

    def _predict(self, dt: float) -> None:
        assert self.x is not None and self.P is not None
        transition = np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self.accel_std ** 2
        process = q * np.array(
            [[dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
             [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
             [dt3 / 2.0, 0.0, dt2, 0.0],
             [0.0, dt3 / 2.0, 0.0, dt2]],
            dtype=np.float64,
        )
        self.x = transition @ self.x
        self.P = transition @ self.P @ transition.T + process

    def _correct(self, measurement: Point) -> None:
        assert self.x is not None and self.P is not None
        observation = np.array([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        noise = np.eye(2, dtype=np.float64) * (self.meas_std ** 2)
        residual = np.asarray(measurement, dtype=np.float64) - observation @ self.x
        innovation = observation @ self.P @ observation.T + noise
        gain = self.P @ observation.T @ np.linalg.inv(innovation)
        self.x = self.x + gain @ residual
        identity = np.eye(4, dtype=np.float64)
        posterior = identity - gain @ observation
        self.P = posterior @ self.P @ posterior.T + gain @ noise @ gain.T


def _bbox_around(center: Point, size: tuple[float, float]) -> BBox:
    width, height = size
    return (
        center[0] - 0.5 * width,
        center[1] - 0.5 * height,
        width,
        height,
    )


def aim_camera_point(
    center: Point,
    velocity: Point,
    *,
    lead_s: float,
    max_lead_px: float = 120.0,
    max_lead_s: float = 1.0,
    min_speed: float = 40.0,
    measured: bool = True,
    previous_velocity: Point | None = None,
    reverse_speed: float = 80.0,
) -> Point | None:
    """Camera pixel where the dot should land. ``None`` means send black.

    ``lead_s`` is how long the upload will take. Only the first ``max_lead_s``
    of that delay is predicted: a 15 s upload times small velocity noise would
    otherwise pin the dot to the lead cap and swing it around the target.
    Speed below ``min_speed`` aims at the current center. A missed measurement
    does not guess. When the velocity flips, the old lead is dropped.
    """
    center_a = np.asarray(center, dtype=np.float64)
    velocity_a = np.asarray(velocity, dtype=np.float64)
    if center_a.shape != (2,) or velocity_a.shape != (2,):
        return None
    if not measured or not np.isfinite(center_a).all() or not np.isfinite(velocity_a).all():
        return None
    ahead = min(max(0.0, float(lead_s)), max(0.0, float(max_lead_s)))
    speed_now = float(np.hypot(velocity_a[0], velocity_a[1]))
    if speed_now < min_speed:
        ahead = 0.0
    if previous_velocity is not None:
        previous_a = np.asarray(previous_velocity, dtype=np.float64)
        if previous_a.shape == (2,) and np.isfinite(previous_a).all():
            speed = float(np.hypot(velocity_a[0], velocity_a[1]))
            previous_speed = float(np.hypot(previous_a[0], previous_a[1]))
            opposing = float(velocity_a @ previous_a) < 0.0
            if opposing and speed >= reverse_speed and previous_speed >= reverse_speed:
                ahead = 0.0
    offset = velocity_a * ahead
    distance = float(np.hypot(offset[0], offset[1]))
    cap = float(max_lead_px)
    if cap > 0.0 and distance > cap:
        offset *= cap / distance
    return (float(center_a[0] + offset[0]), float(center_a[1] + offset[1]))


class SingleObjectTracker:
    """UETrack-style measurements followed by :class:`ConstantVelocityKalman`.

    A measurement whose score is below ``min_score`` is a miss. The appearance
    tracker is left on its own state; only the filter coasts.
    """

    def __init__(
        self,
        measurer: Measurer,
        kalman: ConstantVelocityKalman | None = None,
        *,
        min_score: float = 0.1,
        max_dt: float = 0.5,
    ) -> None:
        self.measurer = measurer
        self.kalman = kalman or ConstantVelocityKalman()
        self.min_score = float(min_score)
        self.max_dt = float(max_dt)
        self.size: tuple[float, float] | None = None

    @property
    def active(self) -> bool:
        return self.kalman.active

    def reset(self) -> None:
        self.kalman.reset()
        self.measurer.reset()
        self.size = None

    def lead(self, seconds: float) -> Point | None:
        return self.kalman.lead(seconds)

    @property
    def velocity(self) -> Point:
        if self.kalman.x is None:
            return (0.0, 0.0)
        return (float(self.kalman.x[2]), float(self.kalman.x[3]))

    def start(self, frame: np.ndarray, bbox: BBox) -> Track:
        self.reset()
        observed = self.measurer.start(frame, bbox)
        self.size = (observed.bbox[2], observed.bbox[3])
        self.kalman.start(observed.center)
        return Track(
            observed.center, (0.0, 0.0), observed.bbox, observed.bbox,
            observed.score, True,
        )

    def update(self, frame: np.ndarray, dt: float) -> Track | None:
        if not self.active or self.size is None:
            raise RuntimeError("no active selection; draw a box")
        observed = self.measurer.track(frame)
        accepted = observed.center if observed.score >= self.min_score else None
        if accepted is None:
            revert = getattr(self.measurer, "revert", None)
            if revert is not None:
                revert()
        step = min(self.max_dt, max(1e-3, float(dt)))
        reading = self.kalman.update(accepted, step)
        if reading is None:
            self.measurer.reset()
            self.size = None
            return None
        center, velocity, measured = reading
        if measured:
            self.size = (observed.bbox[2], observed.bbox[3])
            bbox = observed.bbox
        else:
            bbox = _bbox_around(center, self.size)
        return Track(center, velocity, bbox, observed.bbox, observed.score, measured)
