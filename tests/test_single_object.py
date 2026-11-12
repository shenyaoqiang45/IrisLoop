"""Kalman filter and score gate. No camera, no UETrack weights."""

import numpy as np
import pytest

from irisloop.single_object import (
    BoxMeasurement,
    ConstantVelocityKalman,
    SingleObjectTracker,
    aim_camera_point,
)


class _Scripted:
    def __init__(self, boxes, scores):
        self._boxes = iter(boxes)
        self._scores = iter(scores)
        self.reset_count = 0
        self.reverts = 0

    def start(self, frame, bbox):
        return BoxMeasurement(tuple(bbox), 1.0)

    def track(self, frame):
        return BoxMeasurement(tuple(next(self._boxes)), float(next(self._scores)))

    def reset(self):
        self.reset_count += 1

    def revert(self):
        self.reverts += 1


def test_kalman_learns_constant_velocity_and_leads():
    filt = ConstantVelocityKalman(meas_std=1.0, accel_std=50.0)
    filt.start((0.0, 10.0))
    for step in range(1, 9):
        reading = filt.update((step * 20.0, 10.0), 0.1)
        assert reading is not None
    center, velocity, measured = reading
    assert measured
    assert center == pytest.approx((160.0, 10.0), abs=2.0)
    assert velocity == pytest.approx((200.0, 0.0), abs=15.0)
    assert filt.lead(1.0)[0] == pytest.approx(center[0] + velocity[0], abs=1e-6)


def test_kalman_coasts_then_forgets_the_track():
    filt = ConstantVelocityKalman(meas_std=1.0, accel_std=20.0, max_lost=2)
    filt.start((0.0, 0.0))
    filt.update((10.0, 0.0), 0.1)
    filt.update((20.0, 0.0), 0.1)
    first = filt.update(None, 0.1)
    assert first is not None and not first[2]
    assert first[0][0] > 20.0
    assert filt.update(None, 0.1) is not None
    assert filt.update(None, 0.1) is None
    assert not filt.active
    assert filt.lead(1.0) is None


def test_low_score_is_a_miss_and_does_not_jump_the_filter():
    measurer = _Scripted(
        boxes=[(110, 90, 20, 20), (400, 90, 20, 20), (130, 90, 20, 20)],
        scores=[0.9, 0.01, 0.9],
    )
    tracker = SingleObjectTracker(
        measurer,
        ConstantVelocityKalman(meas_std=1.0, accel_std=40.0, max_lost=4),
        min_score=0.1,
    )
    tracker.start(np.zeros((200, 200, 3), np.uint8), (90, 90, 20, 20))
    first = tracker.update(None, 0.1)
    assert first.measured
    assert first.center[0] == pytest.approx(120.0, abs=3.0)
    rejected = tracker.update(None, 0.1)
    assert rejected is not None and not rejected.measured
    assert measurer.reverts == 1
    assert rejected.raw_bbox[0] == pytest.approx(400.0)
    assert rejected.center[0] < 200.0
    recovered = tracker.update(None, 0.1)
    assert recovered.measured
    assert recovered.center[0] == pytest.approx(140.0, abs=15.0)


def test_repeated_misses_clear_the_selection():
    measurer = _Scripted(
        boxes=[(10, 10, 20, 20)] * 4,
        scores=[0.0] * 4,
    )
    tracker = SingleObjectTracker(
        measurer,
        ConstantVelocityKalman(max_lost=2),
        min_score=0.1,
    )
    frame = np.zeros((40, 40, 3), np.uint8)
    tracker.start(frame, (0, 0, 20, 20))
    assert tracker.update(frame, 0.1) is not None
    assert tracker.update(frame, 0.1) is not None
    assert tracker.update(frame, 0.1) is None
    assert not tracker.active
    with pytest.raises(RuntimeError):
        tracker.update(frame, 0.1)


def test_aim_leads_along_velocity_and_caps_the_offset():
    aim = aim_camera_point((10.0, 20.0), (100.0, 0.0), lead_s=0.5, max_lead_px=120.0)
    assert aim == pytest.approx((60.0, 20.0))
    capped = aim_camera_point((0.0, 0.0), (300.0, 0.0), lead_s=1.0, max_lead_px=120.0)
    assert capped == pytest.approx((120.0, 0.0))


def test_slow_motion_does_not_swing_the_aim_around_a_long_upload():
    still = aim_camera_point((499.0, 464.0), (20.0, -12.0), lead_s=15.8)
    assert still == pytest.approx((499.0, 464.0))
    moving = aim_camera_point((0.0, 0.0), (100.0, 0.0), lead_s=15.0, max_lead_px=120.0)
    assert moving == pytest.approx((100.0, 0.0))


def test_aim_is_dark_on_a_miss_and_drops_the_lead_when_velocity_reverses():
    assert aim_camera_point((10.0, 10.0), (100.0, 0.0), lead_s=1.0, measured=False) is None
    reversed_aim = aim_camera_point(
        (10.0, 10.0), (-200.0, 0.0), lead_s=1.0,
        previous_velocity=(200.0, 0.0),
    )
    assert reversed_aim == pytest.approx((10.0, 10.0))
