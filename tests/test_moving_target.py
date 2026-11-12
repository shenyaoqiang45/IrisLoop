"""Synthetic, hardware-free tests for the moving-object dot scenario."""

import asyncio
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from irisloop.calibrate import Calibration
from tools import track_dot
from irisloop.moving_target import (
    MotionDetector,
    MovingObjectTracker,
    Target,
    camera_to_projector,
    dot_mask,
)


def _frame(rect=None, dot=None, size=(240, 320), gray=False):
    height, width = size
    shape = (height, width) if gray else (height, width, 3)
    frame = np.full(shape, 25, np.uint8)
    if rect is not None:
        x0, y0, x1, y1 = rect
        frame[y0:y1, x0:x1] = 170 if gray else (160, 170, 180)
    if dot is not None:
        value = 255 if gray else (0, 255, 0)
        cv2.circle(frame, dot, 14, value, -1)
    return frame


def _warmed(background, **kwargs):
    detector = MotionDetector(history=20, var_threshold=16, min_area=200, **kwargs)
    for _ in range(25):
        detector.candidates(background)
    return detector


def test_detector_keeps_object_and_rejects_our_green_dot():
    detector = _warmed(_frame())
    candidates = detector.candidates(_frame(rect=(40, 60, 110, 130), dot=(260, 50)))
    assert len(candidates) == 1
    assert candidates[0].center == pytest.approx((75, 95), abs=8)
    assert candidates[0].measured


def test_detector_rejects_bright_dot_in_grayscale():
    detector = _warmed(_frame(gray=True), laser_green_excess=0)
    gray = _frame(gray=True)
    cv2.circle(gray, (260, 50), 12, 255, -1)
    assert detector.candidates(gray) == []


def test_roi_limits_where_an_object_may_be_found():
    detector = _warmed(_frame(), roi=(0, 0, 150, 240))
    assert detector.candidates(_frame(rect=(200, 60, 270, 130))) == []


def test_hint_prefers_the_nearest_object_over_a_distractor():
    detector = _warmed(_frame(), max_jump=100)
    frame = _frame(rect=(40, 60, 110, 130))
    frame[40:100, 240:300] = (160, 170, 180)
    picked = detector.detect(frame, hint=(75, 95))
    assert picked is not None
    assert picked.center == pytest.approx((75, 95), abs=8)
    # A hint with no nearby candidate is a miss, not an excuse to jump to a distractor.
    assert detector.detect(frame, hint=(10, 10)) is None


class _ScriptedDetector:
    def __init__(self, points):
        self.points = iter(points)
        self.hints = []

    def detect(self, frame, hint=None):
        self.hints.append(hint)
        point = next(self.points, None)
        if point is None:
            return None
        x, y = point
        return Target((x, y), (x - 10, y - 10, x + 10, y + 10), 400.0)


def test_tracker_smooths_tracks_velocity_and_leads_the_latency():
    detector = _ScriptedDetector([(100, 100), (120, 100), (140, 100)])
    tracker = MovingObjectTracker(
        detector, max_lost=2, smoothing=1.0, frame_dt=0.1,
    )
    assert tracker.update(None).center == pytest.approx((100, 100))
    assert tracker.update(None).center == pytest.approx((120, 100))
    third = tracker.update(None)
    assert third.center == pytest.approx((140, 100))
    assert tracker.velocity == pytest.approx((200, 0))
    assert tracker.lead(1.0) == pytest.approx((340, 100))
    assert detector.hints[1] == pytest.approx((100, 100))


def test_tracker_coasts_then_declares_loss_and_blanks():
    tracker = MovingObjectTracker(
        _ScriptedDetector([(100, 100), (110, 100), None, None, None]),
        max_lost=2, smoothing=1.0, frame_dt=0.1,
    )
    assert tracker.update(None).measured
    assert tracker.update(None).measured
    first_miss = tracker.update(None)
    assert not first_miss.measured
    assert first_miss.center[0] > 110  # constant-velocity coast
    assert not tracker.update(None).measured
    assert tracker.update(None) is None
    assert not tracker.active
    assert tracker.lead(1.0) is None


def test_camera_to_projector_and_dot_mask():
    mapper = Calibration(np.eye(3)).cam_to_proj
    proj = camera_to_projector((320.0, 240.0), mapper)
    assert proj == pytest.approx((320, 240))
    mask = dot_mask(proj, radius=8)
    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    ys, xs = np.nonzero(mask)
    assert xs.mean() == pytest.approx(320, abs=1)
    assert ys.mean() == pytest.approx(240, abs=1)
    assert set(np.unique(mask)) == {0, 1}


def test_dot_outside_the_light_field_is_rejected():
    mapper = Calibration(np.eye(3)).cam_to_proj
    assert dot_mask(camera_to_projector((-20, 240), mapper)) is None
    assert dot_mask(camera_to_projector((320, 479), mapper)) is None
    assert dot_mask(camera_to_projector((np.nan, 240), mapper)) is None
    assert dot_mask(None) is None


def test_skip_project_loop_follows_a_moving_object(monkeypatch):
    """Whole tool loop with a fake camera and no BLE connection."""
    def frame_with_square(x0=None):
        frame = _frame()
        cv2.circle(frame, (250, 50), 14, (0, 255, 0), -1)
        if x0 is not None:
            frame[80:140, x0:x0 + 60] = (160, 170, 180)
        return frame

    frames = iter([frame_with_square() for _ in range(20)]
                  + [frame_with_square(60 + 8 * i) for i in range(6)])

    class FakeCamera:
        def info(self):
            return "synthetic camera"

        def read(self):
            frame = next(frames, None)
            return (True, frame) if frame is not None else (False, None)

        def release(self):
            pass

    camera = FakeCamera()
    monkeypatch.setattr(track_dot, "_open_camera", lambda args: (camera, camera.read))
    monkeypatch.setattr(
        track_dot, "_homography_mapper",
        lambda path: Calibration(np.eye(3)).cam_to_proj,
    )
    seen = []
    real_dot_mask = track_dot.dot_mask

    def spy(proj, radius=12):
        seen.append(proj)
        return real_dot_mask(proj, radius)

    monkeypatch.setattr(track_dot, "dot_mask", spy)
    args = SimpleNamespace(
        procalib=None, calibration="unused", min_area=200.0, roi=None,
        laser_green_excess=28.0, max_lost=8, smoothing=0.6, skip_project=True,
        address=None, warmup=20, no_preview=True, lead=0.0, radius=12,
        max_frames=6, seconds=0.0, log_every=0, trace=None, trace_every=30,
        min_fill=0.20, max_lead_px=120.0, preview_scale=0.75,
    )
    assert asyncio.run(track_dot.run(args)) == 0
    valid = [point for point in seen if point is not None]
    assert len(valid) >= 3
    assert valid[-1][0] > valid[0][0] + 10  # the dot followed the object
    assert all(abs(point[1] - 110) < 30 for point in valid)  # not the green dot
