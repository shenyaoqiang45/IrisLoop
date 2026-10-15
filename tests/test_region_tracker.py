import cv2
import numpy as np
import pytest

from irisloop.region_tracker import RegionTracker, TrackingLost


def textured_object():
    rng = np.random.default_rng(42)
    frame = np.full((360, 480), 25, dtype=np.uint8)
    frame[90:270, 120:360] = rng.integers(30, 220, (180, 240), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    polygon = np.array([[130, 100], [350, 100], [350, 260], [130, 260]], np.float32)
    return frame, polygon


def test_translation_rotation_remains_bound_to_original_region():
    frame, polygon = textured_object()
    tracker = RegionTracker()
    tracker.start(frame, polygon)
    for step in range(1, 16):
        transform = cv2.getRotationMatrix2D((240, 180), step * 0.6, 1.0)
        transform[:, 2] += (step * 2, step)
        moved = cv2.warpAffine(frame, transform, (480, 360), borderValue=25)
        actual = tracker.update(cv2.cvtColor(moved, cv2.COLOR_GRAY2BGR))
        expected = cv2.transform(polygon.reshape(-1, 1, 2), transform).reshape(-1, 2)
        assert np.max(np.linalg.norm(actual - expected, axis=1)) < 1.0
    assert tracker.active


def test_background_features_do_not_initialize_textureless_selection():
    frame, polygon = textured_object()
    frame[90:270, 120:360] = 100
    rng = np.random.default_rng(1)
    frame[:80] = rng.integers(0, 255, frame[:80].shape, dtype=np.uint8)
    tracker = RegionTracker()
    with pytest.raises(TrackingLost, match="too little texture"):
        tracker.start(frame, polygon)
    assert not tracker.active


def test_occlusion_latches_loss_until_reselection():
    frame, polygon = textured_object()
    tracker = RegionTracker()
    tracker.start(frame, polygon)
    with pytest.raises(TrackingLost):
        tracker.update(np.zeros_like(frame))
    assert not tracker.active
    with pytest.raises(TrackingLost, match="no active selection"):
        tracker.update(frame)
    tracker.start(frame, polygon)
    np.testing.assert_allclose(tracker.update(frame), polygon, atol=0.1)


def test_resolution_change_invalidates_selection():
    frame, polygon = textured_object()
    tracker = RegionTracker()
    tracker.start(frame, polygon)
    with pytest.raises(TrackingLost, match="dimensions changed"):
        tracker.update(frame[:200])
    assert not tracker.active


def test_large_scale_change_rejected():
    frame, polygon = textured_object()
    tracker = RegionTracker()
    tracker.start(frame, polygon)
    with pytest.raises(TrackingLost, match="depth or tilt"):
        for scale in (1.03, 1.06, 1.09, 1.12):
            transform = cv2.getRotationMatrix2D((240, 180), 0, scale)
            tracker.update(cv2.warpAffine(frame, transform, (480, 360)))
    assert not tracker.active


def test_region_leaving_view_is_rejected():
    frame, polygon = textured_object()
    tracker = RegionTracker()
    tracker.start(frame, polygon)
    with pytest.raises(TrackingLost):
        for shift in range(5, 181, 5):
            transform = np.array([[1, 0, shift], [0, 1, 0]], np.float32)
            tracker.update(cv2.warpAffine(frame, transform, (480, 360)))
    assert not tracker.active


@pytest.mark.parametrize("polygon", [
    np.zeros((2, 2)), np.full((4, 2), np.nan), np.zeros((4, 3)),
])
def test_invalid_selection_is_explicit(polygon):
    tracker = RegionTracker()
    with pytest.raises(TrackingLost, match="invalid selection"):
        tracker.start(np.zeros((100, 100), np.uint8), polygon)


def test_returned_polygon_cannot_mutate_tracker():
    frame, polygon = textured_object()
    tracker = RegionTracker()
    result = tracker.start(frame, polygon)
    result[:] = 0
    polygon[:] = 0
    assert np.min(tracker.update(frame)) > 90
