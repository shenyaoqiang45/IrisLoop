import cv2
import numpy as np

from irisloop.calibrate import Calibration
from irisloop.physical_reward import assess_physical


def _capture(mask):
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[..., 1] = mask.astype(np.uint8) * 255
    return image


def _score(target, observed):
    background = np.zeros((*target.shape, 3), dtype=np.uint8)
    return assess_physical(
        target.astype(np.uint8) * 255,
        _capture(observed),
        background,
        Calibration(np.eye(3)),
        tolerance_px=(0, 4),
        selected_tolerance_px=4,
    )[0]


def test_identity_projection_scores_one():
    target = np.zeros((480, 640), dtype=bool)
    target[180:260, 220:420] = True
    result = _score(target, target)
    assert result.strict.iou > 0.99
    assert result.reward > 0.99
    assert result.centroid_error_px < 0.1


def test_iou_decreases_with_translation():
    target = np.zeros((480, 640), dtype=bool)
    target[220:240, 200:440] = True
    scores = []
    for shift in (0, 5, 15):
        observed = np.roll(target, shift, axis=1)
        scores.append(_score(target, observed).strict.iou)
    assert scores[0] > scores[1] > scores[2]


def test_tolerance_accepts_nearby_halo_but_reports_spill():
    target = np.zeros((480, 640), dtype=np.uint8)
    target[235:245, 200:440] = 1
    observed = cv2.dilate(target, np.ones((7, 7), np.uint8)) > 0
    result = _score(target > 0, observed)
    tolerant = next(item for item in result.tolerant if item.radius_px == 4)
    assert result.strict.iou < 0.7
    assert tolerant.f1 > 0.99
    assert result.strict.spill_pixels > 0


def test_tolerance_recovers_thin_line_while_strict_iou_prefers_thick():
    thin = np.zeros((480, 640), dtype=np.uint8)
    cv2.line(thin, (160, 240), (480, 240), 1, 4)
    thick = np.zeros((480, 640), dtype=np.uint8)
    cv2.line(thick, (160, 240), (480, 240), 1, 16)
    bloom = cv2.dilate(thin, np.ones((9, 9), np.uint8)) > 0
    thin_result = _score(thin > 0, bloom)
    thick_result = _score(thick > 0, bloom)
    thin_tol = next(item for item in thin_result.tolerant if item.radius_px == 4)
    assert thick_result.strict.iou > thin_result.strict.iou
    assert thin_result.strict.iou < 0.5
    assert thin_tol.f1 > 0.95
    assert 0.0 <= thin_result.reward <= 1.0
