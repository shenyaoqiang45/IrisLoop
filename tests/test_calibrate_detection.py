import cv2
import numpy as np

from irisloop.calibrate import detect_dot


def test_detect_dot_prefers_compact_laser_over_reflection_strips():
    background = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame = background.copy()
    cv2.circle(frame, (540, 220), 10, (0, 255, 0), -1)
    cv2.rectangle(frame, (450, 400), (750, 410), (0, 255, 0), -1)
    cv2.rectangle(frame, (0, 700), (1279, 719), (0, 255, 0), -1)
    point, area = detect_dot(frame, background)
    assert point is not None
    assert abs(point[0] - 540) < 1
    assert abs(point[1] - 220) < 1
    assert 250 < area < 400


def test_detect_dot_rejects_only_elongated_reflection():
    background = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame = background.copy()
    cv2.rectangle(frame, (450, 400), (750, 410), (0, 255, 0), -1)
    point, area = detect_dot(frame, background)
    assert point is None
    assert area > 1000
