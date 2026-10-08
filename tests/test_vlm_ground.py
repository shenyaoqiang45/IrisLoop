import cv2
import numpy as np
import pytest

from irisloop.vlm_ground import extract_json_value, refine_outline, regions_from_reply

REPLY = (
    '[{"bbox_2d": [100, 200, 300, 400], "label": "海报", '
    '"outline": [[100, 200], [300, 200], [300, 400]]}]'
)


def test_regions_convert_0_1000_to_pixels():
    regions = regions_from_reply(REPLY, 1280, 720)
    r = regions[0]
    assert r["label"] == "海报"
    assert r["bbox"] == (128.0, 144.0, 384.0, 288.0)
    assert r["polygon"] == [(128.0, 144.0), (384.0, 144.0), (384.0, 288.0)]


def test_reply_with_prose_and_fence():
    text = "找到目标了：\n```json\n" + REPLY + "\n```\n以上。"
    regions = regions_from_reply(text, 1280, 720)
    assert regions[0]["bbox"] == (128.0, 144.0, 384.0, 288.0)


def test_single_object_and_clamping():
    text = '{"bbox_2d": [-50, 0, 1200, 1000], "label": "x"}'
    regions = regions_from_reply(text, 1280, 720)
    assert regions[0]["bbox"] == (0.0, 0.0, 1279.0, 719.0)
    assert regions[0]["polygon"] is None


def test_degenerate_and_garbage_rejected():
    with pytest.raises(ValueError):
        regions_from_reply('[{"bbox_2d": [5, 5, 5, 9]}]', 1280, 720)
    with pytest.raises(ValueError):
        regions_from_reply("没有找到目标", 1280, 720)
    with pytest.raises(ValueError):
        extract_json_value("no json here")


def test_refine_outline_extracts_circle():
    frame = np.full((480, 640, 3), 220, np.uint8)
    cv2.circle(frame, (320, 240), 80, (30, 30, 30), -1)
    pts = refine_outline(frame, (230, 150, 410, 330))
    assert len(pts) >= 6  # a polygon, not a rectangle
    poly = np.asarray(pts, np.float32)
    area = cv2.contourArea(poly)
    assert abs(area - np.pi * 80 * 80) / (np.pi * 80 * 80) < 0.15
    cx, cy = poly.mean(axis=0)
    assert abs(cx - 320) < 10 and abs(cy - 240) < 10
