import json

import numpy as np
import pytest

from irisloop.calibrate import Calibration
from irisloop.target_path import TargetPath, render_target


def _calibration(tmp_path):
    path = tmp_path / "homography.npz"
    Calibration(np.eye(3)).save(str(path))
    return path


def test_target_path_roundtrip_and_multiple_widths(tmp_path):
    calibration_path = _calibration(tmp_path)
    target = TargetPath.create(
        [[[100, 100], [200, 120], [300, 180]]],
        scene="test",
        mode="stroke",
        camera_size=(640, 480),
        calibration_path=str(calibration_path),
    )
    path = tmp_path / "path.json"
    target.save(str(path))
    loaded = TargetPath.load(str(path))
    loaded.validate_calibration(str(calibration_path))

    narrow = render_target(loaded, Calibration(np.eye(3)), 4)
    wide = render_target(loaded, Calibration(np.eye(3)), 16)
    assert wide.mask_pixels > narrow.mask_pixels
    assert narrow.mask.shape == (480, 640)


def test_target_rejects_different_calibration(tmp_path):
    first = _calibration(tmp_path)
    target = TargetPath.create(
        [[[100, 100], [200, 120]]],
        scene="test",
        mode="stroke",
        camera_size=(640, 480),
        calibration_path=str(first),
    )
    second = tmp_path / "other.npz"
    Calibration(np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], float)).save(
        str(second)
    )
    with pytest.raises(ValueError, match="different calibration"):
        target.validate_calibration(str(second))
