"""Offline regression checks; never connect to a projector or open a camera."""

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import cv2
import numpy as np
import pytest

from irisloop.image_pack import BMP_HEAD, build_stream
from tools import aim_five, chick_sunglasses

DATA = Path(__file__).resolve().parents[1] / "captures" / "procalib"


def test_transport_packing_matches_experiment_orientation():
    mask = np.zeros((480, 640), np.uint8)
    mask[12:18, 40:51] = 1
    expected = BMP_HEAD + np.packbits(mask[::-1, ::-1].ravel()).tobytes()
    assert build_stream(mask) == expected
    assert len(expected) == 38462


def test_bridge_clears_slot_only_after_first_successful_upload():
    async def exercise():
        bridge = aim_five.Bridge(SimpleNamespace(stdin=object(), stdout=object()))
        bridge.rpc = AsyncMock(side_effect=[RuntimeError("upload failed"), {}, {}])
        mask = np.zeros((480, 640), np.uint8)
        with pytest.raises(RuntimeError, match="upload failed"):
            await bridge.push_current(mask)
        assert not bridge.slot_prepared
        await bridge.push_current(mask)
        await bridge.push_current(mask)
        calls = bridge.rpc.call_args_list
        assert [call.args[1]["clear_slot"] for call in calls] == [True, True, False]
        assert base64.b64decode(calls[1].args[1]["data_base64"]) == build_stream(mask)
    asyncio.run(exercise())


def test_rpc_handles_utf8_notifications_and_clean_shutdown():
    async def exercise():
        writer = SimpleNamespace(write=lambda data: None, drain=AsyncMock(), close=lambda: None)
        reader = asyncio.StreamReader()
        rows = [
            {"method": "event.disconnected"},
            {"id": 1, "result": {"message": "\u5df2\u8fde\u63a5"}},
            {"id": 2, "result": {"ok": True}},
        ]
        reader.feed_data("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))
        process = SimpleNamespace(stdin=writer, stdout=reader, wait=AsyncMock(return_value=0))
        bridge = aim_five.Bridge(process)
        bridge.connected = True
        result = await bridge.rpc("ping")
        assert result["message"] == "\u5df2\u8fde\u63a5"
        assert not bridge.connected
        await bridge.close()
        process.wait.assert_awaited_once()
    asyncio.run(exercise())


def test_large_circles_are_selected_by_area_not_grid_label():
    dots = [{"cx": 30 + 30 * c, "cy": 30 + 30 * r, "area": 100}
            for r in range(9) for c in range(11)]
    large_indices = (15, 26, 40, 53, 76)
    for index in large_indices:
        dots[index]["area"] = 400
    hull = np.array([(0, 0), (400, 0), (400, 400), (0, 400)], np.float32)
    targets, radii = aim_five.large_circles(dots, hull)
    expected = sorted([(dots[i]["cx"], dots[i]["cy"]) for i in large_indices], key=lambda p: (p[1], p[0]))
    np.testing.assert_equal(targets, expected)
    np.testing.assert_allclose(radii, np.sqrt(400 / np.pi))
    dots[large_indices[-1]]["area"] = 100
    with pytest.raises(RuntimeError, match="exactly five"):
        aim_five.large_circles(dots, hull)


def five_spot_scene(dx=0, count=5):
    bg = np.zeros((600, 800, 3), np.uint8)
    frame = bg.copy()
    targets = np.array([(200, 150), (550, 180), (220, 350), (450, 350), (550, 450)], float)
    for x, y in targets[:count]:
        cv2.circle(frame, (int(x + dx), int(y)), 4, (80, 80, 80), -1)
    hull = np.array([(50, 50), (750, 50), (750, 550), (50, 550)], np.float32)
    return frame, bg, hull, targets, [24] * 5


def test_five_point_acceptance_requires_all_points_and_exact_error_limit():
    result = aim_five.evaluate(*five_spot_scene(dx=5))
    assert len(result) == 5
    assert all(row["error_px"] == pytest.approx(5) for row in result)
    assert all(row["light_inside_circle_fraction"] == 1 for row in result)
    with pytest.raises(RuntimeError, match="failed"):
        aim_five.evaluate(*five_spot_scene(dx=6))
    with pytest.raises(RuntimeError, match="five actual spots"):
        aim_five.evaluate(*five_spot_scene(count=4))


def test_sunglasses_acceptance_rejects_shift_and_board_spill():
    desired = chick_sunglasses.sunglasses()
    bg = np.full((1024, 1280, 3), 40, np.uint8)
    frame = bg + desired[:, :, None] * 100
    result, _ = chick_sunglasses.evaluate(frame, bg, desired)
    assert result["passed"]
    assert result["iou"] > 0.99
    shifted = bg + np.roll(desired, 20, axis=0)[:, :, None] * 100
    result, _ = chick_sunglasses.evaluate(shifted, bg, desired)
    assert not result["passed"]
    frame[400:430, 850:880] = 180
    result, _ = chick_sunglasses.evaluate(frame, bg, desired)
    assert result["board_spill_fraction"] > 0.03
    assert not result["passed"]
    with pytest.raises(RuntimeError, match="no substantial"):
        chick_sunglasses.evaluate(bg, bg, desired)


def test_archived_sunglasses_mask_is_reproducible_from_measured_mapping():
    run = DATA / "chick_sunglasses_20261027_103233"
    with np.load(run / "face_mapping.npz") as data:
        actual = chick_sunglasses.render(data["coefficients"], data["projector"], chick_sunglasses.sunglasses())
        report = json.loads((run / "report.json").read_text(encoding="utf-8"))
        residual = np.linalg.norm(
            chick_sunglasses.basis(data["projector"]) @ data["coefficients"] - data["camera"], axis=1)
    expected = cv2.imread(str(run / "projector_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert expected is not None
    np.testing.assert_array_equal(actual * 255, expected)
    np.testing.assert_allclose(residual, report["surface_fit"]["residual_px"], atol=1e-8)


def test_archived_five_circle_mask_uses_rounded_projector_coordinates():
    run = DATA / "five_circles_20261027_101523"
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    actual = np.zeros((480, 640), np.uint8)
    for point in report["projector_points"]:
        actual |= aim_five.make_dot_image(*point, radius=6)
    expected = cv2.imread(str(run / "projector_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert expected is not None
    np.testing.assert_array_equal(actual * 255, expected)
