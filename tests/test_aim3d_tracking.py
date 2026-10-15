import asyncio
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools import aim3d


def test_projection_coalesces_pending_masks_and_serializes_black(monkeypatch):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        sent = []

        async def project(cli, mask, settle):
            sent.append(mask.copy())
            if len(sent) == 1:
                entered.set()
                await release.wait()

        monkeypatch.setattr(aim3d, "_project", project)
        output = aim3d.LatestProjection(aim3d.IrisBleClient("unused"))
        first = aim3d._black_image()
        first[10:20, 10:20] = 1
        middle = np.roll(first, 20, axis=1)
        latest = np.roll(first, 40, axis=1)
        output.submit(first)
        output.pump()
        await entered.wait()
        output.submit(middle)
        output.submit(latest)
        output.pump()
        assert len(sent) == 1
        release.set()
        await asyncio.sleep(0)
        output.pump()
        await asyncio.sleep(0)
        np.testing.assert_array_equal(sent[1], latest)
        output.submit(latest)
        output.pump()
        assert len(sent) == 2
        output.submit(middle)
        output.submit(aim3d._black_image())
        output.pump()
        await output.finish()
        assert len(sent) == 3
        assert not sent[-1].any()

    asyncio.run(scenario())


def test_projection_failure_is_not_silenced(monkeypatch):
    async def project(cli, mask, settle):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(aim3d, "_project", project)

    async def scenario():
        output = aim3d.LatestProjection(aim3d.IrisBleClient("unused"))
        output.submit(np.ones((480, 640), np.uint8))
        output.pump()
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="upload failed"):
            output.pump()
        with pytest.raises(RuntimeError, match="upload failed"):
            await output.finish()

    asyncio.run(scenario())


def test_new_drawing_drops_old_pending_selection_and_clears_trail():
    ui = aim3d.ClickUI()
    ui.pending_poly = np.ones((4, 2))
    ui.on_mouse(cv2.EVENT_LBUTTONDOWN, 30, 30, 0, None)
    assert ui.pending_poly is None
    ui.on_mouse(cv2.EVENT_MOUSEMOVE, 100, 30, 0, None)
    ui.on_mouse(cv2.EVENT_MOUSEMOVE, 100, 100, 0, None)
    ui.on_mouse(cv2.EVENT_LBUTTONUP, 30, 100, 0, None)
    assert ui.pending_poly is not None
    assert not ui.points
    assert not ui.drawing


@pytest.mark.parametrize("static", [False, True])
def test_interactive_loop_updates_light_and_blanks_on_loss(monkeypatch, static):
    rng = np.random.default_rng(6)
    gray = np.full((300, 400), 25, np.uint8)
    gray[70:220, 90:300] = rng.integers(30, 220, (150, 210), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    first = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    transform = np.array([[1, 0, 12], [0, 1, 7]], np.float32)
    moved = cv2.warpAffine(first, transform, (400, 300), borderValue=(25, 25, 25))
    frames = iter([first, moved, moved, np.zeros_like(first), first, first])
    sent = []
    callbacks = {}
    ticks = 0

    class Camera:
        def __init__(self, directory):
            pass

        def enumerate(self):
            pass

        def open(self, serial):
            pass

        def set_gain(self, gain):
            pass

        def set_exposure_us(self, exposure):
            pass

        def flush(self, count):
            pass

        def info(self):
            return "synthetic camera"

        def read(self, timeout):
            return True, next(frames)

        def close(self):
            pass

    class Opened:
        def __init__(self):
            self.client = object()
            self.address = "unused"

        async def close(self):
            pass

    async def project(cli, mask, settle):
        sent.append(mask.copy())

    def wait_key(delay):
        nonlocal ticks
        ticks += 1
        if ticks == 1:
            mouse = callbacks["mouse"]
            for event, x, y in [
                (cv2.EVENT_LBUTTONDOWN, 100, 80),
                (cv2.EVENT_MOUSEMOVE, 280, 80),
                (cv2.EVENT_MOUSEMOVE, 280, 200),
                (cv2.EVENT_LBUTTONUP, 100, 200),
            ]:
                mouse(event, x * aim3d.SCALE, y * aim3d.SCALE, 0, None)
        return ord("q") if ticks == 6 else -1

    monkeypatch.setattr(aim3d, "HikCamera", Camera)
    async def open_projector(address):
        return Opened()

    monkeypatch.setattr(aim3d, "open_projector", open_projector)
    monkeypatch.setattr(aim3d, "_project", project)
    monkeypatch.setattr(aim3d, "load_calib", lambda path: (
        np.eye(3), np.zeros(5), {"grid_pts": np.zeros((10, 2))}, 1000.0,
    ))
    monkeypatch.setattr(aim3d, "plane_homography_from_rays", lambda *args: np.eye(3))
    monkeypatch.setattr(cv2, "namedWindow", lambda name: None)
    monkeypatch.setattr(cv2, "setMouseCallback",
                        lambda name, callback: callbacks.update(mouse=callback))
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    args = SimpleNamespace(calib="unused", address="unused", mvs_dir=None,
                           serial="unused", gain=8, exposure=5000,
                           at=None, static=static)
    assert asyncio.run(aim3d.run(args)) == 0
    lit = [mask for mask in sent if mask.any()]
    assert len(lit) >= (1 if static else 2)
    first_y, first_x = np.nonzero(lit[0])
    last_y, last_x = np.nonzero(lit[-1])
    expected_offset = (0, 0) if static else (12, 7)
    np.testing.assert_allclose(
        (last_x.mean() - first_x.mean(), last_y.mean() - first_y.mean()),
        expected_offset, atol=1.0,
    )
    assert sent[-1].shape == (480, 640)
    assert not sent[-1].any()
    if not static:
        last_lit = max(i for i, mask in enumerate(sent) if mask.any())
        # Blank on loss, stay blank when the object returns, then blank on exit.
        assert len(sent) - last_lit - 1 >= 2
