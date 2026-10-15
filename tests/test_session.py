import asyncio
import json

import numpy as np
import pytest

from irisloop import session
from irisloop.image_pack import HEIGHT, WIDTH
from irisloop.loop import push_image as real_push_image
from irisloop.protocol import Response


def test_play_units_quantizes_to_100ms_and_reports_the_device_clock():
    total, interval, seconds, frame_ms = session.play_units(60, 300)
    assert (total, interval, seconds, frame_ms) == (600, 3, 60.0, 300)
    _, _, seconds, frame_ms = session.play_units(0.04, 40)
    assert seconds == 0.1
    assert frame_ms == 100


def test_play_units_rejects_more_than_one_hour():
    with pytest.raises(ValueError, match="1 hour"):
        session.play_units(3601, 300)


def test_prepare_frame_keeps_a_binary_mask_and_resizes_gray():
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    mask[0, 0] = 1
    assert session.prepare_frame(mask)[0, 0] == 1
    gray = np.zeros((20, 20), np.uint8)
    gray[:10, :] = 255
    prepared = session.prepare_frame(gray)
    assert prepared.shape == (HEIGHT, WIDTH)
    assert prepared.max() == 1


def test_animation_over_the_slot_limit_does_not_clear_the_group():
    calls = []

    class Client:
        address = "20:6E:F1:81:71:62"

        async def send_command(self, frame):
            calls.append(frame)
            return Response(ok=True)

    projector = session.ProjectorSession(Client(), name="Iris-G-817162")
    frames = [np.zeros((HEIGHT, WIDTH), np.uint8) for _ in range(11)]
    with pytest.raises(session.FrameLimitError):
        asyncio.run(projector.show_animation(frames, frame_ms=300, seconds=10))
    assert calls == []


def test_show_animation_reports_the_boot_slot_and_quantized_play(monkeypatch):
    sent = []

    class Client:
        address = "20:6E:F1:81:71:62"

        async def send_command(self, frame):
            sent.append(("cmd", bytes(frame)))
            return Response(ok=True)

        async def stop(self):
            sent.append(("stop",))
            return Response(ok=True)

        async def play(self, group_id, loop, total_100ms, interval_100ms):
            sent.append(("play", group_id, loop, total_100ms, interval_100ms))
            return Response(ok=True)

    async def push(cli, stream, name):
        sent.append(("push", name, len(stream)))

    monkeypatch.setattr(session, "push_image", push)
    projector = session.ProjectorSession(Client(), name="Iris-G-817162")
    frames = [np.full((HEIGHT, WIDTH), index % 2, np.uint8) for index in range(2)]
    result = asyncio.run(projector.show_animation(frames, frame_ms=250, seconds=12.04))

    assert result.address == "20:6E:F1:81:71:62"
    assert result.name == "Iris-G-817162"
    assert result.files == ("1_1.bmp", "1_2.bmp")
    assert result.replaced == "group 1 (boot)"
    assert result.play_seconds == 12.0
    assert result.frame_ms == 300
    assert [item[0] for item in sent].count("push") == 2
    assert sent[-1] == ("play", 1, True, 120, 3)
    assert real_push_image is not push


def test_find_keeps_only_iris_advertisements_and_sorts_by_rssi(monkeypatch):
    async def fake_scan(seconds):
        assert seconds == 2
        return [
            ("AA:AA:AA:AA:AA:AA", "headphones", -40, []),
            ("20:6E:F1:81:71:62", "Iris-G-817162", -71, []),
            ("F4:12:FA:B6:B7:CA", "Iris-G-B6B7CA", -58, []),
            ("BB:BB:BB:BB:BB:BB", "", -30, ["adb401c0-b1c6-11ed-afa1-0242ac120002"]),
        ]

    monkeypatch.setattr(session, "scan_advertisements", fake_scan)
    found = asyncio.run(session.find(2))
    assert [item.address for item in found] == [
        "BB:BB:BB:BB:BB:BB",
        "F4:12:FA:B6:B7:CA",
        "20:6E:F1:81:71:62",
    ]
    assert found[1].name == "Iris-G-B6B7CA"


def test_find_command_prints_json_and_does_not_pick_a_device(capsys, monkeypatch):
    async def fake_find(seconds=8.0):
        assert seconds == 1
        return [
            session.Sighting("F4:12:FA:B6:B7:CA", "Iris-G-B6B7CA", -58),
            session.Sighting("20:6E:F1:81:71:62", "Iris-G-817162", -71),
        ]

    monkeypatch.setattr(session, "find", fake_find)
    assert session.main(["find", "--seconds", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["address"] for item in payload["devices"]] == [
        "F4:12:FA:B6:B7:CA",
        "20:6E:F1:81:71:62",
    ]


def test_connect_refuses_a_missing_address():
    with pytest.raises(ValueError, match="find"):
        asyncio.run(session.connect("  "))


def test_blank_waits_for_the_show_already_holding_the_lock(monkeypatch):
    order = []
    entered = asyncio.Event()
    release = asyncio.Event()

    class Client:
        address = "20:6E:F1:81:71:62"

        async def send_command(self, frame):
            return Response(ok=True)

        async def stop(self):
            return Response(ok=True)

        async def play(self, **kwargs):
            return Response(ok=True)

    async def push(cli, stream, name):
        order.append(name)
        if name == "1_1.bmp" and order.count("1_1.bmp") == 1:
            entered.set()
            await release.wait()

    monkeypatch.setattr(session, "push_image", push)

    async def scenario():
        projector = session.ProjectorSession(Client(), name="Iris-G-817162")
        show = asyncio.create_task(
            projector.show_still(np.ones((HEIGHT, WIDTH), np.uint8), seconds=5)
        )
        await entered.wait()
        blank = asyncio.create_task(projector.blank(seconds=1))
        await asyncio.sleep(0)
        assert blank.done() is False
        release.set()
        await show
        await blank
        assert order == ["1_1.bmp", "1_1.bmp"]

    asyncio.run(scenario())


def test_resolve_address_uses_a_given_address_or_the_only_projector(monkeypatch, capsys):
    async def fake_find(seconds=8.0):
        assert seconds == 1
        return [session.Sighting("20:6E:F1:81:71:62", "Iris-G-817162", -70)]

    monkeypatch.setattr(session, "find", fake_find)
    assert asyncio.run(session.resolve_address(" aa:bb ", 1)) == "aa:bb"
    assert asyncio.run(session.resolve_address(None, 1)) == "20:6E:F1:81:71:62"
    assert "Iris-G-817162" in capsys.readouterr().out


def test_resolve_address_refuses_zero_or_many(monkeypatch):
    async def none(seconds=8.0):
        return []

    monkeypatch.setattr(session, "find", none)
    with pytest.raises(RuntimeError, match="no IrisGreen"):
        asyncio.run(session.resolve_address("  "))

    async def many(seconds=8.0):
        return [
            session.Sighting("F4:12:FA:B6:B7:CA", "Iris-G-B6B7CA", -40),
            session.Sighting("20:6E:F1:81:71:62", "Iris-G-817162", -70),
        ]

    monkeypatch.setattr(session, "find", many)
    with pytest.raises(RuntimeError, match="pass --address"):
        asyncio.run(session.resolve_address(None))


def test_play_and_push_current_do_not_delete_the_group(monkeypatch):
    calls = []

    class Client:
        address = "20:6E:F1:81:71:62"

        async def send_command(self, frame):
            calls.append("delete")
            return Response(ok=True)

        async def stop(self):
            calls.append("stop")
            return Response(ok=True)

        async def play(self, group_id, loop, total_100ms, interval_100ms):
            calls.append(("play", group_id, loop, total_100ms, interval_100ms))
            return Response(ok=True)

    async def push(cli, stream, name):
        calls.append(("push", name))

    monkeypatch.setattr(session, "push_image", push)
    projector = session.ProjectorSession(Client(), name="Iris-G-817162")
    asyncio.run(projector.play(11, seconds=6, frame_ms=200, loop=False))
    asyncio.run(projector.push_current(np.zeros((HEIGHT, WIDTH), np.uint8), seconds=10))
    assert calls[0] == ("play", 11, False, 60, 2)
    assert "delete" not in calls
    assert ("push", "1_1.bmp") in calls
