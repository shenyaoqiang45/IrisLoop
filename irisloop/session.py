"""Agent-facing IrisGreen session.

Callers choose a device, a picture, and how long it stays up. Scanning,
file names, the boot-group slot, and the protocol's 100 ms clocks stay here.

    from irisloop.session import find, connect

    sights = await find()
    session = await connect(sights[0].address)
    try:
        result = await session.show_animation(frames, frame_ms=300, seconds=60)
    finally:
        await session.close()  # close does not stop playback; call stop() for that
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from irisloop import protocol as P
from irisloop.ble_client import IrisBleClient
from irisloop.image_pack import HEIGHT, WIDTH, binarize, build_stream
from irisloop.loop import push_image
from irisloop.projector import CHAR_DEVICE_NAME, is_iris_projector

# The loop already uses group 1 as the writable slot. It is the boot animation
# and the device only accepts 10 frames there.
SLOT_GROUP = 1
SLOT_MAX_FRAMES = 10
PLAY_UNIT_MS = 100
PLAY_CAP_UNITS = 36000


def _half_up(value: float, unit: int) -> int:
    """Round value/unit to the nearest unit, with halves going up."""
    return max(1, int((value + unit / 2) // unit))


class FrameLimitError(ValueError):
    """The slot cannot hold this many frames. Nothing was deleted."""


@dataclass(frozen=True)
class Sighting:
    address: str
    name: str
    rssi: int | None


@dataclass(frozen=True)
class Projection:
    address: str
    name: str
    group_id: int
    group_label: str
    files: tuple[str, ...]
    frames: int
    play_seconds: float
    frame_ms: int

    @property
    def replaced(self) -> str:
        return f"group {self.group_id} ({self.group_label})"


def play_units(seconds: float, frame_ms: int) -> tuple[int, int, float, int]:
    """Convert caller seconds and frame time into device clocks.

    Returns protocol units plus the quantized seconds and frame milliseconds
    the device will actually use. Units are 100 ms. Sub-unit times round up
    to one unit; anything over one hour is rejected.
    """
    if seconds <= 0:
        raise ValueError("play time must be positive")
    if frame_ms <= 0:
        raise ValueError("frame time must be positive")
    total = _half_up(seconds * 1000, PLAY_UNIT_MS)
    if total > PLAY_CAP_UNITS:
        raise ValueError("play time cap is 1 hour")
    interval = _half_up(frame_ms, PLAY_UNIT_MS)
    if interval > 0xFFFF:
        raise ValueError("frame time is too long")
    return total, interval, total * PLAY_UNIT_MS / 1000, interval * PLAY_UNIT_MS


def prepare_frame(image: np.ndarray | str) -> np.ndarray:
    """Load a path or array and return a 640×480 1bpp mask."""
    if isinstance(image, str):
        buf = np.fromfile(image, dtype=np.uint8)
        gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"cannot read {image}")
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.ndim == 2:
            gray = arr
        else:
            raise ValueError("image must be a path or a 2-D or 3-D array")
    if gray.shape[:2] != (HEIGHT, WIDTH):
        gray = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    unique = np.unique(gray)
    if unique.size <= 2 and int(unique.max()) <= 1:
        return gray.astype(np.uint8)
    return binarize(gray)


async def scan_advertisements(seconds: float) -> list[tuple[str, str, int | None, list[str]]]:
    """Return address, name, rssi, and service UUIDs for every advertisement."""
    from bleak import BleakScanner

    scanner = BleakScanner()
    await scanner.start()
    try:
        await asyncio.sleep(seconds)
    finally:
        await scanner.stop()
    rows: list[tuple[str, str, int | None, list[str]]] = []
    found = scanner.discovered_devices_and_advertisement_data
    for addr, (dev, adv) in found.items():
        name = dev.name or adv.local_name or ""
        uuids = [str(u) for u in (adv.service_uuids or [])]
        rows.append((addr, name, adv.rssi, uuids))
    return rows


async def find(seconds: float = 8.0) -> list[Sighting]:
    """Scan for IrisGreen projectors. An empty list means none were advertising."""
    if seconds <= 0:
        raise ValueError("scan time must be positive")
    seen: dict[str, Sighting] = {}
    for address, name, rssi, uuids in await scan_advertisements(seconds):
        if not is_iris_projector(name, uuids):
            continue
        key = address.upper()
        current = seen.get(key)
        if current is None or (rssi is not None and (current.rssi is None or rssi > current.rssi)):
            seen[key] = Sighting(address=address, name=name or "(unnamed)", rssi=rssi)
    return sorted(seen.values(), key=lambda item: item.rssi if item.rssi is not None else -999, reverse=True)


class ProjectorSession:
    """One connected projector. Methods serialize so a blank waits for an upload."""

    def __init__(self, client: IrisBleClient, name: str = ""):
        self.client = client
        self.address = client.address
        self.name = name
        self._lock = asyncio.Lock()

    async def show_still(self, image: np.ndarray | str, seconds: float = 60.0) -> Projection:
        return await self._project([prepare_frame(image)], frame_ms=1000, seconds=seconds)

    async def show_animation(
        self,
        frames: list[np.ndarray | str],
        frame_ms: int = 300,
        seconds: float = 60.0,
    ) -> Projection:
        if not frames:
            raise ValueError("animation needs at least one frame")
        prepared = [prepare_frame(frame) for frame in frames]
        return await self._project(prepared, frame_ms=frame_ms, seconds=seconds)

    async def stop(self) -> bool:
        """Ask the device to stop. Disconnecting does not do this."""
        async with self._lock:
            response = await self.client.stop()
        return bool(response.ok)

    async def blank(self, seconds: float = 5.0) -> Projection:
        """Project black. Waits until any in-flight show on this session finishes."""
        black = np.zeros((HEIGHT, WIDTH), np.uint8)
        return await self._project([black], frame_ms=1000, seconds=seconds)

    async def push_current(self, image: np.ndarray | str, seconds: float = 3600.0) -> None:
        """Overwrite ``1_1.bmp`` and play it. Does not delete the group.

        Live correction calls this every frame. Deleting first would blank the
        beam between uploads.
        """
        async with self._lock:
            await deliver_mask(self.client, image, seconds=seconds)

    async def play(
        self,
        group_id: int,
        seconds: float,
        frame_ms: int = 200,
        loop: bool = False,
    ) -> None:
        """Play a group that is already stored. Nothing is uploaded or deleted."""
        if not 1 <= group_id <= 0xFF:
            raise ValueError(f"group must be 1..255, got {group_id}")
        total, interval, _, _ = play_units(seconds, frame_ms)
        async with self._lock:
            played = await self.client.play(group_id, loop, total, interval)
        if not played.ok:
            raise RuntimeError(played.error or "play was rejected")

    async def clear_slot(self) -> None:
        """Delete group 1 once, so the next ``push_current`` is the only frame."""
        async with self._lock:
            deleted = await self.client.send_command(P.cmd_delete_group(SLOT_GROUP))
            if not deleted.ok:
                raise RuntimeError(deleted.error or "delete group was rejected")
            await asyncio.sleep(0.5)

    async def close(self) -> None:
        """Drop the BLE link. Playback already started keeps running."""
        try:
            await self.client.disconnect()
        except Exception as exc:
            print(f"[warn] BLE disconnect failed: {exc}", file=sys.stderr)

    async def _project(self, frames: list[np.ndarray], frame_ms: int, seconds: float) -> Projection:
        if len(frames) > SLOT_MAX_FRAMES:
            raise FrameLimitError(
                f"group {SLOT_GROUP} ({P.group_name(SLOT_GROUP)}) holds at most "
                f"{SLOT_MAX_FRAMES} frames, got {len(frames)}"
            )
        total, interval, play_seconds, actual_frame_ms = play_units(seconds, frame_ms)
        async with self._lock:
            deleted = await self.client.send_command(P.cmd_delete_group(SLOT_GROUP))
            if not deleted.ok:
                raise RuntimeError(deleted.error or f"failed to clear group {SLOT_GROUP}")
            await asyncio.sleep(0.5)
            files: list[str] = []
            for index, mask in enumerate(frames, 1):
                name = f"{SLOT_GROUP}_{index}.bmp"
                await push_image(self.client, build_stream(mask), name)
                files.append(name)
            await self.client.stop()
            await asyncio.sleep(0.3)
            played = await self.client.play(
                group_id=SLOT_GROUP,
                loop=True,
                total_100ms=total,
                interval_100ms=interval,
            )
            if not played.ok:
                raise RuntimeError(played.error or "play was rejected")
        return Projection(
            address=self.address,
            name=self.name,
            group_id=SLOT_GROUP,
            group_label=P.group_name(SLOT_GROUP),
            files=tuple(files),
            frames=len(frames),
            play_seconds=play_seconds,
            frame_ms=actual_frame_ms,
        )


def moving_bar_frames(count: int = 8) -> list[np.ndarray]:
    """High-contrast frames: a bar steps from left to right, with a frame number."""
    if count < 1:
        raise ValueError("frame count must be positive")
    frames: list[np.ndarray] = []
    for index in range(count):
        image = np.zeros((HEIGHT, WIDTH), np.uint8)
        x = 30 + index * max(1, (WIDTH - 150) // max(1, count - 1))
        cv2.rectangle(image, (8, 8), (WIDTH - 9, HEIGHT - 9), 255, 3)
        cv2.rectangle(image, (x, 150), (min(WIDTH - 20, x + 90), 330), 255, -1)
        cv2.putText(
            image, str(index + 1), (290, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 5, cv2.LINE_AA,
        )
        frames.append(image)
    return frames


def find_payload(sightings: list[Sighting]) -> dict:
    return {
        "devices": [
            {"address": item.address, "name": item.name, "rssi": item.rssi}
            for item in sightings
        ]
    }


def projection_payload(result: Projection) -> dict:
    return {
        "ok": True,
        "address": result.address,
        "name": result.name,
        "replaced": result.replaced,
        "files": list(result.files),
        "frames": result.frames,
        "play_seconds": result.play_seconds,
        "frame_ms": result.frame_ms,
    }


async def resolve_address(address: str | None, scan_seconds: float = 8.0) -> str:
    """Use the given address, or the only IrisGreen that answers.

    Zero devices and several devices both fail. Pass an address when more
    than one projector is in the room.
    """
    given = (address or "").strip()
    if given:
        return given
    found = await find(scan_seconds)
    if len(found) == 1:
        only = found[0]
        print(f"using {only.name} {only.address}", flush=True)
        return only.address
    if not found:
        raise RuntimeError("no IrisGreen is advertising")
    lines = "\n".join(
        f"  {item.name}  {item.address}  rssi={item.rssi}" for item in found
    )
    raise RuntimeError("more than one IrisGreen is nearby; pass --address\n" + lines)


async def deliver_mask(
    client: IrisBleClient,
    image: np.ndarray | str,
    seconds: float = 3600.0,
    settle: float = 0.0,
) -> None:
    """Overwrite ``1_1.bmp`` on an already-open client and start playback."""
    mask = prepare_frame(image)
    total, interval, _, _ = play_units(seconds, 1000)
    await push_image(client, build_stream(mask), f"{SLOT_GROUP}_1.bmp")
    await client.stop()
    await asyncio.sleep(0.3)
    played = await client.play(SLOT_GROUP, True, total, interval)
    if not played.ok:
        raise RuntimeError(played.error or "play was rejected")
    if settle:
        await asyncio.sleep(settle)


async def open_slot(address: str | None, scan_seconds: float = 8.0) -> ProjectorSession:
    """Resolve, connect, and delete group 1. The caller owns ``close()``."""
    chosen = await resolve_address(address, scan_seconds)
    opened = await connect(chosen)
    try:
        await opened.clear_slot()
    except Exception:
        await opened.close()
        raise
    return opened


async def connect(address: str) -> ProjectorSession:
    """Connect to one projector. Pass an address from find(); there is no default.

    Fails fast: the IrisGreen firmware wedges if a handshake is interrupted,
    so retrying only makes recovery slower.
    """
    if not address or not str(address).strip():
        raise ValueError("address is required; call find() and pass one of the results")
    client = IrisBleClient(address)
    try:
        await client.connect()
        await client.load_info()
        name = await client.read_gatt_string(CHAR_DEVICE_NAME)
        return ProjectorSession(client, name=name or client.info.model or address)
    except Exception:
        try:
            await client.disconnect()
        except Exception:
            pass
        raise


def _emit(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _image_paths(directory: str) -> list[str]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"not a directory: {directory}")
    paths = [
        str(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not paths:
        raise ValueError(f"no images in {directory}")
    return paths


def main(argv: list[str] | None = None) -> int:
    """JSON entry. stdout is one JSON object and nothing else."""
    parser = argparse.ArgumentParser(prog="python -m irisloop.session")
    sub = parser.add_subparsers(dest="command", required=True)

    find_cmd = sub.add_parser("find", help="list advertising IrisGreen devices")
    find_cmd.add_argument("--seconds", type=float, default=8.0)

    project = sub.add_parser("project", help="project the built-in moving bar")
    project.add_argument("--address", default=None)
    project.add_argument("--seconds", type=float, default=60.0)
    project.add_argument("--frame-ms", type=int, default=300)

    still = sub.add_parser("still", help="project one image into group 1")
    still.add_argument("image")
    still.add_argument("--address", default=None)
    still.add_argument("--seconds", type=float, default=60.0)

    frames = sub.add_parser("frames", help="project a directory of images into group 1")
    frames.add_argument("directory")
    frames.add_argument("--address", default=None)
    frames.add_argument("--seconds", type=float, default=60.0)
    frames.add_argument("--frame-ms", type=int, default=300)

    play = sub.add_parser("play", help="play a group already stored on the device")
    play.add_argument("--group", type=lambda value: int(value, 0), required=True)
    play.add_argument("--address", default=None)
    play.add_argument("--seconds", type=float, default=60.0)
    play.add_argument("--frame-ms", type=int, default=200)
    play.add_argument("--loop", action="store_true")

    stop = sub.add_parser("stop", help="stop playback")
    stop.add_argument("--address", default=None)

    blank = sub.add_parser("blank", help="replace group 1 with black")
    blank.add_argument("--address", default=None)
    blank.add_argument("--seconds", type=float, default=5.0)

    args = parser.parse_args(argv)
    try:
        if args.command == "find":
            _emit(find_payload(asyncio.run(find(args.seconds))))
            return 0
        _emit(asyncio.run(_dispatch(args)))
        return 0
    except Exception as exc:
        _emit({"ok": False, "error": str(exc)})
        return 1


async def _dispatch(args: argparse.Namespace) -> dict:
    if args.command == "project":
        return projection_payload(await _project_demo(args.address, args.seconds, args.frame_ms))
    if args.command == "still":
        return projection_payload(await _show_still(args.address, args.image, args.seconds))
    if args.command == "frames":
        return projection_payload(
            await _show_frames(args.address, args.directory, args.seconds, args.frame_ms)
        )
    if args.command == "play":
        return await _play_stored(
            args.address, args.group, args.seconds, args.frame_ms, args.loop
        )
    if args.command == "stop":
        return await _stop(args.address)
    if args.command == "blank":
        return projection_payload(await _blank(args.address, args.seconds))
    raise ValueError(f"unknown command {args.command}")


async def _project_demo(address: str | None, seconds: float, frame_ms: int) -> Projection:
    opened = await connect(await resolve_address(address))
    try:
        return await opened.show_animation(moving_bar_frames(), frame_ms=frame_ms, seconds=seconds)
    finally:
        await opened.close()


async def _show_still(address: str | None, image: str, seconds: float) -> Projection:
    opened = await connect(await resolve_address(address))
    try:
        return await opened.show_still(image, seconds=seconds)
    finally:
        await opened.close()


async def _show_frames(
    address: str | None, directory: str, seconds: float, frame_ms: int
) -> Projection:
    opened = await connect(await resolve_address(address))
    try:
        return await opened.show_animation(
            _image_paths(directory), frame_ms=frame_ms, seconds=seconds
        )
    finally:
        await opened.close()


async def _play_stored(
    address: str | None, group_id: int, seconds: float, frame_ms: int, loop: bool
) -> dict:
    chosen = await resolve_address(address)
    opened = await connect(chosen)
    try:
        await opened.play(group_id, seconds=seconds, frame_ms=frame_ms, loop=loop)
    finally:
        await opened.close()
    return {"ok": True, "address": chosen, "group": group_id}


async def _stop(address: str | None) -> dict:
    chosen = await resolve_address(address)
    opened = await connect(chosen)
    try:
        ok = await opened.stop()
    finally:
        await opened.close()
    return {"ok": ok, "address": chosen}


async def _blank(address: str | None, seconds: float) -> Projection:
    opened = await connect(await resolve_address(address))
    try:
        return await opened.blank(seconds=seconds)
    finally:
        await opened.close()


if __name__ == "__main__":
    raise SystemExit(main())
