"""IrisGreen BLE client.

Channel map (protocol doc + on-device GATT enumeration):
    adb40003 (write_indicate)  -> control commands, response required
    adb40004 (write_no_rsp)    -> data stream
    adb40002 (read_notify)     -> device status reports
    adb40006-...-2/3/4         -> file transfer: start / data / end
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from bleak import BleakClient

from . import protocol as P
from .projector import (
    CHAR_BATTERY_LEVEL,
    CHAR_FIRMWARE_REV,
    CHAR_MAIN_CMD,
    CHAR_MAIN_DATA,
    CHAR_MAIN_NOTIFY,
    CHAR_MANUFACTURER,
    CHAR_MODEL_NUMBER,
    CHAR_SERIAL_NUMBER,
    ProjectorInfo,
)

WRITE_TIMEOUT = 15.0


class IrisBleClient:
    def __init__(self, address: str):
        self.address = address
        self.client: BleakClient | None = None
        self.info = ProjectorInfo(address=address)
        self._notify_buf: list[bytes] = []
        self._notify_cb: Callable[[bytes], None] | None = None

    # ---------- connection ----------

    async def connect(self) -> None:
        self.client = BleakClient(self.address, timeout=20.0)
        await self.client.connect()
        try:
            self.info.mtu = self.client.mtu_size
        except Exception:
            pass

    async def disconnect(self) -> None:
        if self.client is not None:
            try:
                await self.client.disconnect()
            except Exception:
                pass
            self.client = None

    async def __aenter__(self) -> "IrisBleClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.disconnect()

    # ---------- command I/O ----------

    async def send_command(
        self,
        frame: bytes,
        wait_response: bool = True,
        timeout: float = WRITE_TIMEOUT,
    ) -> P.Response:
        """Send a command on adb40003 (write_indicate, with response)."""
        if self.client is None:
            raise RuntimeError("not connected")

        if not wait_response:
            await self.client.write_gatt_char(CHAR_MAIN_CMD, frame, response=True)
            return P.Response(ok=True, raw=frame)

        got: list[bytes] = []

        def _cb(_sender, data: bytearray):
            got.append(bytes(data))
            if self._notify_cb:
                self._notify_cb(bytes(data))

        # Protocol note: adb40003 is write_indicate; the response arrives as an
        # indicate on the same characteristic (not the adb40002 notify status frame).
        await self.client.start_notify(CHAR_MAIN_CMD, _cb)
        try:
            await self.client.write_gatt_char(CHAR_MAIN_CMD, frame, response=True)

            deadline = time.perf_counter() + timeout
            seen = 0
            while time.perf_counter() < deadline:
                while seen < len(got):
                    raw = got[seen]
                    seen += 1
                    resp = P.parse_response(raw)
                    if resp.ok and resp.cmd == frame[0]:
                        return resp
                await asyncio.sleep(0.02)

            if got:
                last = P.parse_response(got[-1])
                return P.Response(
                    ok=False, raw=last.raw,
                    error=f"no matching cmd=0x{frame[0]:02X}, received {len(got)} frame(s)",
                )
            return P.Response(ok=False, error="timeout, no response")
        finally:
            try:
                await self.client.stop_notify(CHAR_MAIN_NOTIFY)
            except Exception:
                pass

    async def write_data(self, payload: bytes) -> None:
        """Write the data-stream characteristic adb40004 (write-without-response)."""
        if self.client is None:
            raise RuntimeError("not connected")
        await self.client.write_gatt_char(CHAR_MAIN_DATA, payload, response=False)

    # ---------- device info ----------

    async def read_gatt_string(self, uuid: str) -> str:
        if self.client is None:
            raise RuntimeError("not connected")
        try:
            raw = await self.client.read_gatt_char(uuid)
            return bytes(raw).decode("ascii", errors="replace")
        except Exception:
            return ""

    async def read_gatt_int(self, uuid: str) -> int | None:
        if self.client is None:
            raise RuntimeError("not connected")
        try:
            raw = await self.client.read_gatt_char(uuid)
            b = bytes(raw)
            if len(b) >= 4:
                return int.from_bytes(b[:4], "little")
            if b:
                return b[0]
        except Exception:
            pass
        return None

    async def load_info(self) -> ProjectorInfo:
        self.info.name = await self.read_gatt_string(CHAR_MODEL_NUMBER) or ""
        self.info.model = await self.read_gatt_string(CHAR_MODEL_NUMBER)
        self.info.serial = await self.read_gatt_string(CHAR_SERIAL_NUMBER)
        self.info.firmware = await self.read_gatt_string(CHAR_FIRMWARE_REV)
        self.info.manufacturer = await self.read_gatt_string(CHAR_MANUFACTURER)
        raw_batt = b""
        try:
            if self.client is not None:
                raw_batt = bytes(await self.client.read_gatt_char(CHAR_BATTERY_LEVEL))
        except Exception:
            pass
        self.info.battery_raw = raw_batt
        return self.info

    # ---------- high-level commands ----------

    async def get_picture_count(self) -> int | None:
        resp = await self.send_command(P.build_read(P.CMD_PIC_COUNT))
        return resp.u16_le if resp.ok else None

    async def get_brightness(self) -> int | None:
        resp = await self.send_command(P.build_read(P.CMD_BRIGHTNESS))
        return resp.data[0] if resp.ok and resp.data else None

    async def set_brightness(self, percent: int) -> P.Response:
        return await self.send_command(P.cmd_set_brightness(percent))

    async def get_keystone_mode(self) -> int | None:
        resp = await self.send_command(P.build_read(P.CMD_KEYSTONE_MODE))
        return resp.data[0] if resp.ok and resp.data else None

    async def set_keystone_mode(self, mode: int) -> P.Response:
        return await self.send_command(P.cmd_set_keystone(mode))

    async def set_keystone_angle(self, angle: int) -> P.Response:
        return await self.send_command(P.cmd_set_keystone_angle(angle))

    async def set_fov(self, level: int) -> P.Response:
        return await self.send_command(P.cmd_set_fov(level))

    async def set_mirror(self, enable: bool) -> P.Response:
        return await self.send_command(P.cmd_set_mirror(enable))

    async def set_flip(self, upside_down: bool) -> P.Response:
        return await self.send_command(P.cmd_set_flip(upside_down))

    async def play(
        self,
        group_id: int,
        loop: bool = False,
        total_100ms: int = 100,
        interval_100ms: int = 2,
    ) -> P.Response:
        frame = P.cmd_play(group_id, loop, total_100ms, interval_100ms)
        return await self.send_command(frame)

    async def stop(self) -> P.Response:
        return await self.send_command(P.cmd_play_stop())
