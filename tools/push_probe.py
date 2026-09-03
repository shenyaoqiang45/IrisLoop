"""BLE file-transfer push probe — determine START/END frame format.

Known:
  - Protocol docs list channel roles but not START/END frame layout
  - Firmware-upgrade sample frame: 00 0F 45 10 + "xingzhe_ctrl_mems.bin"
    where 0x000F4510 = 1002768 ≈ 1MB, likely "file size (u32 BE) + file name"
  - On-device images are 38462-byte full BMPs (including 62-byte header)
  - Old push_image.py wrote data to 0004 (END) and END to 0003 (DATA)

How we decide:
  - cmd 0x05 picture count (increases when pushed to a new slot)
  - Unusual changes on adb40002 status frames / adb40003 indicate replies

Usage:
  python tools/push_probe.py                 # all stages: baseline -> START candidates -> full transfer
  python tools/push_probe.py --stage base    # baseline only
  python tools/push_probe.py --stage start   # START candidates only
  python tools/push_probe.py --stage full --start-fmt size_be_name --end-fmt one
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from irisloop import protocol as P
from irisloop.image_pack import build_stream, describe, make_test_image, save_bmp
from irisloop.projector import (
    CHAR_MAIN_CMD,
    CHAR_MAIN_NOTIFY,
    CHAR_SEC_WRITE_1,  # adb40006-...-0003 data
    CHAR_SEC_WRITE_2,  # adb40006-...-0002 start
    CHAR_SEC_WRITE_3,  # adb40006-...-0004 end
)

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"

# Device-generated BMP header (first 62 bytes of test-data/1_1.bmp). Differs from
# image_pack.BMP_HEAD only in xPelsPerMeter/yPelsPerMeter (0 vs 2835) and clrUsed (2 vs 0).
# Use the device header when pushing so firmware header checks do not reject us.
DEVICE_BMP_HEAD = bytes.fromhex(
    "424D3E960000000000003E0000002800000080020000E00100000100010000"
    "00000000960000000000000000000002000000000000000000000000FFFFFF00"
)

CH_START = CHAR_SEC_WRITE_2   # ...0002 start
CH_DATA = CHAR_SEC_WRITE_1    # ...0003 data
CH_END = CHAR_SEC_WRITE_3     # ...0004 end


def build_start(fmt: str, name: str, size: int) -> bytes:
    n = name.encode("ascii")
    if fmt == "size_be_name":
        return size.to_bytes(4, "big") + n
    if fmt == "size_le_name":
        return size.to_bytes(4, "little") + n
    if fmt == "name":
        return n
    if fmt == "type_name":
        return b"\x01" + n
    if fmt == "type_size_le_name":
        return b"\x01" + size.to_bytes(4, "little") + n
    if fmt == "one":
        return b"\x01"
    raise ValueError(fmt)


def build_end(fmt: str, size: int) -> bytes | None:
    if fmt == "one":
        return b"\x01"
    if fmt == "size_le":
        return size.to_bytes(4, "little")
    if fmt == "none":
        return None
    raise ValueError(fmt)


class Probe:
    def __init__(self, addr: str):
        self.addr = addr
        self.client = None
        self.frames: list[tuple[float, str, bytes]] = []
        self.t0 = time.perf_counter()

    def _log(self, ch: str, data: bytearray):
        b = bytes(data)
        tag = "STATUS" if P.is_status_frame(b) else "OTHER "
        self.frames.append((time.perf_counter() - self.t0, f"{ch}:{tag}", b))

    async def connect(self):
        from bleak import BleakClient

        self.client = BleakClient(self.addr, timeout=20.0)
        await self.client.connect()
        print(f"connected  mtu={self.client.mtu_size}")
        await self.client.start_notify(CHAR_MAIN_NOTIFY,
                                       lambda s, d: self._log("40002", d))
        try:
            await self.client.start_notify(CHAR_MAIN_CMD,
                                           lambda s, d: self._log("40003", d))
        except Exception as e:
            print(f"  (adb40003 indicate subscribe failed: {e})")

    async def disconnect(self):
        if self.client:
            try:
                await self.client.disconnect()
            except Exception:
                pass
            self.client = None

    async def pic_count(self) -> int | None:
        """Read picture count via adb40003."""
        got: list[bytes] = []
        assert self.client is not None

        def _cb(_s, d: bytearray):
            got.append(bytes(d))

        try:
            await self.client.start_notify(CHAR_MAIN_CMD, _cb)
        except Exception:
            pass
        await self.client.write_gatt_char(CHAR_MAIN_CMD,
                                          P.build_read(P.CMD_PIC_COUNT),
                                          response=True)
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            for raw in got:
                r = P.parse_response(raw)
                if r.ok and r.cmd == P.CMD_PIC_COUNT:
                    return r.u16_le
            await asyncio.sleep(0.05)
        return None

    def new_frames_since(self, mark: int) -> list[tuple[float, str, bytes]]:
        return self.frames[mark:]

    def dump_since(self, mark: int, limit: int = 16):
        fresh = self.frames[mark:]
        if not fresh:
            print("    (no new frames)")
            return
        for ts, ch, b in fresh[:limit]:
            print(f"    t={ts:6.2f}s {ch} {b.hex()}")
        if len(fresh) > limit:
            print(f"    ... {len(fresh)} frames total")


async def stage_base(p: Probe) -> None:
    print("\n--- baseline: picture count + channel activity ---")
    n = await p.pic_count()
    print(f"  picture count = {n}")
    mark = len(p.frames)
    await asyncio.sleep(2.0)
    p.dump_since(mark)


async def stage_start(p: Probe, name: str, size: int, fmts: list[str]) -> None:
    print("\n--- START frame candidates (START only, watch 2s) ---")
    for fmt in fmts:
        payload = build_start(fmt, name, size)
        mark = len(p.frames)
        print(f"  [{fmt}] -> {payload.hex()} ({len(payload)}B)")
        try:
            await p.client.write_gatt_char(CH_START, payload, response=True)
            print("    write ok")
        except Exception as e:
            print(f"    write FAIL: {e}")
            continue
        await asyncio.sleep(2.0)
        p.dump_since(mark)


async def stage_full(p: Probe, name: str, stream: bytes,
                     start_fmt: str, end_fmt: str,
                     chunk: int, gap: float) -> None:
    print(f"\n--- full transfer {name} ({len(stream)}B) "
          f"start={start_fmt} end={end_fmt} chunk={chunk} ---")

    before = await p.pic_count()
    print(f"  picture count before transfer = {before}")

    start_payload = build_start(start_fmt, name, len(stream))
    mark = len(p.frames)
    print(f"  START -> {start_payload.hex()}")
    await p.client.write_gatt_char(CH_START, start_payload, response=True)
    await asyncio.sleep(0.5)

    n = (len(stream) + chunk - 1) // chunk
    print(f"  DATA {len(stream)}B / {chunk} = {n} packets -> adb40006-0003")
    sent = 0
    t = time.perf_counter()
    for i in range(n):
        part = stream[i * chunk:(i + 1) * chunk]
        await p.client.write_gatt_char(CH_DATA, part, response=False)
        sent += len(part)
        if gap:
            await asyncio.sleep(gap)
        if (i + 1) % 50 == 0 or i == n - 1:
            el = time.perf_counter() - t
            print(f"    {i+1}/{n}  {sent}B  {sent/el/1024:.1f} KB/s", flush=True)

    end_payload = build_end(end_fmt, len(stream))
    if end_payload is not None:
        print(f"  END -> {end_payload.hex()}")
        await p.client.write_gatt_char(CH_END, end_payload, response=True)
    else:
        print("  END (not sent)")

    await asyncio.sleep(2.0)
    print("  new frames during transfer:")
    p.dump_since(mark, limit=24)

    # Device often disconnects after a complete file to process it; reconnect then re-read
    await asyncio.sleep(1.0)
    print("  force reconnect after transfer, then re-read picture count...")
    await p.disconnect()
    await asyncio.sleep(1.0)
    await p.connect()
    after = await p.pic_count()
    print(f"  picture count after transfer = {after}")
    if before is not None and after is not None:
        if after > before:
            print(f"  picture count +{after-before}, device accepted the file")
        elif after == before:
            print("  picture count unchanged: maybe same-name overwrite (needs pull-back compare), or rejected")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--stage", default="all",
                    choices=["base", "start", "full", "all"])
    ap.add_argument("--name", default="11_20.bmp",
                    help="target filename (default 11_20.bmp: last animation-group slot, likely a new add)")
    ap.add_argument("--kind", default="checker")
    ap.add_argument("--start-fmt", default="size_be_name",
                    choices=["size_be_name", "size_le_name", "name",
                             "type_name", "type_size_le_name", "one"])
    ap.add_argument("--end-fmt", default="one",
                    choices=["one", "size_le", "none"])
    ap.add_argument("--chunk", type=int, default=244)
    ap.add_argument("--gap", type=float, default=0.0)
    ap.add_argument("--device-head", action="store_true", default=True,
                    help="use the device's 62B header (on by default)")
    args = ap.parse_args()

    img = make_test_image(args.kind)
    stream = build_stream(img)
    if args.device_head:
        stream = DEVICE_BMP_HEAD + stream[62:]
    print(f"=== image {describe(img)}  stream {len(stream)}B ===")
    save_bmp("captures/push_probe_src.bmp", img)

    p = Probe(args.address)
    print(f"=== CONNECT {args.address} ===")
    await p.connect()
    try:
        if args.stage in ("base", "all"):
            await stage_base(p)
        if args.stage in ("start", "all"):
            await stage_start(p, args.name, len(stream),
                              ["size_be_name", "size_le_name", "name",
                               "type_name", "type_size_le_name", "one"])
        if args.stage in ("full", "all"):
            await stage_full(p, args.name, stream,
                             args.start_fmt, args.end_fmt,
                             args.chunk, args.gap)
    finally:
        await p.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
