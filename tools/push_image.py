"""Push a 640x480 1bpp image to IrisGreen over the BLE file-transfer service.

Protocol (reversed from iris-g-sdk: IrisProtocolConfig.kt + IrisUseCases.TransferFile):

    Channels (note: opposite of protocol xlsx R13/R14; SDK is authoritative):
        START = adb40006-...-0003  (write, with response)
        DATA  = adb40006-...-0002  (write-no-response)
        END   = adb40006-...-0004  (write, with response)

    START/END frame (identical contents, buildNamePacket):
        [0..3]  file size u32 big-endian
        [4..]   UTF-8 file name, at least 16 bytes, zero-padded on the right

    Flow:
        1. Negotiate MTU -> target 500 (fall back to 23)
        2. Write START frame -> wait 100ms
        3. DATA chunks (mtu-3) of file contents -> 30ms between packets
        4. Write END frame (same namePacket)

Usage:
    python tools/push_image.py --dry-run                # build local stream only
    python tools/push_image.py --name 11_20.bmp         # push a test image
    python tools/push_image.py --kind iris --name 1_1.bmp --verify
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from irisloop.ble_client import IrisBleClient
from irisloop.image_pack import build_stream, describe, make_test_image, save_bmp
from irisloop.projector import (
    CHAR_FILE_DATA,
    CHAR_FILE_END,
    CHAR_FILE_START,
)

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"

# SDK TransferFile constants
TARGET_MTU = 500
DEFAULT_ATT_MTU = 23
ATT_HEADER_BYTES = 3
FILE_NAME_FIELD_MIN_BYTES = 16
START_DELAY_S = 0.1
PACKET_DELAY_S = 0.03


def build_name_packet(file_size: int, file_name: str) -> bytes:
    """START/END frame: size u32 BE + file name (>=16B, zero-padded)."""
    if file_size < 0:
        raise ValueError("file_size must be non-negative")
    clean = file_name.strip().replace("\\", "/").split("/")[-1]
    if not clean:
        raise ValueError("file_name must not be empty")
    name_bytes = clean.encode("utf-8")
    field_size = max(FILE_NAME_FIELD_MIN_BYTES, len(name_bytes))
    return file_size.to_bytes(4, "big") + name_bytes.ljust(field_size, b"\x00")


async def push(
    addr: str,
    file_name: str,
    stream: bytes,
    verify: bool,
) -> bool:
    cli = IrisBleClient(addr)
    print(f"=== CONNECT {addr} ===")
    await cli.connect()
    print(f"  connected  mtu={cli.info.mtu}")

    packet = build_name_packet(len(stream), file_name)
    print(f"  namePacket {len(packet)}B: {packet.hex()}")
    print(f"  file {file_name}  {len(stream)}B")

    count_before = None
    if verify:
        count_before = await cli.get_picture_count()
        print(f"  picture count before transfer = {count_before}")

    client = cli.client
    assert client is not None

    try:
        # 1. START
        print("\n=== START -> ...0003 ===")
        await client.write_gatt_char(CHAR_FILE_START, packet, response=True)
        await asyncio.sleep(START_DELAY_S)

        # 2. DATA
        mtu = getattr(client, "mtu_size", DEFAULT_ATT_MTU) or DEFAULT_ATT_MTU
        chunk = max(1, mtu - ATT_HEADER_BYTES)
        n = (len(stream) + chunk - 1) // chunk
        print(f"=== DATA {len(stream)}B / chunk={chunk} = {n} packets -> ...0002 ===")
        t0 = time.perf_counter()
        for i in range(n):
            part = stream[i * chunk:(i + 1) * chunk]
            await client.write_gatt_char(CHAR_FILE_DATA, part, response=False)
            await asyncio.sleep(PACKET_DELAY_S)
            if (i + 1) % 20 == 0 or i == n - 1:
                el = time.perf_counter() - t0
                sent = min((i + 1) * chunk, len(stream))
                print(f"  {i+1}/{n}  {sent}B  {sent/el/1024:.1f} KB/s", flush=True)

        # 3. END (same packet)
        print("=== END -> ...0004 ===")
        await client.write_gatt_char(CHAR_FILE_END, packet, response=True)
        print("  transfer done, waiting for device to process...")
        await asyncio.sleep(2.0)

        # 4. Verify picture count (device may briefly disconnect while processing; reconnect)
        if verify:
            try:
                connected = client.is_connected
            except Exception:
                connected = False
            if not connected:
                print("  connection dropped, reconnecting...")
                await cli.disconnect()
                await asyncio.sleep(1.0)
                await cli.connect()
                client = cli.client
                assert client is not None
            count_after = await cli.get_picture_count()
            print(f"  picture count after transfer = {count_after}")
            if count_before is not None and count_after is not None:
                if count_after > count_before:
                    print(f"  [OK] picture count +{count_after - count_before}, device accepted")
                    return True
                print("  [WARN] picture count unchanged (same-name overwrite needs a pull-back compare)")
        return True
    finally:
        await cli.disconnect()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--kind", default="checker",
                    choices=["iris", "checker", "grid", "solid"])
    ap.add_argument("--name", default="11_20.bmp",
                    help="target filename (group_index.bmp, e.g. 11_20.bmp)")
    ap.add_argument("--dry-run", action="store_true", help="build local stream only, do not send")
    ap.add_argument("--verify", action="store_true", default=True,
                    help="read picture count before/after transfer (on by default)")
    args = ap.parse_args()

    img = make_test_image(args.kind)
    print(f"=== image {describe(img)} ===")
    save_bmp("captures/push_src.bmp", img)
    stream = build_stream(img)
    print(f"  stream {len(stream)}B (header 62 + data 38400)")

    if args.dry_run:
        p = build_name_packet(len(stream), args.name)
        print(f"  namePacket: {p.hex()}")
        return 0

    ok = asyncio.run(push(args.address, args.name, stream, args.verify))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
