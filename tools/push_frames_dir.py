"""Push a directory of frames to an IrisGreen group and loop it.

Reuses the writer frames already on disk — skips director / writer / review.
Useful for A/B-testing pack modes on the real device.

Usage:
    python tools/push_frames_dir.py captures/loop_round1/20260904_140804/writer/frames
    python tools/push_frames_dir.py <dir> --pack-mode dither --group 1 --interval 3
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import cv2
import numpy as np

from irisloop import protocol as P
from irisloop.ble_client import IrisBleClient
from irisloop.image_pack import build_stream, describe, to_1bpp
from irisloop.projector import CHAR_FILE_DATA, CHAR_FILE_END, CHAR_FILE_START

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"
ATT_HEADER_BYTES = 3
DEFAULT_ATT_MTU = 23
TARGET_MTU = 500
FILE_NAME_FIELD_MIN_BYTES = 16
START_DELAY_S = 0.1
PACKET_DELAY_S = 0.03
INTER_FILE_DELAY_S = 1.0


def build_name_packet(file_size: int, file_name: str) -> bytes:
    nb = file_name.encode("utf-8")
    field = max(FILE_NAME_FIELD_MIN_BYTES, len(nb))
    return file_size.to_bytes(4, "big") + nb.ljust(field, b"\x00")


async def push_one(client, file_name: str, stream: bytes) -> None:
    packet = build_name_packet(len(stream), file_name)
    # START: write with response, wait for device acknowledge
    rsp = await client.write_gatt_char(CHAR_FILE_START, packet, response=True)
    await asyncio.sleep(START_DELAY_S)
    mtu = getattr(client, "mtu_size", DEFAULT_ATT_MTU) or DEFAULT_ATT_MTU
    chunk = max(1, mtu - ATT_HEADER_BYTES)
    n = (len(stream) + chunk - 1) // chunk
    print(f"    {len(stream)}B in {n} chunks of {chunk}B", flush=True)
    for i in range(0, len(stream), chunk):
        await client.write_gatt_char(
            CHAR_FILE_DATA, stream[i : i + chunk], response=False
        )
        await asyncio.sleep(PACKET_DELAY_S)
    # END: write with response — device may need this to commit the file
    await client.write_gatt_char(CHAR_FILE_END, packet, response=True)
    await asyncio.sleep(0.2)  # commit delay


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames_dir", help="directory with frame_*.jpg")
    ap.add_argument("--pack-mode", default="dither",
                    choices=["threshold", "threshold_inv", "otsu", "dither", "dither_flat", "edges"])
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--interval", type=int, default=3, help="play interval, units of 100ms")
    ap.add_argument("--total", type=int, default=600, help="total play time, units of 100ms (default 60s)")
    ap.add_argument("--no-play", action="store_true", help="push only, do not start playback")
    args = ap.parse_args()

    frames = sorted(glob.glob(str(args.frames_dir).rstrip("/\\") + "/frame_*.jpg"))
    if not frames:
        frames = sorted(glob.glob(str(args.frames_dir).rstrip("/\\") + "/*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {args.frames_dir}")
    print(f"{len(frames)} frames from {args.frames_dir}, pack_mode={args.pack_mode}")

    cli = IrisBleClient(args.address)
    await cli.connect()
    print(f"connected mtu={cli.info.mtu}")
    # negotiate higher MTU for bulk transfer (same as push_image.py)
    if cli.info.mtu < TARGET_MTU:
        print(f"  negotiating MTU -> {TARGET_MTU}...")
        try:
            await cli.client.request_mtu(TARGET_MTU)
            print(f"  mtu now {cli.client.mtu_size}")
        except Exception as e:
            print(f"  MTU negotiation failed: {e}, keeping {cli.info.mtu}")
    try:
        # match loop_round1.py proven flow: stop first, then push, then play
        print("  stop any previous playback...")
        await cli.stop()
        await asyncio.sleep(0.5)
        r = await cli.send_command(P.cmd_delete_group(args.group))
        print(f"delete group{args.group}:", "ok" if r.ok else r.error)
        await asyncio.sleep(0.5)
        client = cli.client
        assert client is not None
        for i, f in enumerate(frames, 1):
            img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"cannot read {f}")
            if img.shape != (480, 640):
                img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
            bw = to_1bpp(img, args.pack_mode)
            stream = build_stream(bw)
            name = f"{args.group}_{i}.bmp"
            print(f"  [{i}/{len(frames)}] {name}  {describe(bw)}", flush=True)
            await push_one(client, name, stream)
            await asyncio.sleep(INTER_FILE_DELAY_S)
        if not args.no_play:
            await cli.stop()
            await asyncio.sleep(0.3)
            # total=600 (60s) with loop=True: device cycles the group for 60s,
            # not 36000 (1h) which leaves it stuck on the last frame
            r = await cli.play(
                group_id=args.group, loop=True,
                total_100ms=args.total, interval_100ms=args.interval,
            )
            print(f"play: {'ok' if r.ok else r.error} (loop, interval={args.interval*100}ms, total={args.total/10:.0f}s)")
    finally:
        await cli.disconnect()
        print("disconnected; projector keeps playing")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
