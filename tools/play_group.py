"""Play an IrisGreen material group.

Material group IDs from the protocol (MATERIAL_GROUPS):
    1  = boot        2 = charging    3 = disconnected   4 = connected
    5  = low battery  0x0A=grid      0x0B=animation     0x0C=handwriting
    0x0D=prompt       0x0E=speed 30KM  0x0F=odometer    0x10=drink water
    0x11=speed 40KM   0x12=light-wheel trail  0x13=climb  0x14=descend
    ...  0x33=battery   0x37=timer    ...

Usage:
    python tools/play_group.py --group 1              # play group 1, no loop
    python tools/play_group.py --group 20 --loop      # loop group 20
    python tools/play_group.py --group 1 --interval 50 --total 300
    python tools/play_group.py --stop                 # stop playback
    python tools/play_group.py --group 1 --images test-data/1_*.bmp  # upload then play
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from irisloop import protocol as P
from irisloop.ble_client import IrisBleClient

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--group", type=lambda x: int(x, 0), help="material group ID, e.g. 1 / 20 / 0x0B")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--total", type=int, default=100, help="total play time, units of 100ms")
    ap.add_argument("--interval", type=int, default=2, help="per-frame interval, units of 100ms")
    ap.add_argument("--stop", action="store_true", help="stop playback")
    args = ap.parse_args()

    if not args.stop and args.group is None:
        print("need --group or --stop")
        return 1

    cli = IrisBleClient(args.address)
    print(f"=== CONNECT {args.address} ===")
    await cli.connect()
    print(f"  connected  mtu={cli.info.mtu}")

    try:
        if args.stop:
            print("\n=== STOP ===")
            r = await cli.stop()
            print(f"  {'ok' if r.ok else 'fail'}  raw={r.raw.hex()}  {r.error}")
            return 0

        gname = P.group_name(args.group)
        print(f"\n=== PLAY group={args.group} ({gname}) ===")
        print(f"  loop={args.loop} total={args.total*100}ms interval={args.interval*100}ms")

        r = await cli.play(
            group_id=args.group,
            loop=args.loop,
            total_100ms=args.total,
            interval_100ms=args.interval,
        )
        if r.ok:
            print(f"  ok  raw={r.raw.hex()}")
        else:
            print(f"  {r.error}  raw={r.raw.hex()}")

        # Re-read status during play to confirm the device is working
        print("\n=== status watch (6s) ===")
        for i in range(6):
            await asyncio.sleep(1.0)
            print(f"  {i+1}s ...", flush=True)
        return 0
    finally:
        await cli.disconnect()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
