"""Listen on adb40002 and inspect status-frame layout and cadence.

Optionally send one read-only command to see whether a command response is mixed in.

Usage:
  python tools/ble_listen.py F4:12:FA:B6:B7:CA 12
  python tools/ble_listen.py F4:12:FA:B6:B7:CA 12 --cmd 11
"""

from __future__ import annotations

import asyncio
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from irisloop import protocol as P
from irisloop.ble_client import IrisBleClient
from irisloop.projector import CHAR_MAIN_CMD, CHAR_MAIN_NOTIFY


async def main(address: str, seconds: float, cmd: int | None) -> int:
    from bleak import BleakClient

    client = BleakClient(address, timeout=20.0)
    print(f"=== CONNECT {address} ===")
    await client.connect()
    print(f"  connected  mtu={client.mtu_size}")

    frames: list[tuple[float, bytes]] = []
    t0 = time.perf_counter()

    def _cb(_sender, data: bytearray):
        frames.append((time.perf_counter() - t0, bytes(data)))

    await client.start_notify(CHAR_MAIN_NOTIFY, _cb)

    sent_at = None
    if cmd is not None:
        frame = P.build_read(cmd)
        await asyncio.sleep(0.5)
        sent_at = time.perf_counter() - t0
        print(f"\n  -> send read cmd=0x{cmd:02X} : {frame.hex()}")
        await client.write_gatt_char(CHAR_MAIN_CMD, frame, response=True)

    print(f"\n=== LISTEN {seconds}s ===")
    await asyncio.sleep(seconds)
    await client.stop_notify(CHAR_MAIN_NOTIFY)
    await client.disconnect()

    print(f"  received {len(frames)} frames")
    if sent_at is not None:
        print(f"  command sent at t={sent_at:.2f}s")

    print("\n=== FRAMES ===")
    prev: bytes | None = None
    for ts, raw in frames:
        tag = ""
        if P.is_status_frame(raw):
            tag = "STATUS"
        elif raw and raw[0] in (0x80, 0x08):
            r = P.parse_response(raw)
            tag = f"RESP cmd=0x{r.cmd:02X} len={len(r.data)}"
        else:
            tag = "OTHER"

        diff = ""
        if prev is not None and len(prev) == len(raw):
            changed = [i for i in range(len(raw)) if prev[i] != raw[i]]
            if changed:
                diff = "  Δ" + ",".join(
                    f"[{i}]{prev[i]:02X}->{raw[i]:02X}" for i in changed
                )
        prev = raw

        mark = ""
        if sent_at is not None and abs(ts - sent_at) < 0.3:
            mark = "  <<< command send"
        print(f"  t={ts:6.2f}s  {raw.hex():<24} {tag}{diff}{mark}")

    # Cadence stats
    if len(frames) >= 2:
        gaps = [frames[i + 1][0] - frames[i][0] for i in range(len(frames) - 1)]
        avg = sum(gaps) / len(gaps)
        print(f"\n  mean frame interval: {avg*1000:.0f} ms  ({1/avg:.1f} fps)" if avg > 0 else "")

    status_n = sum(1 for _, r in frames if P.is_status_frame(r))
    resp_n = sum(1 for _, r in frames if r and r[0] in (0x80, 0x08))
    print(f"\n  status frames {status_n} / command responses {resp_n}")
    return 0


if __name__ == "__main__":
    addr = sys.argv[1] if len(sys.argv) > 1 else "F4:12:FA:B6:B7:CA"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    c = None
    if "--cmd" in sys.argv:
        c = int(sys.argv[sys.argv.index("--cmd") + 1], 16 if "x" in sys.argv[sys.argv.index("--cmd") + 1].lower() else 10)
    raise SystemExit(asyncio.run(main(addr, secs, c)))
