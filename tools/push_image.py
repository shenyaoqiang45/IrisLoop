"""经 BLE 文件传输服务推送 640x480 1bpp 图片到 IrisGreen。

文件传输服务（协议文档 R12-R15）:
    adb40006-...-0001  read   文件传输服务 UUID
    adb40006-...-0002  write  传输开始
    adb40006-...-0003  write  传输数据
    adb40006-...-0004  write  传输结束

用法:
    python tools/push_image.py --dry-run                 # 只生成本地图
    python tools/push_image.py --probe                   # 只发开始帧，观察状态
    python tools/push_image.py                           # 完整推送
    python tools/push_image.py --kind checker --play 11
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import numpy as np

from irisloop import protocol as P
from irisloop.image_pack import build_stream, describe, make_test_image, save_bmp
from irisloop.projector import (
    CHAR_MAIN_CMD,
    CHAR_MAIN_NOTIFY,
    CHAR_SEC_WRITE_1,
    CHAR_SEC_WRITE_2,
    CHAR_SEC_WRITE_3,
)

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"


async def transfer(
    addr: str,
    stream: bytes,
    chunk: int,
    start_payload: bytes,
    end_payload: bytes,
    gap: float,
) -> None:
    from bleak import BleakClient

    log: list[tuple[float, str]] = []
    t0 = time.perf_counter()

    def _cb(_s, data: bytearray):
        b = bytes(data)
        if P.is_status_frame(b):
            log.append((time.perf_counter() - t0, f"STATUS {b.hex()}"))
        else:
            log.append((time.perf_counter() - t0, f"OTHER  {b.hex()}"))

    client = BleakClient(addr, timeout=20.0)
    print(f"=== CONNECT {addr} ===")
    await client.connect()
    print(f"  connected  mtu={client.mtu_size}")

    await client.start_notify(CHAR_MAIN_NOTIFY, _cb)

    try:
        print(f"\n=== START (adb40006-...-0002) {start_payload.hex()} ===")
        await client.write_gatt_char(CHAR_SEC_WRITE_2, start_payload, response=True)
        await asyncio.sleep(0.4)

        total = len(stream)
        n = (total + chunk - 1) // chunk
        print(f"=== DATA {total} bytes / {chunk} = {n} chunks ===")
        sent = 0
        t_start = time.perf_counter()
        for i in range(n):
            part = stream[i * chunk:(i + 1) * chunk]
            await client.write_gatt_char(CHAR_SEC_WRITE_3, part, response=False)
            sent += len(part)
            if gap:
                await asyncio.sleep(gap)
            if (i + 1) % 10 == 0 or i == n - 1:
                el = time.perf_counter() - t_start
                rate = sent / el / 1024 if el > 0 else 0
                print(f"  {i+1}/{n}  {sent}/{total}  {rate:.1f} KB/s", flush=True)
        el = time.perf_counter() - t_start
        print(f"  传输完成 {sent}B / {el:.2f}s = {sent/el/1024:.1f} KB/s")

        print(f"\n=== END (adb40006-...-0004) {end_payload.hex()} ===")
        await client.write_gatt_char(CHAR_SEC_WRITE_1, end_payload, response=True)
        await asyncio.sleep(1.0)

        print("\n=== 状态帧 ===")
        for ts, s in log[-12:]:
            print(f"  t={ts:6.2f}s  {s}")
        if not log:
            print("  (无状态帧)")
    finally:
        try:
            await client.stop_notify(CHAR_MAIN_NOTIFY)
        except Exception:
            pass
        await client.disconnect()


async def probe(addr: str, start_payload: bytes, seconds: float) -> None:
    from bleak import BleakClient

    log: list[tuple[float, bytes]] = []
    t0 = time.perf_counter()

    def _cb(_s, data: bytearray):
        log.append((time.perf_counter() - t0, bytes(data)))

    client = BleakClient(addr, timeout=20.0)
    print(f"=== PROBE {addr} ===")
    await client.connect()
    print(f"  connected  mtu={client.mtu_size}")
    await client.start_notify(CHAR_MAIN_NOTIFY, _cb)

    await asyncio.sleep(1.0)
    base = len(log)
    print(f"  基线 {base} 帧")

    print(f"  发送 START -> {start_payload.hex()}")
    await client.write_gatt_char(CHAR_SEC_WRITE_2, start_payload, response=True)
    await asyncio.sleep(seconds)

    await client.stop_notify(CHAR_MAIN_NOTIFY)
    await client.disconnect()

    print(f"\n=== 收到 {len(log)} 帧 ===")
    for ts, b in log[:20]:
        tag = "STATUS" if P.is_status_frame(b) else "OTHER"
        print(f"  t={ts:6.2f}s  {b.hex():<24} {tag}")
    if len(log) > base:
        print(f"\n  START 后新增 {len(log)-base} 帧")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--kind", default="iris",
                    choices=["iris", "checker", "grid", "solid"])
    ap.add_argument("--chunk", type=int, default=244, help="BLE 分包大小")
    ap.add_argument("--gap", type=float, default=0.0, help="包间延时秒")
    ap.add_argument("--dry-run", action="store_true", help="只生成本地图")
    ap.add_argument("--probe", action="store_true", help="只发开始帧")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    img = make_test_image(args.kind)
    print(f"=== 图像 {describe(img)} ===")
    save_bmp("captures/test_640x480_1bpp.bmp", img)
    print("  已保存 captures/test_640x480_1bpp.bmp")

    stream = build_stream(img)
    print(f"  设备流 {len(stream)} 字节 (头62 + 数据38400)")

    if args.dry_run:
        return 0

    if args.probe:
        asyncio.run(probe(args.address, b"\x01", args.seconds))
        return 0

    asyncio.run(
        transfer(
            args.address, stream, args.chunk,
            start_payload=b"\x01", end_payload=b"\x01", gap=args.gap,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
