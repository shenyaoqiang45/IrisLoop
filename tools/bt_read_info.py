"""读取 BLE 设备的可读特征：设备信息、电量、自定义状态特征。

用法:
  python tools/bt_read_info.py --address F4:12:FA:B6:B7:CA
  python tools/bt_read_info.py --address F4:12:FA:B6:B7:CA --notify 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from bleak import BleakClient

# 标准 Device Information Service 字段
DEVICE_INFO = {
    "00002a24-0000-1000-8000-00805f9b34fb": "Model Number",
    "00002a25-0000-1000-8000-00805f9b34fb": "Serial Number",
    "00002a26-0000-1000-8000-00805f9b34fb": "Firmware Revision",
    "00002a27-0000-1000-8000-00805f9b34fb": "Hardware Revision",
    "00002a28-0000-1000-8000-00805f9b34fb": "Software Revision",
    "00002a29-0000-1000-8000-00805f9b34fb": "Manufacturer",
    "00002a00-0000-1000-8000-00805f9b34fb": "Device Name",
    "00002a19-0000-1000-8000-00805f9b34fb": "Battery Level",
}

# 投影仪自定义服务里的可读特征
CUSTOM_READ = {
    "adb40001-b1c6-11ed-afa1-0242ac120001": "custom:adb40001 (read)",
    "adb40002-b1c6-11ed-afa1-0242ac120002": "custom:adb40002 (notify/read)",
}


def _hex(data: bytearray | bytes) -> str:
    return bytes(data).hex()


def _try_ascii(data: bytearray | bytes) -> str | None:
    b = bytes(data)
    if not b:
        return None
    printable = sum(1 for c in b if 32 <= c < 127 or c in (9, 10, 13))
    if printable / len(b) < 0.7:
        return None
    return b.decode("ascii", errors="replace")


async def read_char(client: BleakClient, uuid: str, label: str) -> None:
    try:
        raw = await client.read_gatt_char(uuid)
    except Exception as e:
        print(f"  {label:<34} [ERR] {type(e).__name__}: {e}")
        return

    hexs = _hex(raw)
    txt = _try_ascii(raw)
    extra = f"  ascii={txt!r}" if txt else ""
    ints = ""
    if len(raw) <= 4:
        ints = f"  u32le={int.from_bytes(raw, 'little')} u32be={int.from_bytes(raw, 'big')}"
    print(f"  {label:<34} len={len(raw):<4} {hexs}{extra}{ints}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True)
    ap.add_argument("--notify", type=float, default=0.0,
                    help="订阅 notify 特征并监听 N 秒（观察是否有主动上报）")
    args = ap.parse_args()

    client = BleakClient(args.address, timeout=20.0)
    print(f"=== CONNECT {args.address} ===")
    try:
        await client.connect()
    except Exception as e:
        print(f"[error] 连接失败: {type(e).__name__}: {e}")
        return 1

    try:
        print(f"  connected: {client.is_connected}")
        try:
            print(f"  mtu      : {client.mtu_size}")
        except Exception:
            pass

        print("\n=== DEVICE INFO (standard) ===")
        for uuid, label in DEVICE_INFO.items():
            await read_char(client, uuid, label)

        print("\n=== CUSTOM (projector) ===")
        for uuid, label in CUSTOM_READ.items():
            await read_char(client, uuid, label)

        if args.notify > 0:
            print(f"\n=== NOTIFY adb40002 ({args.notify}s) ===")
            got = []

            def cb(_sender, data: bytearray):
                got.append(bytes(data))
                print(f"  <- {_hex(data)}  len={len(data)}  ascii={_try_ascii(data)!r}")

            try:
                await client.start_notify(
                    "adb40002-b1c6-11ed-afa1-0242ac120002", cb
                )
                await asyncio.sleep(args.notify)
                await client.stop_notify(
                    "adb40002-b1c6-11ed-afa1-0242ac120002"
                )
            except Exception as e:
                print(f"  [ERR] notify: {type(e).__name__}: {e}")
            if not got:
                print("  (no notification during window)")

        return 0
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
