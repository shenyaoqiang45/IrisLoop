"""BLE 协议冒烟测试：只跑只读命令，不改动设备任何设置。

用法: python tools/ble_smoke.py F4:12:FA:B6:B7:CA
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from irisloop import protocol as P
from irisloop.ble_client import IrisBleClient


async def main(address: str) -> int:
    cli = IrisBleClient(address)
    print(f"=== CONNECT {address} ===")
    await cli.connect()
    print(f"  connected  mtu={cli.info.mtu}")

    print("\n=== DEVICE INFO ===")
    info = await cli.load_info()
    print(info.summary())

    print("\n=== READ-ONLY COMMANDS ===")

    # 读图片数
    frame = P.build_read(P.CMD_PIC_COUNT)
    print(f"  -> {P.describe(frame)}")
    r = await cli.send_command(frame, timeout=6.0)
    if r.ok:
        print(f"  <- ok  pics={r.u16_le}  raw={r.raw.hex()}")
    else:
        print(f"  <- {r.error or 'fail'}  raw={r.raw.hex()}")

    # 读亮度
    frame = P.build_read(P.CMD_BRIGHTNESS)
    print(f"  -> {P.describe(frame)}")
    r = await cli.send_command(frame, timeout=6.0)
    if r.ok:
        print(f"  <- ok  brightness={r.data[0] if r.data else '?'}%  raw={r.raw.hex()}")
    else:
        print(f"  <- {r.error or 'fail'}  raw={r.raw.hex()}")

    # 读畸变校正模式
    frame = P.build_read(P.CMD_KEYSTONE_MODE)
    print(f"  -> {P.describe(frame)}")
    r = await cli.send_command(frame, timeout=6.0)
    if r.ok:
        print(f"  <- ok  keystone_mode={r.data[0] if r.data else '?'}  raw={r.raw.hex()}")
    else:
        print(f"  <- {r.error or 'fail'}  raw={r.raw.hex()}")

    # 读 SN（走命令通道，验证小端 ASCII 解析）
    frame = P.build_read(P.CMD_SN)
    print(f"  -> {P.describe(frame)}")
    r = await cli.send_command(frame, timeout=6.0)
    if r.ok:
        print(f"  <- ok  sn={r.text_ascii!r}  raw={r.raw.hex()}")
    else:
        print(f"  <- {r.error or 'fail'}  raw={r.raw.hex()}")

    await cli.disconnect()
    print("\n=== DONE (未修改任何设备设置) ===")
    return 0


if __name__ == "__main__":
    addr = sys.argv[1] if len(sys.argv) > 1 else "F4:12:FA:B6:B7:CA"
    raise SystemExit(asyncio.run(main(addr)))
