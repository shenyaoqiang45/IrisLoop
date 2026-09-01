"""播放 IrisGreen 素材组。

素材组 ID 参考协议文档（MATERIAL_GROUPS）：
    1  = 开机        2 = 充电       3 = 未连接     4 = 连接成功
    5  = 低电量      0x0A=网格      0x0B=动画      0x0C=手写
    0x0D=提示词      0x0E=速度达成30KM  0x0F=里程    0x10=喝水
    0x11=速度达成40KM 0x12=光轮轨迹 0x13=爬坡      0x14=下坡
    ...  0x33=电量   0x37=计时器    ...

用法:
    python tools/play_group.py --group 1              # 播组1，不循环
    python tools/play_group.py --group 20 --loop      # 循环播组20
    python tools/play_group.py --group 1 --interval 50 --total 300
    python tools/play_group.py --stop                 # 停止播放
    python tools/play_group.py --group 1 --images test-data/1_*.bmp  # 先上传再播
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
    ap.add_argument("--group", type=lambda x: int(x, 0), help="素材组 ID，如 1 / 20 / 0x0B")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--total", type=int, default=100, help="总播放时长，单位 100ms")
    ap.add_argument("--interval", type=int, default=2, help="每帧间隔，单位 100ms")
    ap.add_argument("--stop", action="store_true", help="停止播放")
    args = ap.parse_args()

    if not args.stop and args.group is None:
        print("需要 --group 或 --stop")
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

        # 播放期间回读状态，确认设备在工作
        print("\n=== 状态观察 (6s) ===")
        for i in range(6):
            await asyncio.sleep(1.0)
            print(f"  {i+1}s ...", flush=True)
        return 0
    finally:
        await cli.disconnect()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
