"""标定板检测预检：抓一帧海康图像，跑圆点检测+点阵匹配，输出叠加图目检。

用法:
    python tools/board_check.py [--exp 25000] [--thresh 80] [--out tmp/board_check.png]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from irisloop.hik_camera import HikCamera  # noqa: E402
from irisloop.procalib import (  # noqa: E402
    BoardModel, _black_image, _project, detect_board_dots, match_lattice,
)
from irisloop import protocol as P  # noqa: E402
from irisloop.ble_client import IrisBleClient  # noqa: E402
from irisloop.loop import DEFAULT_ADDR  # noqa: E402


async def go_dark(address: str) -> None:
    cli = IrisBleClient(address)
    await cli.connect()
    try:
        await cli.send_command(P.cmd_delete_group(1))
        await asyncio.sleep(0.5)
        await _project(cli, _black_image(), settle=1.0)
        print("  projector -> black")
    finally:
        await cli.disconnect()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=float, default=40000)
    ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--block", type=int, default=51)
    ap.add_argument("--c", type=int, default=8)
    ap.add_argument("--layout", default="cbinfo/CB-0300-W-001.layout.json")
    ap.add_argument("--out", default="tmp/board_check.png")
    ap.add_argument("--dark", action="store_true", help="先 BLE 推全黑再采集")
    ap.add_argument("--address", default=DEFAULT_ADDR)
    args = ap.parse_args()

    if args.dark:
        asyncio.run(go_dark(args.address))

    board = BoardModel.load(args.layout)
    cam = HikCamera()
    usb = [d for d in cam.enumerate() if d["interface"] == "USB3"]
    cam.open(serial=usb[0]["serial"] if usb else None)
    try:
        cam.set_gain(args.gain)
        cam.set_exposure_us(args.exp)
        cam.flush(3)
        ok, frame = cam.read(2000)
    finally:
        cam.close()
    if not ok or frame is None:
        print("[error] 抓帧失败")
        return 2

    dots = detect_board_dots(frame, args.block, args.c)
    matched = match_lattice(dots, board)
    print(f"blobs={len(dots)}  lattice-matched={len(matched) if matched else 0}/99")

    vis = frame.copy()
    if matched:
        fid = set()
        for (c, r), (x, y) in matched.items():
            big = (c, r) in board.fiducials
            fid.add((c, r))
            cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255) if big else (0, 255, 0), 2)
            cv2.putText(vis, f"{c},{r}", (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if big else (0, 255, 0), 1)
        print(f"fiducials found: {sorted(fid & board.fiducials)}")
    cv2.imwrite(args.out, vis)
    print(f"-> {args.out}")
    return 0 if matched and len(matched) >= 60 else 1


if __name__ == "__main__":
    raise SystemExit(main())
