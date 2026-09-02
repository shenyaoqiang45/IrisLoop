"""推送单张图片到 1_1.bmp 并常亮显示 —— 最小闭环专用。

用法:
    python tools/push_single.py data/01a_upload_alignment_h.jpg
    python tools/push_single.py data/foo.png --name 1_1.bmp --no-play
"""

from __future__ import annotations

import argparse
import asyncio
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import cv2
import numpy as np

from irisloop.ble_client import IrisBleClient
from irisloop.image_pack import binarize, build_stream, describe
from irisloop.projector import CHAR_FILE_DATA, CHAR_FILE_END, CHAR_FILE_START

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"

ATT_HEADER_BYTES = 3
DEFAULT_ATT_MTU = 23
FILE_NAME_FIELD_MIN_BYTES = 16
START_DELAY_S = 0.1
PACKET_DELAY_S = 0.03


def build_name_packet(file_size: int, file_name: str) -> bytes:
    name_bytes = file_name.encode("utf-8")
    field_size = max(FILE_NAME_FIELD_MIN_BYTES, len(name_bytes))
    return file_size.to_bytes(4, "big") + name_bytes.ljust(field_size, b"\x00")


def load_bw(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取 {path}")
    if img.shape != (480, 640):
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    return binarize(img)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="源图片路径")
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--name", default="1_1.bmp", help="目标文件名（默认 1_1.bmp）")
    ap.add_argument("--no-play", action="store_true", help="推送后不触发播放")
    args = ap.parse_args()

    bw = load_bw(args.image)
    stream = build_stream(bw)
    print(f"=== {args.image} ===")
    print(f"  {describe(bw)}  流={len(stream)}B")

    cli = IrisBleClient(args.address)
    print(f"=== CONNECT {args.address} ===")
    await cli.connect()
    client = cli.client
    assert client is not None

    try:
        # 先清空组 1，避免残留多帧轮播（如之前推入的敦煌 1_2..1_10）
        from irisloop import protocol as P
        print("=== 清空组 1 (0xA2) ===")
        r = await cli.send_command(P.cmd_delete_group(1))
        print(f"  {'ok' if r.ok else r.error}")
        await asyncio.sleep(0.5)

        packet = build_name_packet(len(stream), args.name)
        print(f"=== PUSH {args.name} ===")
        await client.write_gatt_char(CHAR_FILE_START, packet, response=True)
        await asyncio.sleep(START_DELAY_S)

        mtu = getattr(client, "mtu_size", DEFAULT_ATT_MTU) or DEFAULT_ATT_MTU
        chunk = max(1, mtu - ATT_HEADER_BYTES)
        n = (len(stream) + chunk - 1) // chunk
        for i in range(n):
            await client.write_gatt_char(
                CHAR_FILE_DATA, stream[i * chunk:(i + 1) * chunk], response=False
            )
            await asyncio.sleep(PACKET_DELAY_S)
        await client.write_gatt_char(CHAR_FILE_END, packet, response=True)
        print(f"  pushed {len(stream)}B / {n} 包")

        if not args.no_play:
            # 组 1 单张播放：interval 设大避免闪烁（实际只有 1 张无所谓）
            from irisloop import protocol as P
            await cli.stop()
            await asyncio.sleep(0.3)
            # 读图片数确认删除生效
            n = await cli.get_picture_count()
            print(f"  当前图片总数 = {n}")
            r = await cli.play(group_id=1, loop=True, total_100ms=36000, interval_100ms=50)
            print(f"  play group1: {'ok' if r.ok else r.error}")
    finally:
        await cli.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
