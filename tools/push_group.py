"""把一组图片(JPG/PNG)推送到 IrisGreen 指定素材组（覆盖式）。

流程: 读取图片 -> 灰度 -> 二值化 -> image_pack 打包 38462B -> 按 <gid>_<n>.bmp 命名推送。

组 1（开机）最大 10 张，敦煌素材 30 帧只取前 10。

用法:
    python tools/push_group.py --dir "data/敦煌_frames" --group 1 --count 10
    python tools/push_group.py --dir "data/敦煌_frames" --group 1 --count 10 --start-index 1
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import cv2
import numpy as np

from irisloop.ble_client import IrisBleClient
from irisloop.image_pack import binarize, build_stream, describe
from irisloop.projector import CHAR_FILE_DATA, CHAR_FILE_END, CHAR_FILE_START

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"

TARGET_MTU = 500
DEFAULT_ATT_MTU = 23
ATT_HEADER_BYTES = 3
FILE_NAME_FIELD_MIN_BYTES = 16
START_DELAY_S = 0.1
PACKET_DELAY_S = 0.03
INTER_FILE_DELAY_S = 1.0


def build_name_packet(file_size: int, file_name: str) -> bytes:
    name_bytes = file_name.encode("utf-8")
    field_size = max(FILE_NAME_FIELD_MIN_BYTES, len(name_bytes))
    return file_size.to_bytes(4, "big") + name_bytes.ljust(field_size, b"\x00")


def load_image_as_bw(path: str) -> np.ndarray:
    """读图 -> 灰度 -> 缩放到 640x480 -> 二值化。"""
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取 {path}")
    if img.shape != (480, 640):
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    return binarize(img, threshold=127)


async def push_one(client, file_name: str, stream: bytes) -> None:
    packet = build_name_packet(len(stream), file_name)
    mtu = getattr(client, "mtu_size", DEFAULT_ATT_MTU) or DEFAULT_ATT_MTU
    chunk = max(1, mtu - ATT_HEADER_BYTES)
    n = (len(stream) + chunk - 1) // chunk

    await client.write_gatt_char(CHAR_FILE_START, packet, response=True)
    await asyncio.sleep(START_DELAY_S)
    for i in range(n):
        await client.write_gatt_char(
            CHAR_FILE_DATA, stream[i * chunk:(i + 1) * chunk], response=False
        )
        await asyncio.sleep(PACKET_DELAY_S)
    await client.write_gatt_char(CHAR_FILE_END, packet, response=True)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--dir", required=True, help="图片目录")
    ap.add_argument("--group", type=int, required=True, help="目标素材组 ID")
    ap.add_argument("--count", type=int, default=10, help="推送张数（默认 10）")
    ap.add_argument("--start-index", type=int, default=1, help="源图起始序号（默认 1）")
    args = ap.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.dir, "*.jpg"))
        + glob.glob(os.path.join(args.dir, "*.png")),
        key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0),
    )
    if not files:
        print(f"[error] {args.dir} 下无 jpg/png")
        return 1

    # 取 start-index 起的 count 张
    src = files[args.start_index - 1: args.start_index - 1 + args.count]
    print(f"=== 准备 {len(src)} 张 -> 组 {args.group} ===")

    cli = IrisBleClient(args.address)
    print(f"=== CONNECT {args.address} ===")
    await cli.connect()
    print(f"  connected  mtu={cli.info.mtu}")
    count0 = await cli.get_picture_count()
    print(f"  推送前图片总数 = {count0}")

    client = cli.client
    assert client is not None

    try:
        for i, path in enumerate(src, 1):
            bw = load_image_as_bw(path)
            stream = build_stream(bw)
            name = f"{args.group}_{i}.bmp"
            print(f"\n[{i}/{len(src)}] {os.path.basename(path)} -> {name}  "
                  f"{describe(bw)}")
            t0 = time.perf_counter()
            await push_one(client, name, stream)
            el = time.perf_counter() - t0
            print(f"  done  {len(stream)}B  {el:.1f}s  {len(stream)/el/1024:.1f} KB/s")
            await asyncio.sleep(INTER_FILE_DELAY_S)

        await asyncio.sleep(2.0)
        count1 = await cli.get_picture_count()
        print(f"\n=== 推送后图片总数 = {count1} (前={count0}) ===")
        print(f"已完成 {len(src)} 张推送 -> 组 {args.group}")
        print(f"播放验证: python tools/play_group.py --group {args.group} "
              f"--total {len(src) * 3} --interval 3")
    finally:
        await cli.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
