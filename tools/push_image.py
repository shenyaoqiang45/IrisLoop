"""经 BLE 文件传输服务推送 640x480 1bpp 图片到 IrisGreen。

协议（逆向自 iris-g-sdk: IrisProtocolConfig.kt + IrisUseCases.TransferFile）：

    通道（注意与协议 xlsx R13/R14 描述相反，以 SDK 为准）:
        START = adb40006-...-0003  (write, 带响应)
        DATA  = adb40006-...-0002  (write-no-response)
        END   = adb40006-...-0004  (write, 带响应)

    START/END 帧（内容完全相同，buildNamePacket）:
        [0..3]  文件大小 u32 大端
        [4..]   文件名 UTF-8，至少 16 字节，不足右侧补 0x00

    流程:
        1. 协商 MTU -> 目标 500（失败回落 23）
        2. 写 START 帧 -> 延时 100ms
        3. DATA 分包 (mtu-3) 写文件内容 -> 每包间隔 30ms
        4. 写 END 帧（同一 namePacket）

用法:
    python tools/push_image.py --dry-run                # 只生成本地流
    python tools/push_image.py --name 11_20.bmp         # 推送测试图
    python tools/push_image.py --kind iris --name 1_1.bmp --verify
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from irisloop.ble_client import IrisBleClient
from irisloop.image_pack import build_stream, describe, make_test_image, save_bmp
from irisloop.projector import (
    CHAR_FILE_DATA,
    CHAR_FILE_END,
    CHAR_FILE_START,
)

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"

# SDK TransferFile 常量
TARGET_MTU = 500
DEFAULT_ATT_MTU = 23
ATT_HEADER_BYTES = 3
FILE_NAME_FIELD_MIN_BYTES = 16
START_DELAY_S = 0.1
PACKET_DELAY_S = 0.03


def build_name_packet(file_size: int, file_name: str) -> bytes:
    """START/END 帧: 大小 u32 BE + 文件名(>=16B, 不足补 0)。"""
    if file_size < 0:
        raise ValueError("file_size 必须非负")
    clean = file_name.strip().replace("\\", "/").split("/")[-1]
    if not clean:
        raise ValueError("file_name 不能为空")
    name_bytes = clean.encode("utf-8")
    field_size = max(FILE_NAME_FIELD_MIN_BYTES, len(name_bytes))
    return file_size.to_bytes(4, "big") + name_bytes.ljust(field_size, b"\x00")


async def push(
    addr: str,
    file_name: str,
    stream: bytes,
    verify: bool,
) -> bool:
    cli = IrisBleClient(addr)
    print(f"=== CONNECT {addr} ===")
    await cli.connect()
    print(f"  connected  mtu={cli.info.mtu}")

    packet = build_name_packet(len(stream), file_name)
    print(f"  namePacket {len(packet)}B: {packet.hex()}")
    print(f"  文件 {file_name}  {len(stream)}B")

    count_before = None
    if verify:
        count_before = await cli.get_picture_count()
        print(f"  传输前图片总数 = {count_before}")

    client = cli.client
    assert client is not None

    try:
        # 1. START
        print("\n=== START -> ...0003 ===")
        await client.write_gatt_char(CHAR_FILE_START, packet, response=True)
        await asyncio.sleep(START_DELAY_S)

        # 2. DATA
        mtu = getattr(client, "mtu_size", DEFAULT_ATT_MTU) or DEFAULT_ATT_MTU
        chunk = max(1, mtu - ATT_HEADER_BYTES)
        n = (len(stream) + chunk - 1) // chunk
        print(f"=== DATA {len(stream)}B / chunk={chunk} = {n} 包 -> ...0002 ===")
        t0 = time.perf_counter()
        for i in range(n):
            part = stream[i * chunk:(i + 1) * chunk]
            await client.write_gatt_char(CHAR_FILE_DATA, part, response=False)
            await asyncio.sleep(PACKET_DELAY_S)
            if (i + 1) % 20 == 0 or i == n - 1:
                el = time.perf_counter() - t0
                sent = min((i + 1) * chunk, len(stream))
                print(f"  {i+1}/{n}  {sent}B  {sent/el/1024:.1f} KB/s", flush=True)

        # 3. END（同一 packet）
        print("=== END -> ...0004 ===")
        await client.write_gatt_char(CHAR_FILE_END, packet, response=True)
        print("  传输完成，等待设备处理...")
        await asyncio.sleep(2.0)

        # 4. 验证图片数（设备处理期间可能短暂断开，需重连）
        if verify:
            try:
                connected = client.is_connected
            except Exception:
                connected = False
            if not connected:
                print("  连接已断，重连...")
                await cli.disconnect()
                await asyncio.sleep(1.0)
                await cli.connect()
                client = cli.client
                assert client is not None
            count_after = await cli.get_picture_count()
            print(f"  传输后图片总数 = {count_after}")
            if count_before is not None and count_after is not None:
                if count_after > count_before:
                    print(f"  [OK] 图片数 +{count_after - count_before}，设备已接受")
                    return True
                print("  [WARN] 图片数不变（同名覆盖需拉取回读比对确认）")
        return True
    finally:
        await cli.disconnect()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--kind", default="checker",
                    choices=["iris", "checker", "grid", "solid"])
    ap.add_argument("--name", default="11_20.bmp",
                    help="目标文件名（组_序号.bmp，如 11_20.bmp）")
    ap.add_argument("--dry-run", action="store_true", help="只生成本地流不发")
    ap.add_argument("--verify", action="store_true", default=True,
                    help="传输前后读图片数验证（默认开）")
    args = ap.parse_args()

    img = make_test_image(args.kind)
    print(f"=== 图像 {describe(img)} ===")
    save_bmp("captures/push_src.bmp", img)
    stream = build_stream(img)
    print(f"  流 {len(stream)}B (头62 + 数据38400)")

    if args.dry_run:
        p = build_name_packet(len(stream), args.name)
        print(f"  namePacket: {p.hex()}")
        return 0

    ok = asyncio.run(push(args.address, args.name, stream, args.verify))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
