"""BLE 文件传输服务推图探针 —— 确定开始/结束帧格式。

已知:
  - 协议文档只给了通道用途，没给开始/结束帧格式
  - 固件升级示例帧: 00 0F 45 10 + "xingzhe_ctrl_mems.bin"
    其中 0x000F4510 = 1002768 ≈ 1MB，疑似「文件大小(u32 BE) + 文件名」
  - 设备内图片均为 38462 字节完整 BMP（含 62 字节头）
  - ⚠️ 旧 push_image.py 把数据写到了 0004(结束) 通道、结束写到了 0003(数据) 通道

判定手段:
  - cmd 0x05 读图片总数（推送到新槽位会增加）
  - adb40002 状态帧 / adb40003 indicate 回执的异常变化

用法:
  python tools/push_probe.py                 # 全阶段: 基线 -> START 候选 -> 完整传输
  python tools/push_probe.py --stage base    # 只测基线
  python tools/push_probe.py --stage start   # 只试 START 候选
  python tools/push_probe.py --stage full --start-fmt size_be_name --end-fmt one
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from irisloop import protocol as P
from irisloop.image_pack import build_stream, describe, make_test_image, save_bmp
from irisloop.projector import (
    CHAR_MAIN_CMD,
    CHAR_MAIN_NOTIFY,
    CHAR_SEC_WRITE_1,  # adb40006-...-0003 数据
    CHAR_SEC_WRITE_2,  # adb40006-...-0002 开始
    CHAR_SEC_WRITE_3,  # adb40006-...-0004 结束
)

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"

# 设备自身生成的 BMP 头（test-data/1_1.bmp 前 62 字节），与 image_pack.BMP_HEAD
# 仅差 xPelsPerMeter/yPelsPerMeter(0 vs 2835) 和 clrUsed(2 vs 0)。
# 推送时用设备同款头，排除固件头部校验差异。
DEVICE_BMP_HEAD = bytes.fromhex(
    "424D3E960000000000003E0000002800000080020000E00100000100010000"
    "00000000960000000000000000000002000000000000000000000000FFFFFF00"
)

CH_START = CHAR_SEC_WRITE_2   # ...0002 开始
CH_DATA = CHAR_SEC_WRITE_1    # ...0003 数据
CH_END = CHAR_SEC_WRITE_3     # ...0004 结束


def build_start(fmt: str, name: str, size: int) -> bytes:
    n = name.encode("ascii")
    if fmt == "size_be_name":
        return size.to_bytes(4, "big") + n
    if fmt == "size_le_name":
        return size.to_bytes(4, "little") + n
    if fmt == "name":
        return n
    if fmt == "type_name":
        return b"\x01" + n
    if fmt == "type_size_le_name":
        return b"\x01" + size.to_bytes(4, "little") + n
    if fmt == "one":
        return b"\x01"
    raise ValueError(fmt)


def build_end(fmt: str, size: int) -> bytes | None:
    if fmt == "one":
        return b"\x01"
    if fmt == "size_le":
        return size.to_bytes(4, "little")
    if fmt == "none":
        return None
    raise ValueError(fmt)


class Probe:
    def __init__(self, addr: str):
        self.addr = addr
        self.client = None
        self.frames: list[tuple[float, str, bytes]] = []
        self.t0 = time.perf_counter()

    def _log(self, ch: str, data: bytearray):
        b = bytes(data)
        tag = "STATUS" if P.is_status_frame(b) else "OTHER "
        self.frames.append((time.perf_counter() - self.t0, f"{ch}:{tag}", b))

    async def connect(self):
        from bleak import BleakClient

        self.client = BleakClient(self.addr, timeout=20.0)
        await self.client.connect()
        print(f"connected  mtu={self.client.mtu_size}")
        await self.client.start_notify(CHAR_MAIN_NOTIFY,
                                       lambda s, d: self._log("40002", d))
        try:
            await self.client.start_notify(CHAR_MAIN_CMD,
                                           lambda s, d: self._log("40003", d))
        except Exception as e:
            print(f"  (adb40003 indicate 订阅失败: {e})")

    async def disconnect(self):
        if self.client:
            try:
                await self.client.disconnect()
            except Exception:
                pass
            self.client = None

    async def pic_count(self) -> int | None:
        """经 adb40003 读图片总数。"""
        got: list[bytes] = []
        assert self.client is not None

        def _cb(_s, d: bytearray):
            got.append(bytes(d))

        try:
            await self.client.start_notify(CHAR_MAIN_CMD, _cb)
        except Exception:
            pass
        await self.client.write_gatt_char(CHAR_MAIN_CMD,
                                          P.build_read(P.CMD_PIC_COUNT),
                                          response=True)
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            for raw in got:
                r = P.parse_response(raw)
                if r.ok and r.cmd == P.CMD_PIC_COUNT:
                    return r.u16_le
            await asyncio.sleep(0.05)
        return None

    def new_frames_since(self, mark: int) -> list[tuple[float, str, bytes]]:
        return self.frames[mark:]

    def dump_since(self, mark: int, limit: int = 16):
        fresh = self.frames[mark:]
        if not fresh:
            print("    (无新帧)")
            return
        for ts, ch, b in fresh[:limit]:
            print(f"    t={ts:6.2f}s {ch} {b.hex()}")
        if len(fresh) > limit:
            print(f"    ... 共 {len(fresh)} 帧")


async def stage_base(p: Probe) -> None:
    print("\n--- 基线: 图片总数 + 通道活动 ---")
    n = await p.pic_count()
    print(f"  图片总数 = {n}")
    mark = len(p.frames)
    await asyncio.sleep(2.0)
    p.dump_since(mark)


async def stage_start(p: Probe, name: str, size: int, fmts: list[str]) -> None:
    print("\n--- START 帧候选（只发开始帧，观察 2s）---")
    for fmt in fmts:
        payload = build_start(fmt, name, size)
        mark = len(p.frames)
        print(f"  [{fmt}] -> {payload.hex()} ({len(payload)}B)")
        try:
            await p.client.write_gatt_char(CH_START, payload, response=True)
            print("    write ok")
        except Exception as e:
            print(f"    write FAIL: {e}")
            continue
        await asyncio.sleep(2.0)
        p.dump_since(mark)


async def stage_full(p: Probe, name: str, stream: bytes,
                     start_fmt: str, end_fmt: str,
                     chunk: int, gap: float) -> None:
    print(f"\n--- 完整传输 {name} ({len(stream)}B) "
          f"start={start_fmt} end={end_fmt} chunk={chunk} ---")

    before = await p.pic_count()
    print(f"  传输前图片总数 = {before}")

    start_payload = build_start(start_fmt, name, len(stream))
    mark = len(p.frames)
    print(f"  START -> {start_payload.hex()}")
    await p.client.write_gatt_char(CH_START, start_payload, response=True)
    await asyncio.sleep(0.5)

    n = (len(stream) + chunk - 1) // chunk
    print(f"  DATA {len(stream)}B / {chunk} = {n} 包 -> adb40006-0003")
    sent = 0
    t = time.perf_counter()
    for i in range(n):
        part = stream[i * chunk:(i + 1) * chunk]
        await p.client.write_gatt_char(CH_DATA, part, response=False)
        sent += len(part)
        if gap:
            await asyncio.sleep(gap)
        if (i + 1) % 50 == 0 or i == n - 1:
            el = time.perf_counter() - t
            print(f"    {i+1}/{n}  {sent}B  {sent/el/1024:.1f} KB/s", flush=True)

    end_payload = build_end(end_fmt, len(stream))
    if end_payload is not None:
        print(f"  END -> {end_payload.hex()}")
        await p.client.write_gatt_char(CH_END, end_payload, response=True)
    else:
        print("  END (不发)")

    await asyncio.sleep(2.0)
    print("  传输期间新帧:")
    p.dump_since(mark, limit=24)

    # 设备常在收到完整文件后主动断开处理，需重连再读
    await asyncio.sleep(1.0)
    print("  传输后强制重连再读图片数...")
    await p.disconnect()
    await asyncio.sleep(1.0)
    await p.connect()
    after = await p.pic_count()
    print(f"  传输后图片总数 = {after}")
    if before is not None and after is not None:
        if after > before:
            print(f"  ✅ 图片数 +{after-before}，设备接受了文件！")
        elif after == before:
            print("  ⚠️ 图片数不变：可能是覆盖了同名文件（需回读比对确认），或被拒绝")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--stage", default="all",
                    choices=["base", "start", "full", "all"])
    ap.add_argument("--name", default="11_20.bmp",
                    help="目标文件名（默认 11_20.bmp：动画组末尾槽位，大概率是新增）")
    ap.add_argument("--kind", default="checker")
    ap.add_argument("--start-fmt", default="size_be_name",
                    choices=["size_be_name", "size_le_name", "name",
                             "type_name", "type_size_le_name", "one"])
    ap.add_argument("--end-fmt", default="one",
                    choices=["one", "size_le", "none"])
    ap.add_argument("--chunk", type=int, default=244)
    ap.add_argument("--gap", type=float, default=0.0)
    ap.add_argument("--device-head", action="store_true", default=True,
                    help="用设备同款 62B 头（默认开）")
    args = ap.parse_args()

    img = make_test_image(args.kind)
    stream = build_stream(img)
    if args.device_head:
        stream = DEVICE_BMP_HEAD + stream[62:]
    print(f"=== 图像 {describe(img)}  流 {len(stream)}B ===")
    save_bmp("captures/push_probe_src.bmp", img)

    p = Probe(args.address)
    print(f"=== CONNECT {args.address} ===")
    await p.connect()
    try:
        if args.stage in ("base", "all"):
            await stage_base(p)
        if args.stage in ("start", "all"):
            await stage_start(p, args.name, len(stream),
                              ["size_be_name", "size_le_name", "name",
                               "type_name", "type_size_le_name", "one"])
        if args.stage in ("full", "all"):
            await stage_full(p, args.name, stream,
                             args.start_fmt, args.end_fmt,
                             args.chunk, args.gap)
    finally:
        await p.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
