"""IrisGreen 蓝牙控制协议（V1.9，2026-03-24）。

来源：《Iris Green蓝牙通信协议_20260324_V1.9.xlsx》

帧格式（写 adb40003 / 数据流 adb40004）：
    type(1) + len(1) + data(n)
响应帧：
    80 + type(1) + len(1) + data(n)

注意：文档 R2 描述响应前缀为 "80"，而各命令示例里写作 "08"。
以示例字节流为准（`80 11 01 64`），响应首字节取 0x80。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable

# ---------------- 命令类型 ----------------

CMD_SN = 0x01                # 读:0  写:15
CMD_MODEL = 0x02             # 读:0  写:9
CMD_DEVICE_NAME = 0x03       # 读:0  写:<=20 字节（UTF-8，中文<=6 字）
CMD_RESOURCE_VER = 0x04      # 读:0
CMD_PIC_COUNT = 0x05         # 读:0  -> 2 字节图片数
CMD_FORMAT = 0x06            # 写:10  "Green@2026"
CMD_SHUTDOWN = 0x07          # 写:1
CMD_REBOOT = 0x08            # 写:1

CMD_TIME = 0x10              # 写:4  本地时间戳（小端 u32）
CMD_BRIGHTNESS = 0x11        # 读:0  写:1   0~100
CMD_FLIP = 0x12              # 读:0  写:1   0=正装 1=倒装
CMD_CHEER_MODE = 0x13        # 读:0  写:1
CMD_HOURLY_CHIME = 0x14      # 读:0  写:1
CMD_MILEAGE_ANNOUNCE = 0x15  # 读:0  写:1
CMD_MILEAGE_INTERVAL = 0x16  # 读:0  写:1   单位 100m
CMD_DRINK_REMIND = 0x17      # 读:0  写:1
CMD_DRINK_INTERVAL = 0x18    # 读:0  写:1   单位 分钟
CMD_LAYOUT = 0x19            # 写:1   0=极简 1=标准
CMD_SPEED_UNIT = 0x1A        # 写:1   0=KMH 1=MPH
CMD_LOW_BATTERY = 0x1B       # 写:1   百分比

CMD_KEYSTONE_MODE = 0x1C     # 写:1   0=不修正 1=左直角梯形 2=等腰梯形 3=右直角梯形
CMD_KEYSTONE_ANGLE = 0x1D    # 写:1   -70~70；>70 表示负数
CMD_MIRROR = 0x1E            # 写:1   0/1
CMD_FOV = 0x1F               # 写:1   0=10*5 1=30*10 2=20*10

CMD_PLAY = 0xA0              # 写:6  通用整图素材播放
CMD_PLAY_STOP = 0xA1         # 写:1
CMD_DELETE_PIC = 0xA2        # 写:1  删除指定组素材
CMD_MATERIAL_PARAM = 0xA3    # 读:0  写:2
CMD_WEATHER_TIME = 0xA4      # 写:7
CMD_BG_PLAY = 0xA5           # 写:4  背景素材（计时器/整点报时）
CMD_NOTICE = 0xA6            # 写:<=32
CMD_PHONE_ALERT = 0xA7       # 写:<=32

RESP_PREFIX = 0x80

# ---------------- 素材组 ID ----------------

MATERIAL_GROUPS = {
    1: "开机",
    2: "充电",
    3: "未连接",
    4: "连接成功",
    5: "低电量",
    0x0A: "网格",
    0x0B: "动画",
    0x0C: "手写",
    0x0D: "提示词",
    0x0E: "速度达成30KM",
    0x0F: "里程素材",
    0x10: "喝水素材",
    0x11: "速度达成40KM",
    0x12: "光轮轨迹",
    0x13: "爬坡素材",
    0x14: "下坡素材",
    0x15: "围栏素材",
    0x16: "停车素材",
    0x17: "尾流素材",
    0x18: "电话提醒",
    # 以下为背景图，编号从 51 开始
    0x33: "电量",
    0x34: "天气/日历_1",
    0x35: "天气/日历_2",
    0x36: "天气/日历_3",
    0x37: "计时器素材",
    0x38: "极简布局_速度",
    0x39: "极简布局_平均速度",
    0x3A: "极简布局_里程",
    0x3B: "极简布局_时间",
    0x3C: "极简布局_海拔",
    0x3D: "极简布局_卡路里",
    0x3E: "标准布局_1",
    0x3F: "标准布局_2",
    0x40: "标准布局_3",
    0x41: "导航_左转",
    0x42: "导航_右转",
    0x43: "导航_直行",
    0x44: "导航_掉头",
    0x45: "整点报时",
}

MAX_MATERIAL_TOTAL = 200


def group_name(gid: int) -> str:
    return MATERIAL_GROUPS.get(gid, f"unknown(0x{gid:02X})")


# ---------------- 帧构造 ----------------


def build_write(cmd: int, data: bytes | Iterable[int] = b"") -> bytes:
    """构造写帧: type + len + data。"""
    d = bytes(data)
    if len(d) > 255:
        raise ValueError(f"data 过长: {len(d)} > 255")
    return bytes([cmd & 0xFF, len(d)]) + d


def build_read(cmd: int) -> bytes:
    """构造读帧: type + 0x00（len=0 表示读）。"""
    return bytes([cmd & 0xFF, 0x00])


@dataclass
class Response:
    ok: bool
    cmd: int = 0
    data: bytes = b""
    raw: bytes = b""
    error: str = ""

    @property
    def text_ascii(self) -> str:
        """SN/Model 等字段按小端反转后转 ASCII（协议规定）。"""
        return bytes(self.data)[::-1].decode("ascii", errors="replace")

    @property
    def u16_le(self) -> int | None:
        if len(self.data) >= 2:
            return struct.unpack("<H", self.data[:2])[0]
        return None

    @property
    def u32_le(self) -> int | None:
        if len(self.data) >= 4:
            return struct.unpack("<I", self.data[:4])[0]
        return None


def is_status_frame(raw: bytes | bytearray) -> bool:
    """判断是否为设备自发的状态上报帧（adb40002）。

    实机观测：固定 8 字节，形如 00 00 01 22 00 00 0f 00，
    第 4 字节为递增计数器（0x22 -> 0x23）。它不是命令响应。
    """
    b = bytes(raw)
    if len(b) != 8:
        return False
    # 状态帧前 3 字节固定为 00 00 01，末字节 00
    return b[0] == 0x00 and b[1] == 0x00 and b[2] == 0x01 and b[7] == 0x00


def parse_response(raw: bytes | bytearray) -> Response:
    """解析响应帧。

    文档描述响应前缀为 0x80，但实机 adb40002 通道观测到的是
    8 字节自发状态帧（非命令响应）。因此：
      - 状态帧 -> 标记为非命令响应
      - 0x80/0x08 前缀 -> 按文档解析
      - 其余 -> 原样返回，交由调用方判断
    """
    b = bytes(raw)
    if len(b) < 3:
        return Response(ok=False, raw=b, error=f"响应过短: {b.hex()}")

    if is_status_frame(b):
        return Response(ok=False, raw=b, error="状态上报帧(非命令响应)")

    if b[0] in (0x80, 0x08):
        cmd = b[1]
        length = b[2]
        data = b[3 : 3 + length]
        return Response(ok=True, cmd=cmd, data=data, raw=b)

    return Response(ok=False, raw=b, error=f"未知帧格式, 首字节=0x{b[0]:02X}")


# ---------------- 常用命令快捷构造 ----------------


def cmd_set_keystone(mode: int) -> bytes:
    """畸变校正模式: 0=不修正 1=左直角梯形 2=等腰梯形 3=右直角梯形。"""
    if mode not in (0, 1, 2, 3):
        raise ValueError("mode 必须是 0..3")
    return build_write(CMD_KEYSTONE_MODE, [mode])


def cmd_set_keystone_angle(angle: int) -> bytes:
    """畸变校正角度 -70~70。负值按 >70 编码（文档规则）。

    例: -10 度 -> 80；-70 度 -> 140
    """
    if not -70 <= angle <= 70:
        raise ValueError("angle 必须在 -70..70")
    return build_write(CMD_KEYSTONE_ANGLE, [angle if angle >= 0 else 70 + abs(angle)])


def cmd_set_fov(level: int) -> bytes:
    """幅面档位: 0=10*5 1=30*10 2=20*10。"""
    if level not in (0, 1, 2):
        raise ValueError("level 必须是 0..2")
    return build_write(CMD_FOV, [level])


def cmd_set_brightness(percent: int) -> bytes:
    if not 0 <= percent <= 100:
        raise ValueError("brightness 必须在 0..100")
    return build_write(CMD_BRIGHTNESS, [percent])


def cmd_set_mirror(enable: bool) -> bytes:
    return build_write(CMD_MIRROR, [1 if enable else 0])


def cmd_set_flip(upside_down: bool) -> bytes:
    return build_write(CMD_FLIP, [1 if upside_down else 0])


def cmd_play(
    group_id: int,
    loop: bool = False,
    total_100ms: int = 100,
    interval_100ms: int = 2,
) -> bytes:
    """通用整图素材播放（0xA0，6 字节）。

    文档示例: A0 06 00 64 00 02 00 0B
      loop, total(u16 le), interval(u16 le), group_id
    """
    if not 0 <= group_id <= 0xFF:
        raise ValueError("group_id 超出范围")
    if total_100ms > 36000:
        raise ValueError("总播放时长上限 1 小时（36000 * 100ms）")
    data = (
        bytes([1 if loop else 0])
        + struct.pack("<H", total_100ms)
        + struct.pack("<H", interval_100ms)
        + bytes([group_id])
    )
    return build_write(CMD_PLAY, data)


def cmd_play_stop() -> bytes:
    return build_write(CMD_PLAY_STOP, [1])


def cmd_delete_group(group_id: int) -> bytes:
    return build_write(CMD_DELETE_PIC, [group_id & 0xFF])


def cmd_set_time(timestamp: int) -> bytes:
    return build_write(CMD_TIME, struct.pack("<I", timestamp & 0xFFFFFFFF))


def describe(frame: bytes) -> str:
    """把命令帧转成人类可读说明，便于日志。"""
    if not frame:
        return "(empty)"
    cmd, length = frame[0], frame[1] if len(frame) > 1 else 0
    data = frame[2:]
    names = {
        CMD_KEYSTONE_MODE: "畸变校正模式",
        CMD_KEYSTONE_ANGLE: "畸变校正角度",
        CMD_FOV: "幅面设置",
        CMD_BRIGHTNESS: "亮度",
        CMD_MIRROR: "左右镜像",
        CMD_FLIP: "翻转",
        CMD_PLAY: "播放素材",
        CMD_PLAY_STOP: "停止播放",
        CMD_PIC_COUNT: "读图片数",
        CMD_DELETE_PIC: "删除素材组",
        CMD_TIME: "设置时间",
    }
    name = names.get(cmd, "")
    return f"cmd=0x{cmd:02X}{name and '(' + name + ')'} len={length} data={data.hex()}"
