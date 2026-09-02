"""IrisGreen (AINSTEC) 投影仪 BLE 设备档案。

通过实机扫描/连接确认（2026-09-01）。

设备身份:
    Model: IrisGreen      Manufacturer: AINSTEC
    Serial: DGIG00260101001
    Firmware: 36130326032001   HW/SW: 1.0.0
    Address: F4:12:FA:B6:B7:CA (示例，实际地址会变)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------- 主自定义服务（图片/指令传输） ----------------

SVC_MAIN = "adb401c0-b1c6-11ed-afa1-0242ac120002"

# 可读，实测返回 01000000
CHAR_MAIN_STATUS_READ = "adb40001-b1c6-11ed-afa1-0242ac120001"

# notify + read，实测返回 8 字节状态帧 0500010700000300
# 推测为设备状态上报（含电量等），待协议确认
CHAR_MAIN_NOTIFY = "adb40002-b1c6-11ed-afa1-0242ac120002"

# write + indicate：需要应答的指令通道（可靠传输）
CHAR_MAIN_CMD = "adb40003-b1c6-11ed-afa1-0242ac120003"

# write-without-response：无应答批量写入，适合图片数据流（高吞吐）
CHAR_MAIN_DATA = "adb40004-b1c6-11ed-afa1-0242ac120004"

# write：备用指令通道
CHAR_MAIN_CMD2 = "adb40005-b1c6-11ed-afa1-0242ac120005"

# ---------------- 次自定义服务（文件传输） ----------------

# 文件传输服务 adb40006-...（通道映射以 iris-g-sdk IrisProtocolConfig.kt 为准，
# 协议 xlsx R13/R14 的「开始/数据」描述与 UUID 尾号是反的，勿按文档顺序映射）
SVC_SECONDARY = "adb40006-b1c6-11ed-afa1-0242ac120001"

CHAR_FILE_START = "adb40006-b1c6-11ed-afa1-0242ac120003"  # write: 传输开始包
CHAR_FILE_DATA = "adb40006-b1c6-11ed-afa1-0242ac120002"   # write_no_rsp: 数据流
CHAR_FILE_END = "adb40006-b1c6-11ed-afa1-0242ac120004"    # write: 传输结束包

# 兼容旧名（语义已纠正：1=开始 2=数据 3=结束）
CHAR_SEC_WRITE_1 = CHAR_FILE_START
CHAR_SEC_WRITE_2 = CHAR_FILE_DATA
CHAR_SEC_WRITE_3 = CHAR_FILE_END

# ---------------- 标准服务 ----------------

CHAR_BATTERY_LEVEL = "00002a19-0000-1000-8000-00805f9b34fb"
CHAR_DEVICE_NAME = "00002a00-0000-1000-8000-00805f9b34fb"
CHAR_MODEL_NUMBER = "00002a24-0000-1000-8000-00805f9b34fb"
CHAR_SERIAL_NUMBER = "00002a25-0000-1000-8000-00805f9b34fb"
CHAR_FIRMWARE_REV = "00002a26-0000-1000-8000-00805f9b34fb"
CHAR_MANUFACTURER = "00002a29-0000-1000-8000-00805f9b34fb"

# 广播里声明的服务 UUID，用于快速筛选设备
ADV_SERVICE_UUID = SVC_MAIN

# 设备名前缀（广播 local name 形如 Iris-G-B6B7CA）
NAME_PREFIX = "Iris-G"

# 协商后的 MTU。实测 512，远超标准 BLE 的 23/247，
# 说明固件为批量图片传输做了优化。
NEGOTIATED_MTU = 512

# ATT 写入开销 3 字节，实际单包载荷上限
ATT_OVERHEAD = 3
MAX_CHUNK = NEGOTIATED_MTU - ATT_OVERHEAD  # 509


@dataclass
class ProjectorInfo:
    """连接后读取到的设备身份。"""

    address: str = ""
    name: str = ""
    model: str = ""
    serial: str = ""
    firmware: str = ""
    hardware_rev: str = ""
    software_rev: str = ""
    manufacturer: str = ""
    battery_raw: bytes = b""
    mtu: int = 0
    rssi: int | None = None
    status_frames: list[bytes] = field(default_factory=list)

    @property
    def battery_u32(self) -> int | None:
        """电量原始值。设备返回 4 字节（非标准），量纲待确认。"""
        if len(self.battery_raw) >= 4:
            return int.from_bytes(self.battery_raw[:4], "little")
        if self.battery_raw:
            return self.battery_raw[0]
        return None

    def summary(self) -> str:
        lines = [
            f"  address      : {self.address}",
            f"  name         : {self.name}",
            f"  model        : {self.model}",
            f"  manufacturer : {self.manufacturer}",
            f"  serial       : {self.serial}",
            f"  firmware     : {self.firmware}",
            f"  hw / sw      : {self.hardware_rev} / {self.software_rev}",
            f"  mtu          : {self.mtu}",
        ]
        if self.rssi is not None:
            lines.append(f"  rssi         : {self.rssi} dBm")
        b = self.battery_u32
        if b is not None:
            lines.append(f"  battery(raw) : {b}  (非标准4字节，量纲待确认)")
        return "\n".join(lines)


def is_iris_projector(name: str | None, service_uuids: list[str] | None = None) -> bool:
    """判断广播是否为 IrisGreen 投影仪。"""
    if name and NAME_PREFIX.lower() in name.lower():
        return True
    if service_uuids:
        return any(u.lower() == ADV_SERVICE_UUID for u in service_uuids)
    return False
