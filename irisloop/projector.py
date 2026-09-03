"""IrisGreen (AINSTEC) projector BLE device profile.

Confirmed by live scan/connect (2026-09-01).

Device identity:
    Model: IrisGreen      Manufacturer: AINSTEC
    Serial: DGIG00260101001
    Firmware: 36130326032001   HW/SW: 1.0.0
    Address: F4:12:FA:B6:B7:CA (example; actual address changes)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------- primary custom service (image / command transfer) ----------------

SVC_MAIN = "adb401c0-b1c6-11ed-afa1-0242ac120002"

# readable; observed return 01000000
CHAR_MAIN_STATUS_READ = "adb40001-b1c6-11ed-afa1-0242ac120001"

# notify + read; observed 8-byte status frame 0500010700000300
# likely device status (battery, etc.); protocol not fully confirmed
CHAR_MAIN_NOTIFY = "adb40002-b1c6-11ed-afa1-0242ac120002"

# write + indicate: command channel that requires a response (reliable)
CHAR_MAIN_CMD = "adb40003-b1c6-11ed-afa1-0242ac120003"

# write-without-response: bulk writes, suited to image streams (high throughput)
CHAR_MAIN_DATA = "adb40004-b1c6-11ed-afa1-0242ac120004"

# write: spare command channel
CHAR_MAIN_CMD2 = "adb40005-b1c6-11ed-afa1-0242ac120005"

# ---------------- secondary custom service (file transfer) ----------------

# File-transfer service adb40006-... (channel map follows iris-g-sdk IrisProtocolConfig.kt;
# protocol xlsx R13/R14 start/data descriptions are swapped vs UUID suffixes — do not map by doc order)
SVC_SECONDARY = "adb40006-b1c6-11ed-afa1-0242ac120001"

CHAR_FILE_START = "adb40006-b1c6-11ed-afa1-0242ac120003"  # write: transfer start packet
CHAR_FILE_DATA = "adb40006-b1c6-11ed-afa1-0242ac120002"   # write_no_rsp: data stream
CHAR_FILE_END = "adb40006-b1c6-11ed-afa1-0242ac120004"    # write: transfer end packet

# Legacy aliases (semantics corrected: 1=start 2=data 3=end)
CHAR_SEC_WRITE_1 = CHAR_FILE_START
CHAR_SEC_WRITE_2 = CHAR_FILE_DATA
CHAR_SEC_WRITE_3 = CHAR_FILE_END

# ---------------- standard services ----------------

CHAR_BATTERY_LEVEL = "00002a19-0000-1000-8000-00805f9b34fb"
CHAR_DEVICE_NAME = "00002a00-0000-1000-8000-00805f9b34fb"
CHAR_MODEL_NUMBER = "00002a24-0000-1000-8000-00805f9b34fb"
CHAR_SERIAL_NUMBER = "00002a25-0000-1000-8000-00805f9b34fb"
CHAR_FIRMWARE_REV = "00002a26-0000-1000-8000-00805f9b34fb"
CHAR_MANUFACTURER = "00002a29-0000-1000-8000-00805f9b34fb"

# Service UUID advertised for fast device filtering
ADV_SERVICE_UUID = SVC_MAIN

# Device name prefix (advertised local name looks like Iris-G-B6B7CA)
NAME_PREFIX = "Iris-G"

# Negotiated MTU. Observed 512, well above classic BLE 23/247,
# so the firmware is optimized for bulk image transfer.
NEGOTIATED_MTU = 512

# ATT write overhead is 3 bytes; this is the real per-packet payload cap
ATT_OVERHEAD = 3
MAX_CHUNK = NEGOTIATED_MTU - ATT_OVERHEAD  # 509


@dataclass
class ProjectorInfo:
    """Device identity read after connect."""

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
        """Raw battery value. Device returns 4 bytes (non-standard); units unconfirmed."""
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
            lines.append(f"  battery(raw) : {b}  (non-standard 4 bytes; units unconfirmed)")
        return "\n".join(lines)


def is_iris_projector(name: str | None, service_uuids: list[str] | None = None) -> bool:
    """Return True if an advertisement looks like an IrisGreen projector."""
    if name and NAME_PREFIX.lower() in name.lower():
        return True
    if service_uuids:
        return any(u.lower() == ADV_SERVICE_UUID for u in service_uuids)
    return False
