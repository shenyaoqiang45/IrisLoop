"""BLE scan (bleak, standalone WinRT backend, no PowerShell event pump)."""

from __future__ import annotations

import asyncio
import sys

try:
    from importlib.metadata import version as _v
except Exception:
    _v = None

import bleak
from bleak import BleakScanner

try:
    from bleak.backends.device import BLEDevice
    from bleak.backends.scanner import AdvertisementData
except Exception:
    BLEDevice = AdvertisementData = None


def _ver() -> str:
    if _v:
        try:
            return _v("bleak")
        except Exception:
            pass
    return "unknown"


async def scan(seconds: float, service_uuid: str | None = None):
    kwargs = {}
    # bleak>=0.20 supports return_adv; older versions do not
    import inspect

    sig = inspect.signature(BleakScanner.discover)
    if "return_adv" in sig.parameters:
        kwargs["return_adv"] = True

    if service_uuid:
        scanner = BleakScanner(service_uuids=[service_uuid])
    else:
        scanner = BleakScanner()

    await scanner.start()
    await asyncio.sleep(seconds)
    await scanner.stop()

    # bleak>=0.19: get_discovered_devices() returns (device, adv) tuples
    if hasattr(scanner, "get_discovered_devices"):
        try:
            return scanner.get_discovered_devices()
        except TypeError:
            pass
    return scanner.discovered_devices


def _fmt(dev, adv) -> str:
    name = dev.name or "(unnamed)"
    addr = dev.address
    rssi = getattr(adv, "rssi", None) if adv else None
    rssi_s = f"{rssi}" if rssi is not None else "?"

    uuids = []
    if adv is not None:
        uuids = list(getattr(adv, "service_uuids", []) or [])

    # Manufacturer data in the advertisement often identifies the chip family
    mfg = {}
    if adv is not None:
        try:
            mfg = dict(getattr(adv, "manufacturer_data", {}) or {})
        except Exception:
            mfg = {}

    mfg_s = ", ".join(f"{k:04X}:{v.hex()[:16]}" for k, v in list(mfg.items())[:3])
    uuid_s = ", ".join(uuids[:6]) if uuids else "-"

    lines = [
        f"  {addr}  {name}",
        f"      rssi={rssi_s}  uuids={uuid_s}",
    ]
    if mfg_s:
        lines.append(f"      mfg={mfg_s}")
    return "\n".join(lines)


async def main(seconds: float, service_uuid: str | None) -> int:
    print(f"bleak {_ver()}")
    print(f"=== BLE SCAN ({seconds}s) ===")
    try:
        result = await scan(seconds, service_uuid)
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
        return 1

    if not result:
        print("  (no BLE devices found)")
        return 0

    # Compatible with both return shapes: [(dev, adv)] or [dev]
    rows = []
    for item in result:
        if isinstance(item, tuple) and len(item) == 2:
            rows.append((item[0], item[1]))
        else:
            rows.append((item, None))

    rows.sort(key=lambda r: (r[0].name or "").lower())
    print(f"  found: {len(rows)}")
    print()
    for dev, adv in rows:
        print(_fmt(dev, adv))
    return 0


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    uuid = sys.argv[2] if len(sys.argv) > 2 else None
    raise SystemExit(asyncio.run(main(secs, uuid)))
