"""Deep-probe a single BLE device: advertisement data + GATT services/characteristics.

Usage:
  python tools/bt_probe_device.py --name Iris
  python tools/bt_probe_device.py --address F4:12:FA:B6:B7:CA --connect
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from bleak import BleakClient, BleakScanner


def _adv_summary(dev, adv) -> str:
    lines = []
    name = dev.name or "(unnamed)"
    lines.append(f"  address : {dev.address}")
    lines.append(f"  name    : {name}")

    if adv is not None:
        rssi = getattr(adv, "rssi", None)
        if rssi is not None:
            lines.append(f"  rssi    : {rssi} dBm")
        tx = getattr(adv, "tx_power", None)
        if tx is not None:
            lines.append(f"  txpower : {tx} dBm")

        uuids = list(getattr(adv, "service_uuids", []) or [])
        lines.append(f"  svc uuid: {', '.join(uuids) if uuids else '(none)'}")

        mfg = {}
        try:
            mfg = dict(getattr(adv, "manufacturer_data", {}) or {})
        except Exception:
            pass
        if mfg:
            for cid, data in mfg.items():
                try:
                    txt = data.decode("ascii", errors="replace")
                except Exception:
                    txt = ""
                lines.append(f"  mfg {cid:04X} : {data.hex()}   ascii={txt!r}")

        sdata = {}
        try:
            sdata = dict(getattr(adv, "service_data", {}) or {})
        except Exception:
            pass
        if sdata:
            for su, data in sdata.items():
                lines.append(f"  svc data: {su} -> {data.hex()}")

        local = getattr(adv, "local_name", None)
        if local:
            lines.append(f"  localnm : {local}")
    else:
        lines.append("  (no advertisement data)")
    return "\n".join(lines)


async def scan_target(name: str | None, address: str | None, seconds: float):
    """Scan and keep advertisement data for the target device."""
    hits: dict[str, tuple] = {}

    def cb(device, adv):
        key = device.address.upper()
        match_name = name and name.lower() in (device.name or "").lower()
        match_addr = address and address.upper() == key
        if match_name or match_addr:
            hits[key] = (device, adv)

    kwargs = {}
    import inspect

    try:
        sig = inspect.signature(BleakScanner.__init__)
        if "detection_callback" in sig.parameters:
            kwargs["detection_callback"] = cb
    except Exception:
        pass

    if not kwargs and name:
        # Fallback: when service-uuid filtering is not usable, scan all then filter
        scanner = BleakScanner()
    else:
        scanner = BleakScanner(**kwargs) if kwargs else BleakScanner()

    await scanner.start()
    await asyncio.sleep(seconds)
    await scanner.stop()

    if not hits:
        # Fallback: find by name among already-discovered devices
        try:
            devs = scanner.get_discovered_devices()
        except Exception:
            devs = getattr(scanner, "discovered_devices", [])
        for item in devs:
            dev, adv = (item if isinstance(item, tuple) else (item, None))
            if name and name.lower() in (dev.name or "").lower():
                hits[dev.address.upper()] = (dev, adv)
            elif address and address.upper() == dev.address.upper():
                hits[dev.address.upper()] = (dev, adv)

    return hits


async def dump_gatt(address: str) -> int:
    print(f"\n=== CONNECTING {address} ===")
    try:
        client = BleakClient(address, timeout=20.0)
    except Exception as e:
        print(f"[error] failed to create client: {type(e).__name__}: {e}")
        return 1

    try:
        await client.connect()
    except Exception as e:
        print(f"[error] connect failed: {type(e).__name__}: {e}")
        print("        (you may need to pair first in Windows Settings)")
        return 1

    try:
        paired = client.is_connected
        print(f"  connected: {paired}")
        try:
            mtu = client.mtu_size
            print(f"  mtu      : {mtu}")
        except Exception:
            pass

        print("\n=== GATT SERVICES ===")
        try:
            services = client.services
        except Exception as e:
            print(f"[error] failed to enumerate services: {type(e).__name__}: {e}")
            return 1

        if not services:
            print("  (no services)")
        for svc in services:
            print(f"  [{svc.uuid}] {svc.description}")
            for ch in svc.characteristics:
                props = ",".join(ch.properties)
                print(f"      char {ch.uuid}  props={props}  {ch.description}")
                for d in ch.descriptors:
                    print(f"          desc {d.uuid}")
        return 0
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="device-name substring, case-insensitive")
    ap.add_argument("--address", default=None)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--connect", action="store_true", help="connect and enumerate GATT")
    args = ap.parse_args()

    if not args.name and not args.address:
        print("need --name or --address")
        return 1

    print(f"=== SCAN ({args.timeout}s) target: {args.name or args.address} ===")
    hits = await scan_target(args.name, args.address, args.timeout)

    if not hits:
        print("  no matching device found")
        return 1

    print(f"  matched: {len(hits)}")
    print()
    for addr, (dev, adv) in hits.items():
        print(_adv_summary(dev, adv))
        print()

    if args.connect:
        target = args.address or list(hits.keys())[0]
        return await dump_gatt(target)

    print("  (add --connect to connect and enumerate GATT services)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
