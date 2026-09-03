"""Pull on-device IrisGreen assets to local data/.

There is no BLE read-image command; the protocol switches to a WiFi AP then HTTP:
    Open WiFi:  write 0x02 to adb40005
    Upload/fetch:  http://192.168.4.1/upload/<name>.bmp
    Switch back to BLE:  http://192.168.4.1/switch/ble

Flow:
    1. BLE scan and connect, read picture count
    2. Open device WiFi
    3. Join the AP from this PC's WLAN
    4. Probe the directory, then GET each protocol filename
    5. Switch back to BLE
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from bleak import BleakClient, BleakScanner

from irisloop import protocol as P
from irisloop.ble_client import IrisBleClient
from irisloop.image_pack import HEAD_BYTES, STREAM_BYTES
from irisloop.projector import CHAR_MAIN_CMD2, NAME_PREFIX, SVC_MAIN

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"
DEFAULT_HOST = "192.168.4.1"
OUT_DIR = "data"

# Max frames per group from protocol Sheet2 (decimal group IDs)
GROUP_MAX = {
    1: 10, 2: 3, 3: 1, 4: 1, 5: 2,
    10: 1, 11: 20, 12: 5, 13: 10,
    14: 10, 15: 10, 16: 10, 17: 10, 18: 10, 19: 10, 20: 10,
    21: 10, 22: 10, 23: 10, 24: 10,
}
for _g in range(51, 70):
    GROUP_MAX[_g] = 1

def http_get_retry(url: str, timeout: float = 10.0, tries: int = 5) -> tuple[int, bytes, str]:
    last = (-1, b"", "")
    for i in range(tries):
        code, body, ctype = http_get(url, timeout=timeout)
        last = (code, body, ctype)
        if code == 200 and body:
            return last
        if code == 404:
            return last
        time.sleep(0.4 + 0.3 * i)
    return last


async def find_device(timeout: float = 12.0, address: str | None = None):
    """Return (address, BLEDevice|None). Prefer the scanned device object so WinRT can connect immediately."""
    found = await BleakScanner.discover(timeout=timeout, return_adv=True)
    if address:
        for addr, (dev, adv) in found.items():
            if addr.upper() == address.upper():
                return addr, dev
    for addr, (dev, adv) in found.items():
        name = (dev.name or getattr(adv, "local_name", None) or "")
        uuids = [u.lower() for u in (getattr(adv, "service_uuids", None) or [])]
        if NAME_PREFIX.lower() in name.lower() or SVC_MAIN in uuids:
            return addr, dev
    if address:
        return address, None
    raise RuntimeError("IrisGreen not found in scan")


def build_wifi_cred(ssid: str, password: str) -> bytes:
    """Build adb40005 command 01: configure WiFi SSID/password (98 bytes total)."""
    sb = ssid.encode("utf-8")
    pb = password.encode("utf-8")
    if len(sb) > 32 or len(pb) > 64:
        raise ValueError("SSID/password too long")
    name = sb.ljust(32, b"\x00")
    pwd = pb.ljust(64, b"\x00")
    return bytes([0x01, len(sb)]) + name + pwd


async def ble_prepare(
    address: str,
    ssid: str = "",
    password: str = "AinstecIris123456789",
    device=None,
) -> tuple[int, str, str]:
    """Connect, read picture count, configure and open WiFi. Return (picture count, ssid, password)."""
    suffix = address.replace(":", "")[-6:].upper()
    ssid = ssid or f"Iris-G-WLAN-{suffix}"
    print(f"=== BLE CONNECT {address} ===")
    client = BleakClient(device if device is not None else address, timeout=20.0)
    await client.connect()
    print(f"  connected  mtu={client.mtu_size}")
    cli = IrisBleClient(address)
    cli.client = client
    count = -1
    try:
        n = await cli.get_picture_count()
        if n is not None:
            count = n
            print(f"  picture count {count}")
        else:
            print("  failed to read picture count")
        cred = build_wifi_cred(ssid, password)
        print(f"=== SET WIFI ssid={ssid} pass={password} ===")
        await client.write_gatt_char(CHAR_MAIN_CMD2, cred, response=True)
        await asyncio.sleep(0.5)
        print("=== OPEN WIFI (adb40005 <- 02) ===")
        await client.write_gatt_char(CHAR_MAIN_CMD2, b"\x02", response=True)
        print("  sent open-WiFi")
        await asyncio.sleep(1.0)
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass
    return count, ssid, password


def run(cmd: list[str], timeout: int = 30) -> str:
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    out = (r.stdout or b"") + (r.stderr or b"")
    return out.decode("utf-8", errors="replace") or out.decode("gbk", errors="replace")


def log(msg: str) -> None:
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()


def scan_ssid() -> list[str]:
    out = run(["netsh", "wlan", "show", "networks", "mode=bssid"])
    names = re.findall(r"SSID\s+\d+\s*:\s*(.+)", out)
    return [n.strip() for n in names if n.strip()]


def pick_ssid(ssids: list[str], hint: str) -> str | None:
    hint_u = (hint or "").upper().replace(":", "")[-6:]
    for s in ssids:
        u = s.upper().replace(":", "")
        if hint_u and hint_u in u:
            return s
    for s in ssids:
        if "IRIS-G-WLAN" in s.upper():
            return s
    return None


def write_wlan_profile(ssid: str, password: str, path: str) -> None:
    # Device samples use SSID=password; if the network is open, still try WPA2 first
    xml = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{ssid}</name>
    <SSIDConfig>
        <SSID><name>{ssid}</name></SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>manual</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>WPA2PSK</authentication>
                <encryption>AES</encryption>
                <useOneX>false</useOneX>
            </authEncryption>
            <sharedKey>
                <keyType>passPhrase</keyType>
                <protected>false</protected>
                <keyMaterial>{password}</keyMaterial>
            </sharedKey>
        </security>
    </MSM>
</WLANProfile>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)


def wifi_connected_to(ssid: str) -> bool:
    out = run(["netsh", "wlan", "show", "interfaces"])
    # Association success prints SSID/BSSID; disconnect has neither line.
    # Do not depend on the Chinese "connected" string (console encoding garbles it).
    has_ssid = re.search(rf"SSID\s*:\s*{re.escape(ssid)}\s*$", out, re.I | re.M)
    has_bssid = re.search(r"BSSID\s*:\s*([0-9a-f:]{17})", out, re.I)
    return bool(has_ssid and has_bssid)


def connect_wifi(ssid: str, password: str) -> bool:
    if wifi_connected_to(ssid):
        log(f"  WLAN already on {ssid}")
        return True
    profile = os.path.join(os.environ.get("TEMP", "."), "iris_wlan.xml")
    run(["netsh", "wlan", "delete", "profile", f"name={ssid}"])
    write_wlan_profile(ssid, password, profile)
    log(run(["netsh", "wlan", "add", "profile", f"filename={profile}", "user=current"]))
    log(run(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}", "interface=WLAN"]))
    deadline = time.time() + 30
    while time.time() < deadline:
        if wifi_connected_to(ssid):
            log(f"  WLAN connected {ssid}")
            # Wait for DHCP
            for _ in range(15):
                ip = run(["netsh", "interface", "ip", "show", "addresses", "WLAN"])
                if "192.168.4." in ip:
                    log("  got 192.168.4.x")
                    return True
                time.sleep(1.0)
            log("  associated but no 192.168.4.x yet")
            return True
        time.sleep(1.0)
    log("  connect wait timed out")
    log(run(["netsh", "wlan", "show", "interfaces"]))
    return False


def _wlan_local_ip() -> str | None:
    """Local address while on the device hotspot (usually 192.168.4.2)."""
    out = run(["netsh", "interface", "ip", "show", "addresses", "WLAN"])
    m = re.search(r"192\.168\.4\.\d+", out)
    return m.group(0) if m else None


def http_get(url: str, timeout: float = 8.0) -> tuple[int, bytes, str]:
    """Talk to the device AP directly: bind the WLAN address and disable system proxy (avoids Meta/Clash TUN hijack)."""
    import http.client
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or DEFAULT_HOST
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    class BoundHTTPConnection(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.create_connection(
                (self.host, self.port),
                self.timeout,
                source_address=(src, 0) if src else None,
            )

    src = _wlan_local_ip()
    # Env proxies can hijack too; clear them temporarily
    old_env = {k: os.environ.pop(k) for k in list(os.environ)
               if k.lower() in ("http_proxy", "https_proxy", "all_proxy", "http_proxy", "https_proxy")}
    try:
        conn = BoundHTTPConnection(host, port, timeout=timeout)
        conn.request("GET", path, headers={"Connection": "close", "User-Agent": "IrisLoop/pull"})
        resp = conn.getresponse()
        body = resp.read()
        ctype = resp.getheader("Content-Type", "") or ""
        code = resp.status
        conn.close()
        return code, body, ctype
    except Exception as e:
        return -1, str(e).encode("utf-8", errors="replace"), ""
    finally:
        os.environ.update(old_env)


def looks_like_bmp(data: bytes) -> bool:
    if len(data) < 2 or data[:2] != b"BM":
        return False
    if len(data) in (STREAM_BYTES, HEAD_BYTES + 38400):
        return True
    return len(data) >= HEAD_BYTES


def save_file(out_dir: str, name: str, data: bytes) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    path = os.path.join(out_dir, safe)
    with open(path, "wb") as f:
        f.write(data)
    return path


def parse_names(html: bytes) -> list[str]:
    text = html.decode("utf-8", errors="replace")
    names: set[str] = set()
    for href in re.findall(r'href="(/[^"]+)"', text):
        if href in ("/", "/upload", "/list"):
            continue
        names.add(href.lstrip("/"))
    names.update(re.findall(r"[\w.\-]+\.bmp", text, flags=re.I))
    names.update(re.findall(r"[\w.\-]+\.json", text, flags=re.I))
    return sorted(n for n in names if n and not n.startswith("http"))


def names_from_resource(data: bytes) -> list[str]:
    text = data.decode("utf-8", errors="replace")
    names = set(re.findall(r'"([^"]+\.(?:bmp|json|bin|dat))"', text, flags=re.I))
    names.update(re.findall(r'[\w.\-/]+\.bmp', text, flags=re.I))
    # Common fields
    try:
        import json
        obj = json.loads(text)
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
            elif isinstance(cur, str) and ("." in cur):
                names.add(cur.lstrip("/"))
    except Exception:
        pass
    return sorted(names)


def candidate_names() -> list[str]:
    names: list[str] = []
    for gid, nmax in sorted(GROUP_MAX.items()):
        start = 0 if gid == 21 else 1
        for i in range(start, nmax + 1):
            names.append(f"{gid}_{i}.bmp")
            names.append(f"group_{gid}_{i}.bmp")
    return names


def host_up(host: str) -> bool:
    code, _, _ = http_get(f"http://{host}/", timeout=3.0)
    return code != -1


def pull_all(host: str, out_dir: str, expected: int, switch_ble: bool = False) -> int:
    os.makedirs(out_dir, exist_ok=True)
    base = f"http://{host}"
    saved = 0

    print("=== HTTP FILE LIST ===")
    code, body, ctype = http_get_retry(base + "/", timeout=8.0, tries=8)
    print(f"  / -> {code} {len(body)}B {ctype}")
    if code != 200:
        print(f"  {host} unreachable or file service not ready")
        return 0
    save_file(out_dir, "_index.html", body)
    listed = parse_names(body)

    # resource.json usually has the full asset list; the home page only shows the first 5 files
    code, res, _ = http_get_retry(base + "/resource.json", timeout=8.0, tries=6)
    if code == 200 and res:
        save_file(out_dir, "resource.json", res)
        listed.extend(names_from_resource(res))
        print(f"  resource.json {len(res)}B")
    code, cfg, _ = http_get_retry(base + "/config.json", timeout=8.0, tries=4)
    if code == 200 and cfg:
        save_file(out_dir, "config.json", cfg)
        listed.extend(names_from_resource(cfg))

    # Dedupe; prefer real filenames
    targets: list[str] = []
    seen: set[str] = set()
    for n in listed:
        key = n.lstrip("/").lower()
        if key in seen or key.startswith("_"):
            continue
        seen.add(key)
        targets.append(n.lstrip("/"))

    if len(targets) < 10:
        print("  listing looks short; append protocol candidate names")
        for n in candidate_names():
            key = n.lower()
            if key not in seen:
                seen.add(key)
                targets.append(n)

    print(f"=== DOWNLOAD {len(targets)} files -> {out_dir}/ ===")
    miss = 0
    for i, name in enumerate(targets, 1):
        # ESP32 file server: download path is /filename (upload is /upload/filename)
        url = f"{base}/{name}"
        code, body, _ = http_get_retry(url, timeout=15.0, tries=4)
        if code == -1:
            miss += 1
            print(f"  fail {name}")
            if miss >= 12:
                print("  too many consecutive failures; pause 3s then continue")
                time.sleep(3.0)
                miss = 0
            continue
        if code == 404:
            continue
        miss = 0
        if code == 200 and body:
            path = save_file(out_dir, os.path.basename(name), body)
            saved += 1
            tag = "bmp" if looks_like_bmp(body) else "bin"
            if saved <= 20 or saved % 10 == 0 or i == len(targets):
                print(f"  [{saved}] {tag} {os.path.basename(name)}  {len(body)}B")
        time.sleep(0.05)

    print(f"  saved {saved} files" + (f" (device reported {expected})" if expected >= 0 else ""))
    if switch_ble:
        http_get(base + "/switch/ble", timeout=5.0)
        print("  requested switch back to BLE")
    return saved


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--ssid", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--skip-ble", action="store_true")
    ap.add_argument("--skip-wifi", action="store_true", help="this PC is already on the device AP")
    args = ap.parse_args()

    count = -1
    addr = args.address
    ssid = args.ssid
    password = args.password or "AinstecIris123456789"
    device = None
    if not args.skip_ble:
        addr, device = await find_device(address=args.address)
        print(f"  target {addr}")
        count, ssid, password = await ble_prepare(
            addr, ssid=ssid, password=password, device=device,
        )
        print("  waiting for AP to come up ...")
        await asyncio.sleep(5.0)

    if not args.skip_wifi:
        print("=== WLAN SCAN ===")
        ssids = scan_ssid()
        iris = [s for s in ssids if "IRIS" in s.upper()]
        for s in iris:
            print(f"  {s}")
        suffix = addr.replace(":", "")[-6:]
        ssid = ssid or pick_ssid(ssids, suffix) or f"Iris-G-WLAN-{suffix}"
        print(f"=== CONNECT AP {ssid} / {password} ===")
        if not connect_wifi(ssid, password):
            print("  WiFi connect failed, cannot download")
            return 2
        time.sleep(1.0)

    n = pull_all(args.host, args.out, count)
    return 0 if n > 0 else 3


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
