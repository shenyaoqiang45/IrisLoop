"""从 IrisGreen 拉取设备内素材到本地 data/。

设备没有 BLE 读图命令；协议规定切到 WiFi AP 后走 HTTP：
    打开 WiFi:  写 0x02 到 adb40005
    上传/取图:  http://192.168.4.1/upload/<name>.bmp
    切回 BLE:  http://192.168.4.1/switch/ble

流程:
    1. BLE 扫描并连接，读图片总数
    2. 打开设备 WiFi
    3. 本机 WLAN 连上 AP
    4. 探测目录，再按协议文件名逐张 GET
    5. 切回 BLE
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

# 协议 Sheet2 给出的各组最大张数（十进制组号）
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
    """返回 (address, BLEDevice|None)。尽量带回扫描到的设备对象，便于 WinRT 立刻连接。"""
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
    raise RuntimeError("未扫描到 IrisGreen")


def build_wifi_cred(ssid: str, password: str) -> bytes:
    """构造 adb40005 命令01：配置 WiFi 名称/密码（共 98 字节）。"""
    sb = ssid.encode("utf-8")
    pb = password.encode("utf-8")
    if len(sb) > 32 or len(pb) > 64:
        raise ValueError("SSID/密码过长")
    name = sb.ljust(32, b"\x00")
    pwd = pb.ljust(64, b"\x00")
    return bytes([0x01, len(sb)]) + name + pwd


async def ble_prepare(
    address: str,
    ssid: str = "",
    password: str = "AinstecIris123456789",
    device=None,
) -> tuple[int, str, str]:
    """连接、读图片数、配置并打开 WiFi。返回 (图片总数, ssid, password)。"""
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
            print(f"  图片总数 {count}")
        else:
            print("  读图片总数失败")
        cred = build_wifi_cred(ssid, password)
        print(f"=== SET WIFI ssid={ssid} pass={password} ===")
        await client.write_gatt_char(CHAR_MAIN_CMD2, cred, response=True)
        await asyncio.sleep(0.5)
        print("=== OPEN WIFI (adb40005 <- 02) ===")
        await client.write_gatt_char(CHAR_MAIN_CMD2, b"\x02", response=True)
        print("  已发送打开 WiFi")
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
    # 设备示例里 SSID=密码；若开放网络，仍先尝试 WPA2
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
    # 关联成功时会打印 SSID/BSSID；断开时没有这两行。
    # 避免依赖「已连接」中文（控制台编码会乱码）。
    has_ssid = re.search(rf"SSID\s*:\s*{re.escape(ssid)}\s*$", out, re.I | re.M)
    has_bssid = re.search(r"BSSID\s*:\s*([0-9a-f:]{17})", out, re.I)
    return bool(has_ssid and has_bssid)


def connect_wifi(ssid: str, password: str) -> bool:
    if wifi_connected_to(ssid):
        log(f"  WLAN 已在 {ssid}")
        return True
    profile = os.path.join(os.environ.get("TEMP", "."), "iris_wlan.xml")
    run(["netsh", "wlan", "delete", "profile", f"name={ssid}"])
    write_wlan_profile(ssid, password, profile)
    log(run(["netsh", "wlan", "add", "profile", f"filename={profile}", "user=current"]))
    log(run(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}", "interface=WLAN"]))
    deadline = time.time() + 30
    while time.time() < deadline:
        if wifi_connected_to(ssid):
            log(f"  WLAN 已连接 {ssid}")
            # 等 DHCP
            for _ in range(15):
                ip = run(["netsh", "interface", "ip", "show", "addresses", "WLAN"])
                if "192.168.4." in ip:
                    log("  已拿到 192.168.4.x")
                    return True
                time.sleep(1.0)
            log("  已关联但未拿到 192.168.4.x")
            return True
        time.sleep(1.0)
    log("  等待连接超时")
    log(run(["netsh", "wlan", "show", "interfaces"]))
    return False


def _wlan_local_ip() -> str | None:
    """拿到连着设备热点时的本机地址（通常 192.168.4.2）。"""
    out = run(["netsh", "interface", "ip", "show", "addresses", "WLAN"])
    m = re.search(r"192\.168\.4\.\d+", out)
    return m.group(0) if m else None


def http_get(url: str, timeout: float = 8.0) -> tuple[int, bytes, str]:
    """直连设备 AP，绑定 WLAN 地址并禁用系统代理（避免 Meta/Clash TUN 劫持）。"""
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
    # 环境代理也可能劫持；临时清空
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
    # 常见字段
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
        print(f"  {host} 不可达或文件服务未就绪")
        return 0
    save_file(out_dir, "_index.html", body)
    listed = parse_names(body)

    # resource.json 通常有完整素材清单；首页只显示前 5 个文件
    code, res, _ = http_get_retry(base + "/resource.json", timeout=8.0, tries=6)
    if code == 200 and res:
        save_file(out_dir, "resource.json", res)
        listed.extend(names_from_resource(res))
        print(f"  resource.json {len(res)}B")
    code, cfg, _ = http_get_retry(base + "/config.json", timeout=8.0, tries=4)
    if code == 200 and cfg:
        save_file(out_dir, "config.json", cfg)
        listed.extend(names_from_resource(cfg))

    # 去重，优先真实文件名
    targets: list[str] = []
    seen: set[str] = set()
    for n in listed:
        key = n.lstrip("/").lower()
        if key in seen or key.startswith("_"):
            continue
        seen.add(key)
        targets.append(n.lstrip("/"))

    if len(targets) < 10:
        print("  列表偏少，追加协议候选名")
        for n in candidate_names():
            key = n.lower()
            if key not in seen:
                seen.add(key)
                targets.append(n)

    print(f"=== DOWNLOAD {len(targets)} files -> {out_dir}/ ===")
    miss = 0
    for i, name in enumerate(targets, 1):
        # ESP32 文件服务器：下载路径是 /filename（上传才是 /upload/filename）
        url = f"{base}/{name}"
        code, body, _ = http_get_retry(url, timeout=15.0, tries=4)
        if code == -1:
            miss += 1
            print(f"  fail {name}")
            if miss >= 12:
                print("  连续失败过多，暂停 3s 后继续")
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

    print(f"  已保存 {saved} 个文件" + (f"（设备上报 {expected} 张）" if expected >= 0 else ""))
    if switch_ble:
        http_get(base + "/switch/ble", timeout=5.0)
        print("  已请求切回 BLE")
    return saved


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--ssid", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--skip-ble", action="store_true")
    ap.add_argument("--skip-wifi", action="store_true", help="本机已连上设备 AP")
    args = ap.parse_args()

    count = -1
    addr = args.address
    ssid = args.ssid
    password = args.password or "AinstecIris123456789"
    device = None
    if not args.skip_ble:
        addr, device = await find_device(address=args.address)
        print(f"  目标 {addr}")
        count, ssid, password = await ble_prepare(
            addr, ssid=ssid, password=password, device=device,
        )
        print("  等待 AP 起来 ...")
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
            print("  WiFi 连接失败，无法下载")
            return 2
        time.sleep(1.0)

    n = pull_all(args.host, args.out, count)
    return 0 if n > 0 else 3


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
