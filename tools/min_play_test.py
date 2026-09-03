"""最低可玩门槛探测：笔记本自带摄像头能否代替 USB cam。

目标硬件（理想）：IrisGreen（BLE）+ 带蓝牙的笔记本自带摄像头。
不需要外置 USB 摄像头。

用法:
    python tools/min_play_test.py
    python tools/min_play_test.py --ble --group 1 --seconds 5
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import cv2

from irisloop.camera import UsbCamera, probe_cameras, quiet_opencv

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"


def _ps(cmd: str) -> str:
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return (r.stdout or "").strip()
    except Exception as e:
        return f"(powershell failed: {e})"


def host_profile() -> dict:
    profile: dict = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "pc_system_type": None,
        "pc_system_label": "unknown",
        "manufacturer": None,
        "model": None,
        "pnp_cameras": [],
    }
    raw = _ps(
        "Get-CimInstance Win32_ComputerSystem | "
        "Select-Object -Property Manufacturer,Model,PCSystemType | ConvertTo-Json"
    )
    try:
        data = json.loads(raw) if raw else {}
        profile["manufacturer"] = data.get("Manufacturer")
        profile["model"] = data.get("Model")
        t = data.get("PCSystemType")
        profile["pc_system_type"] = t
        # 1=Desktop 2=Mobile(laptop) 3=Workstation 4=Enterprise Server ...
        labels = {1: "desktop", 2: "laptop", 3: "workstation"}
        profile["pc_system_label"] = labels.get(int(t), f"type-{t}") if t is not None else "unknown"
    except Exception:
        profile["cim_raw"] = raw[:400]

    names = _ps(
        "Get-PnpDevice -Class Camera,Image -ErrorAction SilentlyContinue | "
        "Where-Object { $_.Status -eq 'OK' } | "
        "Select-Object -ExpandProperty FriendlyName"
    )
    if names and not names.startswith("(powershell"):
        profile["pnp_cameras"] = [n.strip() for n in names.splitlines() if n.strip()]
    return profile


def classify_names(names: list[str]) -> dict:
    builtin_kw = (
        "integrated", "built-in", "laptop", "front", "rear",
        "ir camera", "infrared", "rgb camera", "user facing",
        "内置", "前置", "后置", "集成",
    )
    usb_kw = ("lifecam", "logitech", "brio", "c920", "c922", "usb")
    builtin, usb, other = [], [], []
    for n in names:
        low = n.lower()
        if any(k in low for k in usb_kw):
            usb.append(n)
        elif any(k in low for k in builtin_kw):
            builtin.append(n)
        else:
            other.append(n)
    return {"likely_builtin": builtin, "likely_usb": usb, "other": other}


def grab_still(index: int, out_path: Path) -> dict:
    cam = UsbCamera(index, 1280, 720, 30)
    cam.open()
    # 丢弃几帧让自动曝光稳定
    frame = None
    ok = False
    for _ in range(12):
        ok, frame = cam.read()
    info = cam.info()
    if ok and frame is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stats = {
            "ok": True,
            "info": info,
            "path": str(out_path),
            "mean": float(g.mean()),
            "std": float(g.std()),
            "max": int(g.max()),
        }
    else:
        stats = {"ok": False, "info": info, "error": "read failed"}
    cam.release()
    return stats


def verdict(profile: dict, probes: list[dict], stills: list[dict]) -> list[str]:
    lines: list[str] = []
    kind = profile.get("pc_system_label")
    pnp = profile.get("pnp_cameras") or []
    cls = classify_names(pnp)
    n_ok = sum(1 for s in stills if s.get("ok"))

    lines.append(f"主机类型: {kind} ({profile.get('manufacturer')} {profile.get('model')})")
    lines.append(f"系统登记的成像设备: {pnp or '无'}")
    lines.append(f"OpenCV 可用索引: {[p['index'] for p in probes] or '无'}")
    lines.append(f"成功出帧的摄像头: {n_ok}/{len(stills)}")

    if kind == "desktop" and not cls["likely_builtin"]:
        lines.append(
            "结论: 当前是台式机，没有笔记本内置/后置摄像头，只有外置 USB cam 可测。"
            "最低门槛（笔记本 BLE + 自带摄像头）需要换一台带摄像头的笔记本实机验证。"
        )
        if n_ok:
            lines.append("本机 USB cam 采集链路可用，软件侧已支持内置摄像头的 MJPG 回退。")
        return lines

    if not probes:
        lines.append(
            "结论: 系统未枚举到摄像头。检查：Windows 隐私-相机允许桌面应用、BIOS 摄像头开关、驱动。"
        )
        return lines

    if n_ok:
        lines.append(
            "结论: 本机摄像头能出帧。用户最低可玩路径可以是："
            "笔记本蓝牙连 IrisGreen + 把自带摄像头对准投影面（不必再买 USB cam）。"
        )
        lines.append(
            "姿势提示: 普通笔记本摄像头在屏幕顶端朝向用户；拍桌面投影可把屏幕合到约 20–40° 让镜头朝下；"
            "拍墙面投影用二合一帐篷模式或把整机侧过来。后置镜头仅部分二合一/翻转屏机型才有。"
        )
    else:
        lines.append("结论: 设备在列表里但读不到帧，多半是占用/权限/IR 摄像头被误选。")
    return lines


async def ble_loop(addr: str, group: int, seconds: float, camera: int, out_dir: str) -> int:
    from tools.project_and_capture import run_group

    print(f"\n=== BLE 投射 + 本机摄像头 ===")
    print(f"  {addr}  group={group}  cam={camera}  {seconds}s")
    r = await run_group(
        addr, group, seconds, camera, None, False, None,
        n_stills=4, jpeg_quality=92, out_dir=out_dir,
    )
    print(f"  result={r}")
    return 0 if r.get("frames") else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="探测笔记本自带摄像头能否代替 USB cam")
    ap.add_argument("--ble", action="store_true", help="同时 BLE 播一组并采集")
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--camera", type=int, default=None, help="BLE 采集用的摄像头索引，默认第一个可用")
    args = ap.parse_args()

    tag = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path("captures") / f"min_play_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=== IrisLoop 最低可玩门槛探测 ===")
    profile = host_profile()
    print(json.dumps({k: profile[k] for k in profile if k != "cim_raw"}, ensure_ascii=False, indent=2))

    with quiet_opencv():
        probes = probe_cameras()
    print("\nOpenCV 探测:")
    for p in probes:
        print(f"  [{p['index']}] {p['width']}x{p['height']} @{p['fps']:.0f}fps {p['backend']}")
    if not probes:
        print("  (无)")

    stills = []
    indexes = [p["index"] for p in probes] or [0]
    for idx in indexes:
        dest = out_root / f"cam{idx}_still.jpg"
        print(f"\n抓拍 cam#{idx} -> {dest}")
        try:
            s = grab_still(idx, dest)
        except Exception as e:
            s = {"ok": False, "index": idx, "error": f"{type(e).__name__}: {e}"}
        stills.append(s)
        print(f"  {s}")

    print("\n=== 判断 ===")
    for line in verdict(profile, probes, stills):
        print("  " + line)

    report = {
        "profile": profile,
        "probes": probes,
        "stills": stills,
        "verdict": verdict(profile, probes, stills),
        "out": str(out_root),
    }
    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告 {report_path}")

    if not args.ble:
        print("\n下一步: 在带摄像头的笔记本上再跑一次；真机投射加 --ble --group 1")
        return 0 if any(s.get("ok") for s in stills) else 1

    import asyncio

    cam_index = args.camera if args.camera is not None else indexes[0]
    ble_dir = str(out_root / "ble")
    os.makedirs(ble_dir, exist_ok=True)
    return asyncio.run(ble_loop(args.address, args.group, args.seconds, cam_index, ble_dir))


if __name__ == "__main__":
    raise SystemExit(main())
