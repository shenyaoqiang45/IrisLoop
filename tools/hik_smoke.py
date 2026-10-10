"""海康相机操作冒烟测试：枚举 → 打开 → 参数设置 → 抓帧存图 → 帧率实测。

用法:
    python tools/hik_smoke.py                          # 默认抓 5 帧存 tmp/hik_smoke.png
    python tools/hik_smoke.py --exposure 8000 --gain 6 # 指定曝光(us)/增益(dB)
    python tools/hik_smoke.py --fps                    # 加测 3 秒实际帧率
    python tools/hik_smoke.py --mvs-dir "D:\\MVS"      # 非默认安装路径
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from irisloop.hik_camera import HikCamera  # noqa: E402

OUT = os.path.join("tmp", "hik_smoke.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None,
                    help="枚举序号；不指定则自动选第一台 USB3 相机")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--exposure", type=float, default=None, help="曝光 us；不指定则保持相机当前值")
    ap.add_argument("--gain", type=float, default=None, help="增益 dB")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--fps", action="store_true", help="实测帧率(3s)")
    ap.add_argument("--mvs-dir", default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    try:
        cam = HikCamera(args.mvs_dir)
    except RuntimeError as e:
        print(f"[error] {e}")
        return 2

    print("=== ENUMERATE ===")
    try:
        devices = cam.enumerate()
    except RuntimeError as e:
        print(f"[error] {e}")
        return 2
    if not devices:
        print("未枚举到设备 —— 检查 USB3 线缆、MVS USB3 驱动、相机供电")
        return 1
    for d in devices:
        print(f"  [{d['index']}] {d['interface']} {d['model']} sn={d['serial']}")

    print("=== OPEN ===")
    index, serial = args.index, args.serial
    if index is None and serial is None:
        usb = [d for d in devices if d["interface"] == "USB3"]
        if usb:
            serial = usb[0]["serial"]
            print(f"  auto-select USB3: {usb[0]['model']} sn={serial}")
        else:
            index = 0
    try:
        cam.open(index=index if index is not None else 0, serial=serial)
    except RuntimeError as e:
        print(f"[error] {e}")
        return 2
    print(f"  {cam.info()}")

    try:
        if args.exposure is not None:
            cam.set_exposure_us(args.exposure)
        if args.gain is not None:
            cam.set_gain(args.gain)
        print(f"  exposure={cam.get_exposure_us()}us gain={cam.get_gain()}dB")

        print(f"=== GRAB {args.frames} frames ===")
        frame = None
        for i in range(args.frames):
            ok, f = cam.read(2000)
            print(f"  frame {i + 1}: {'ok' if ok else 'TIMEOUT/FAIL'}")
            if ok and f is not None:
                frame = f
        if frame is None:
            print("[error] 未抓到任何帧")
            return 3

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, frame)
        h, w = frame.shape[:2]
        mean = float(frame.mean())
        print(f"  saved -> {args.out}  ({w}x{h}, mean={mean:.1f})")

        if args.fps:
            fps = cam.measure_fps(3.0)
            print(f"  measured fps: {fps:.1f}")
    finally:
        cam.close()
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
