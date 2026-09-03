"""投射素材组 + USB 摄像头同步采集 —— IrisLoop 闭环第一步。

把「投射 -> 观察」串起来：BLE 下发播放命令，同时用摄像头录下
投影仪在物理表面上的实际效果，用于确认图案、朝向、亮度、畸变。

用法:
    python tools/project_and_capture.py --group 1 --seconds 6
    python tools/project_and_capture.py --group 20 --seconds 6 --exposure -8
    python tools/project_and_capture.py --group 1 20 --seconds 8   # 依次播两组
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import cv2
import numpy as np

from irisloop import protocol as P
from irisloop.ble_client import IrisBleClient
from irisloop.camera import UsbCamera

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"
OUT_DIR = "captures"


def analyze(frame: np.ndarray) -> dict:
    """量化分析采集帧，判断是否有内容投出。"""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "max": int(g.max()),
        "bright_ratio": float((g > 128).mean()),
        "lap_var": float(cv2.Laplacian(g, cv2.CV_64F).var()),
    }


async def run_group(
    addr: str,
    group: int,
    seconds: float,
    cam_index: int,
    exposure: float | None,
    keepalive: bool,
    interval_100ms: int | None = None,
    n_stills: int = 6,
    jpeg_quality: int = 92,
    out_dir: str | None = None,
) -> dict:
    cli = IrisBleClient(addr)
    await cli.connect()
    await cli.stop()  # 先停掉上一次播放
    await asyncio.sleep(0.3)

    gname = P.group_name(group)
    # 播放总时长留足余量（单位 100ms）；帧间隔默认 2 -> 200ms/帧（5fps）
    total = int(seconds * 10) + 60
    interval = interval_100ms if interval_100ms is not None else 2
    print(f"\n=== 播放 组{group} ({gname}) total={total/10:.0f}s interval={interval*100}ms ===")
    r = await cli.play(group_id=group, loop=False, total_100ms=total, interval_100ms=interval)
    print(f"  {'ok' if r.ok else 'fail'} raw={r.raw.hex()} {r.error}")

    # 投影仪响应 + MEMS 起振需要时间
    await asyncio.sleep(1.2)

    # ---- 摄像头采集 ----
    cam = UsbCamera(cam_index, 1280, 720, 30)
    cam.open()
    print(f"  camera: {cam.info()}")
    if exposure is not None:
        cam.set_auto_exposure(False)
        cam.set_exposure(exposure)
        print(f"  手动曝光 {exposure}")

    if out_dir is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        tag = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(OUT_DIR, f"group{group}_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    frames: list[np.ndarray] = []
    stats: list[dict] = []
    t0 = time.perf_counter()
    idx = 0
    n_stills = max(1, n_stills)
    lo, hi = 0.4, max(0.6, seconds - 0.25)
    shot_times = np.linspace(lo, hi, n_stills)
    next_shot = 0

    print(f"  采集中 ({seconds}s, {n_stills} 张 JPG) ...")
    while time.perf_counter() - t0 < seconds:
        ok, frame = cam.read()
        if not ok or frame is None:
            continue
        el = time.perf_counter() - t0
        frames.append(frame)
        stats.append(analyze(frame))
        if next_shot < len(shot_times) and el >= shot_times[next_shot]:
            p = os.path.join(out_dir, f"frame_{idx:02d}_{el:.1f}s.jpg")
            cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            idx += 1
            next_shot += 1
        await asyncio.sleep(0)

    elapsed = time.perf_counter() - t0
    cam.release()

    if frames:
        # 保存视频
        vid = os.path.join(out_dir, f"group{group}.mp4")
        h, w = frames[0].shape[:2]
        wr = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
        for f in frames:
            wr.write(f)
        wr.release()

        means = [s["mean"] for s in stats]
        brights = [s["bright_ratio"] for s in stats]
        laps = [s["lap_var"] for s in stats]
        print(f"\n  === 采集结果 ===")
        print(f"  帧数 {len(frames)} / {elapsed:.1f}s -> {vid}")
        print(f"  平均亮度 {min(means):.1f} ~ {max(means):.1f} (均值 {np.mean(means):.1f})")
        print(f"  亮区占比 {min(brights)*100:.1f}% ~ {max(brights)*100:.1f}%")
        print(f"  清晰度(LapVar) 均值 {np.mean(laps):.0f}")
        print(f"  截图目录 {out_dir}")

        verdict = []
        if max(means) < 12:
            verdict.append("画面几乎全黑 -> 可能未投出/激光未亮/曝光过低")
        if max(brights) < 0.005:
            verdict.append("亮区极少 -> 可能只有极暗图案")
        if np.mean(laps) < 50:
            verdict.append("画面模糊或无内容")
        if not verdict:
            verdict.append("画面有明显亮暗结构 -> 投影内容已投出")
        for v in verdict:
            print(f"  判断: {v}")

        result = {
            "group": group,
            "frames": len(frames),
            "mean": float(np.mean(means)),
            "max_mean": float(max(means)),
            "bright_ratio": float(max(brights)),
            "lap_var": float(np.mean(laps)),
            "dir": out_dir,
        }
    else:
        print("  [warn] 未采集到任何帧")
        result = {"group": group, "frames": 0}

    if not keepalive:
        await cli.stop()
        print("  已停止播放")
    await cli.disconnect()
    return result


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--group", type=lambda x: int(x, 0), nargs="+", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--interval", type=int, default=None,
                    help="每帧间隔（单位 100ms），默认 2=200ms/帧")
    ap.add_argument("--n-stills", type=int, default=6,
                    help="均匀抓拍 JPG 张数（组 20 建议 10）")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--out-dir", default=None, help="指定输出目录（单组时用）")
    ap.add_argument("--keepalive", action="store_true", help="采集完不停止播放")
    args = ap.parse_args()

    print(f"=== IrisLoop 投射+采集 ===")
    print(f"  投影仪 {args.address}")
    print(f"  摄像头 index={args.camera}")
    print(f"  素材组 {args.group}")

    if args.out_dir and len(args.group) != 1:
        print("--out-dir 仅支持单组")
        return 2

    results = []
    for g in args.group:
        results.append(
            await run_group(
                args.address, g, args.seconds,
                args.camera, args.exposure, args.keepalive, args.interval,
                n_stills=args.n_stills, jpeg_quality=args.jpeg_quality,
                out_dir=args.out_dir,
            )
        )

    print("\n=== 汇总 ===")
    for r in results:
        if r.get("frames"):
            print(f"  组{r['group']}: {r['frames']}帧  平均亮度{r['mean']:.1f}  "
                  f"峰值{r['max_mean']:.1f}  亮区{r['bright_ratio']*100:.1f}%  "
                  f"清晰度{r['lap_var']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
