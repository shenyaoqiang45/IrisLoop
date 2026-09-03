"""Project a material group and capture with a USB camera — IrisLoop closed-loop step 1.

Chains project -> observe: send a BLE play command while recording what the
projector actually looks like on a physical surface (pattern, orientation, brightness, distortion).

Usage:
    python tools/project_and_capture.py --group 1 --seconds 6
    python tools/project_and_capture.py --group 20 --seconds 6 --exposure -8
    python tools/project_and_capture.py --group 1 20 --seconds 8   # play two groups in sequence
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
    """Quantify a captured frame to decide whether content was projected."""
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
    last_err: BaseException | None = None
    for attempt in range(1, 4):
        try:
            await cli.connect()
            await cli.stop()  # stop any previous playback first
            last_err = None
            break
        except OSError as e:
            last_err = e
            print(f"  BLE start abort (attempt {attempt}/3): {e}", flush=True)
            try:
                await cli.disconnect()
            except Exception:
                pass
            await asyncio.sleep(2.0 * attempt)
            cli = IrisBleClient(addr)
    if last_err is not None:
        raise last_err
    await asyncio.sleep(0.3)

    gname = P.group_name(group)
    # Leave headroom on total play time (units of 100ms); default interval 2 -> 200ms/frame (5fps)
    total = int(seconds * 10) + 60
    interval = interval_100ms if interval_100ms is not None else 2
    print(f"\n=== play group {group} ({gname}) total={total/10:.0f}s interval={interval*100}ms ===")
    r = await cli.play(group_id=group, loop=False, total_100ms=total, interval_100ms=interval)
    print(f"  {'ok' if r.ok else 'fail'} raw={r.raw.hex()} {r.error}")

    # Projector response + MEMS spin-up need time
    await asyncio.sleep(1.2)

    # ---- camera capture ----
    cam = UsbCamera(cam_index, 1280, 720, 30)
    cam.open()
    print(f"  camera: {cam.info()}")
    if exposure is not None:
        cam.set_auto_exposure(False)
        cam.set_exposure(exposure)
        print(f"  manual exposure {exposure}")

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

    print(f"  capturing ({seconds}s, {n_stills} JPGs) ...")
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
        # Save video
        vid = os.path.join(out_dir, f"group{group}.mp4")
        h, w = frames[0].shape[:2]
        wr = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
        for f in frames:
            wr.write(f)
        wr.release()

        means = [s["mean"] for s in stats]
        brights = [s["bright_ratio"] for s in stats]
        laps = [s["lap_var"] for s in stats]
        print(f"\n  === capture result ===")
        print(f"  frames {len(frames)} / {elapsed:.1f}s -> {vid}")
        print(f"  brightness {min(means):.1f} ~ {max(means):.1f} (mean {np.mean(means):.1f})")
        print(f"  bright-area ratio {min(brights)*100:.1f}% ~ {max(brights)*100:.1f}%")
        print(f"  sharpness (LapVar) mean {np.mean(laps):.0f}")
        print(f"  stills dir {out_dir}")

        verdict = []
        if max(means) < 12:
            verdict.append("almost fully black -> maybe not projected / laser off / exposure too low")
        if max(brights) < 0.005:
            verdict.append("very little bright area -> maybe an extremely dark pattern")
        if np.mean(laps) < 50:
            verdict.append("blurry or no content")
        if not verdict:
            verdict.append("clear bright/dark structure -> projected content is on")
        for v in verdict:
            print(f"  verdict: {v}")

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
        print("  [warn] no frames captured")
        result = {"group": group, "frames": 0}

    if not keepalive:
        await cli.stop()
        print("  playback stopped")
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
                    help="Per-frame interval (units of 100ms); default 2=200ms/frame")
    ap.add_argument("--n-stills", type=int, default=6,
                    help="Number of evenly spaced JPG stills (10 recommended for group 20)")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--out-dir", default=None, help="Output directory (single-group only)")
    ap.add_argument("--keepalive", action="store_true", help="Do not stop playback after capture")
    args = ap.parse_args()

    print(f"=== IrisLoop project+capture ===")
    print(f"  projector {args.address}")
    print(f"  camera index={args.camera}")
    print(f"  groups {args.group}")

    if args.out_dir and len(args.group) != 1:
        print("--out-dir supports a single group only")
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

    print("\n=== Summary ===")
    for r in results:
        if r.get("frames"):
            print(f"  group {r['group']}: {r['frames']} frames  brightness mean {r['mean']:.1f}  "
                  f"peak {r['max_mean']:.1f}  bright-area {r['bright_ratio']*100:.1f}%  "
                  f"sharpness {r['lap_var']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
