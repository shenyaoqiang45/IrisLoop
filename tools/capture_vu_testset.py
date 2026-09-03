"""采集素材组 1–20 的 USB 实拍，落成视频理解测试集。

每组用独立子进程跑 project_and_capture（规避 Windows Bleak 重连失效）。

每组输出:
    captures/vu_testset_<tag>/group_XX/
        frame_*.jpg / still 同批 JPG
        group.mp4
        stats.json   # 由本脚本根据目录汇总
根目录:
    manifest.json

用法:
    python tools/capture_vu_testset.py
    python tools/capture_vu_testset.py --groups 7 8 9 10 --tag 20260902_175407
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import cv2
import numpy as np

from irisloop import protocol as P

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"
OUT_ROOT = Path("captures")
SETTLE_S = 3.0


def analyze_dir(group: int, out_dir: Path) -> dict:
    gname = P.group_name(group)
    jpgs = sorted(out_dir.glob("*.jpg"))
    mp4_candidates = [
        out_dir / "group.mp4",
        out_dir / f"group{group}.mp4",
    ]
    mp4 = next((p for p in mp4_candidates if p.exists()), None)
    means: list[float] = []
    brights: list[float] = []
    laps: list[float] = []
    for p in jpgs:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        means.append(float(g.mean()))
        brights.append(float((g > 128).mean()))
        laps.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))

    verdict: list[str] = []
    if not jpgs:
        verdict.append("无 JPG")
    else:
        if max(means) < 12:
            verdict.append("几乎全黑")
        if max(brights) < 0.005:
            verdict.append("亮区极少")
        if float(np.mean(laps)) < 50:
            verdict.append("偏模糊或扫描条纹主导")
        if not verdict:
            verdict.append("有明显投影结构")

    entry = {
        "group": group,
        "name": gname,
        "frames": None,
        "stills": [p.name for p in jpgs],
        "video": mp4.name if mp4 is not None else None,
        "mean": float(np.mean(means)) if means else 0.0,
        "max_mean": float(max(means)) if means else 0.0,
        "bright_ratio": float(max(brights)) if brights else 0.0,
        "lap_var": float(np.mean(laps)) if laps else 0.0,
        "verdict": verdict,
        "dir": str(out_dir).replace("\\", "/"),
        "ok": bool(jpgs) and (max(means) if means else 0) >= 12,
    }
    (out_dir / "stats.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return entry


def run_one_group(
    addr: str,
    group: int,
    out_dir: Path,
    seconds: float,
    interval: int,
    n_stills: int,
    camera: int,
    exposure: float,
    jpeg_quality: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "tools/project_and_capture.py",
        "--address",
        addr,
        "--group",
        str(group),
        "--seconds",
        str(seconds),
        "--interval",
        str(interval),
        "--n-stills",
        str(n_stills),
        "--camera",
        str(camera),
        "--exposure",
        str(exposure),
        "--jpeg-quality",
        str(jpeg_quality),
        "--out-dir",
        str(out_dir),
    ]
    print(f"\n=== 子进程采集 组 {group:02d} ({P.group_name(group)}) ===")
    print(" ", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            env=env,
            timeout=seconds + 90,
        )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        print("  [error] 子进程超时")
        rc = -1

    entry = analyze_dir(group, out_dir)
    entry["subprocess_rc"] = rc
    print(
        f"  jpg={len(entry['stills'])} video={'yes' if entry['video'] else 'no'} "
        f"亮度={entry['mean']:.1f} 亮区={entry['bright_ratio']*100:.1f}% "
        f"| {'; '.join(entry['verdict'])}"
    )
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument(
        "--groups",
        type=lambda x: int(x, 0),
        nargs="+",
        default=list(range(1, 21)),
    )
    ap.add_argument("--seconds", type=float, default=22.0)
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--n-stills", type=int, default=10)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--exposure", type=float, default=-7.0)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--settle", type=float, default=SETTLE_S,
                    help="组间等待秒数，给 BLE 栈冷却")
    ap.add_argument("--skip-ok", action="store_true",
                    help="跳过 manifest 里已成功的组")
    args = ap.parse_args()

    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    root = OUT_ROOT / f"vu_testset_{tag}"
    root.mkdir(parents=True, exist_ok=True)
    man_path = root / "manifest.json"

    results_by_group: dict[int, dict] = {}
    if man_path.exists():
        try:
            old = json.loads(man_path.read_text(encoding="utf-8"))
            for r in old.get("groups") or []:
                if r.get("stills"):
                    results_by_group[int(r["group"])] = r
            print(f"已有组: {sorted(results_by_group)}")
        except Exception:
            pass

    print("=== IrisLoop 视频理解测试集采集（子进程模式）===")
    print(f"  输出 {root}")
    print(f"  组 {args.groups}")

    for g in args.groups:
        if args.skip_ok and results_by_group.get(g, {}).get("ok"):
            print(f"\n=== 跳过已成功 组 {g:02d} ===")
            continue
        out_dir = root / f"group_{g:02d}"
        entry = run_one_group(
            args.address,
            g,
            out_dir,
            args.seconds,
            args.interval,
            args.n_stills,
            args.camera,
            args.exposure,
            args.jpeg_quality,
        )
        results_by_group[g] = entry

        # 写中间 manifest，方便中断后续采
        results = [results_by_group[k] for k in sorted(results_by_group)]
        manifest = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "video_understanding_testset",
            "projector": args.address,
            "camera": args.camera,
            "exposure": args.exposure,
            "seconds_per_group": args.seconds,
            "interval_100ms": args.interval,
            "groups": results,
            "ok_groups": [r["group"] for r in results if r.get("ok")],
            "empty_or_dark": [r["group"] for r in results if not r.get("ok")],
        }
        man_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        time.sleep(args.settle)

    results = [results_by_group[k] for k in sorted(results_by_group)]
    print("\n=== 汇总 ===")
    print(f"  目录 {root}")
    print(f"  manifest {man_path}")
    print(f"  成功: {[r['group'] for r in results if r.get('ok')]}")
    print(f"  失败/全黑: {[r['group'] for r in results if not r.get('ok')]}")
    for r in results:
        print(
            f"  组{r['group']:02d} {r.get('name','')}  "
            f"jpg={len(r.get('stills') or [])}  "
            f"{'; '.join(r.get('verdict') or [])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
