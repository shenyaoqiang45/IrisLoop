"""Bailian Kling writer smoke (fallback): min clip (3s) + frames.

Preferred writer: ``tools/wan_siliconflow_smoke.py`` (SiliconFlow Wan2.2).
Bailian Kling v3 does not offer 1s clips — API floor is 3 seconds.

Usage:
    set DASHSCOPE_API_KEY=sk-...
    set DASHSCOPE_WORKSPACE_ID=...
    python tools/kling_bailian_smoke.py "bold green whale silhouette swimming, high contrast, simple shapes"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from irisloop import kling_bailian as K

# MEMS-friendly negative prompt (writer side); director still owns "good enough"
DEFAULT_NEGATIVE = (
    "photorealistic skin, fine texture, busy background, tiny text, "
    "complex shading, soft gradients, clutter"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", help="text prompt for Kling")
    ap.add_argument(
        "--duration",
        type=int,
        default=K.DEFAULT_DURATION_S,
        help=f"seconds (API min {K.MIN_DURATION_S}; default {K.DEFAULT_DURATION_S})",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=K.DEFAULT_FRAME_COUNT,
        help=f"evenly sampled stills (default {K.DEFAULT_FRAME_COUNT})",
    )
    ap.add_argument("--mode", default="std", choices=["std", "pro", "4k"])
    ap.add_argument("--model", default=None, help="override DASHSCOPE_KLING_MODEL")
    ap.add_argument(
        "-o",
        "--out",
        default=None,
        help="output dir (default captures/kling/<timestamp>)",
    )
    ap.add_argument(
        "--negative",
        default=DEFAULT_NEGATIVE,
        help="negative prompt (empty string to disable)",
    )
    args = ap.parse_args(argv)

    if args.duration < K.MIN_DURATION_S:
        print(
            f"[warn] Bailian Kling v3 minimum is {K.MIN_DURATION_S}s "
            f"(not 1s). Using {K.MIN_DURATION_S}s.",
            file=sys.stderr,
        )
        args.duration = K.MIN_DURATION_S

    out = Path(args.out) if args.out else Path("captures") / "kling" / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    neg = args.negative or None

    print(f"model     : {args.model or K.default_model()}")
    print(f"duration  : {args.duration}s (billed)")
    print(f"frames    : {args.frames} stills after download")
    print(f"out       : {out}")
    print("submitting async job…")

    mp4, frames = K.generate_min_clip_and_frames(
        args.prompt,
        out,
        duration=args.duration,
        frame_count=args.frames,
        mode=args.mode,
        negative_prompt=neg,
        model=args.model,
    )
    print(f"video     : {mp4}")
    for p in frames:
        print(f"frame     : {p}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
