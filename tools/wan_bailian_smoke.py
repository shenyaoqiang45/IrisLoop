"""Bailian Wan2.2-T2V-Plus smoke: 480P (832*480) + sampled frames.

Usage:
    set DASHSCOPE_API_KEY=...
    set DASHSCOPE_WORKSPACE_ID=llm-...
    python tools/wan_bailian_smoke.py "bold green whale silhouette ..."
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from irisloop import wan_bailian as W

DEFAULT_NEGATIVE = (
    "photorealistic skin, fine texture, busy background, tiny text, "
    "complex shading, soft gradients, clutter"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument(
        "--size",
        default=None,
        help="default 832*480 (480P 16:9); also 480*832 / 624*624",
    )
    ap.add_argument("--model", default=None, help="default wan2.2-t2v-plus")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else Path("captures") / "wan_bailian" / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    size = args.size or W.default_size()
    print(f"model   : {args.model or W.default_model()}", flush=True)
    print(f"size    : {size} (480P tier)", flush=True)
    print(f"frames  : {args.frames}", flush=True)
    print(f"out     : {out}", flush=True)

    mp4, frames = W.generate_min_clip_and_frames(
        args.prompt,
        out,
        frame_count=args.frames,
        size=size,
        negative_prompt=args.negative or None,
        model=args.model,
    )
    print(f"video   : {mp4}", flush=True)
    for p in frames:
        print(f"frame   : {p}", flush=True)
    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
