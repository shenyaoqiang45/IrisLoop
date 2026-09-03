"""SiliconFlow Wan2.2-T2V-A14B writer smoke: ~5s clip + sampled frames.

Preferred IrisLoop writer path (cost). Kling/Bailian remains as fallback.

Usage:
    set SILICONFLOW_API_KEY=sk-...
    python tools/wan_siliconflow_smoke.py "bold green whale silhouette swimming, high contrast, simple shapes"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from irisloop import wan_siliconflow as W

DEFAULT_NEGATIVE = (
    "photorealistic skin, fine texture, busy background, tiny text, "
    "complex shading, soft gradients, clutter"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", help="text prompt for Wan T2V")
    ap.add_argument(
        "--frames",
        type=int,
        default=3,
        help="evenly sampled stills (default 3; use 10 before push group 1)",
    )
    ap.add_argument(
        "--image-size",
        default=None,
        choices=["1280x720", "720x1280", "960x960"],
        help="official enum only; default 1280x720",
    )
    ap.add_argument("--model", default=None, help="override SILICONFLOW_WAN_MODEL")
    ap.add_argument("-o", "--out", default=None, help="output dir")
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else Path("captures") / "wan" / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    neg = args.negative or None

    size = args.image_size or W.default_image_size()
    print(f"model      : {args.model or W.default_model()}")
    print(f"image_size : {size}")
    print(f"nominal    : ~{W.NOMINAL_DURATION_S}s (fixed by SiliconFlow Wan2.2)")
    print(f"frames     : {args.frames}")
    print(f"out        : {out}")
    print("submitting…", flush=True)

    mp4, frames = W.generate_min_clip_and_frames(
        args.prompt,
        out,
        frame_count=args.frames,
        image_size=size,
        negative_prompt=neg,
        model=args.model,
        seed=args.seed,
    )
    print(f"video      : {mp4}")
    for p in frames:
        print(f"frame      : {p}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
