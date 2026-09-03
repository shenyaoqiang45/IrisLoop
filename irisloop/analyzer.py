"""IrisLoop vision analyzer — compare a target image with a camera frame and emit a structured assessment.

Two modes:
  - rule: OpenCV-only heuristics (no API key, offline)
  - ai:   multimodal LLM assessment (requires MOONSHOT_API_KEY)

Usage:
    python -m irisloop.analyzer target.bmp capture.png
    python -m irisloop.analyzer target.bmp capture.png --mode ai
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np


@dataclass
class Assessment:
    """One projection-quality assessment."""

    ok: bool = False                    # overall pass
    brightness_mean: float = 0.0        # captured mean brightness 0-255
    bright_ratio: float = 0.0           # bright-pixel ratio 0-1
    sharpness: float = 0.0              # Laplacian variance (sharpness)
    orientation: str = "unknown"        # normal / flip_v / flip_h / rot180 / unknown
    similarity: float = 0.0             # structural similarity vs target 0-1
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"  ok={self.ok}  orientation={self.orientation}  "
            f"similarity={self.similarity:.2f}",
            f"  brightness={self.brightness_mean:.1f}  "
            f"bright_ratio={self.bright_ratio*100:.1f}%  "
            f"sharpness={self.sharpness:.0f}",
        ]
        if self.issues:
            lines.append(f"  issues: {'; '.join(self.issues)}")
        if self.suggestions:
            lines.append(f"  suggestions: {'; '.join(self.suggestions)}")
        return "\n".join(lines)


# ---------------- image loading ----------------


def load_gray(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img


# ---------------- rule-based assessment ----------------


def assess_rule(target: np.ndarray, capture: np.ndarray) -> Assessment:
    """OpenCV-only rule assessment."""
    a = Assessment()

    # normalize size
    t = cv2.resize(target, (640, 480)) if target.shape != (480, 640) else target
    c = capture
    if c.shape != (480, 640):
        c = cv2.resize(c, (640, 480))

    # basic metrics
    a.brightness_mean = float(c.mean())
    a.bright_ratio = float((c > 128).mean())
    a.sharpness = float(cv2.Laplacian(c, cv2.CV_64F).var())

    # orientation: try 4 transforms of the capture vs the target
    t_bin = (t > 127).astype(np.float32)
    variants = {
        "normal": c,
        "flip_v": cv2.flip(c, 0),
        "flip_h": cv2.flip(c, 1),
        "rot180": cv2.flip(c, -1),
    }
    best_ori, best_score = "unknown", -1.0
    for name, v in variants.items():
        v_bin = (v > 127).astype(np.float32)
        # normalized correlation
        score = float(np.corrcoef(t_bin.flatten(), v_bin.flatten())[0, 1])
        if score > best_score:
            best_score, best_ori = score, name
    a.orientation = best_ori
    a.similarity = max(0.0, best_score)

    # verdict
    if a.brightness_mean < 10:
        a.issues.append("frame is almost completely black")
        a.suggestions.append("check that content is projected / raise brightness / reduce ambient light")
    if a.bright_ratio < 0.01:
        a.issues.append("bright-area ratio too low")
    if a.sharpness < 50:
        a.issues.append("frame is blurry")
        a.suggestions.append("adjust focus / shorten throw distance")
    if best_ori == "flip_v":
        a.issues.append("image is flipped vertically")
        a.suggestions.append("apply flipud before packing, or send flip command 0x12")
    elif best_ori == "flip_h":
        a.issues.append("image is mirrored horizontally")
        a.suggestions.append("apply fliplr before packing, or send mirror command 0x1E")
    elif best_ori == "rot180":
        a.issues.append("image is rotated 180°")
        a.suggestions.append("apply rot180 before packing, or combine flip commands")
    if a.similarity < 0.3 and a.brightness_mean >= 10:
        a.issues.append("large difference vs target image")

    a.ok = (
        a.brightness_mean >= 10
        and a.bright_ratio >= 0.01
        and a.sharpness >= 50
        and best_ori == "normal"
        and a.similarity >= 0.4
    )
    return a


# ---------------- AI assessment ----------------


def _b64_image(path: str) -> str:
    """Load an image and compress to JPEG base64 (raw PNG often exceeds the 413 limit)."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"cannot read {path}")
    # shrink longest side to 800 and JPEG-compress
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def assess_ai(target_path: str, capture_path: str) -> Assessment:
    """Call Kimi K3 to assess target vs captured image."""
    from irisloop import kimi_client as K

    prompt = """You are a MEMS laser-projection inspector. The first image is the target (what should be projected); the second is a camera capture of the actual projection.
Green scan stripes are capture artifacts. Do not suggest changing focus, FOV, or hardware brightness. Reply with JSON only:
{
  "ok": bool,
  "orientation": "normal|flip_v|flip_h|rot180|unknown",
  "brightness": "too_dark|ok|too_bright",
  "sharpness": "blurry|ok",
  "distortion": "none|keystone|other",
  "issues": ["..."],
  "suggestions": ["..."]
}"""

    resp = K.chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": K.b64_data_url(target_path)},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": K.b64_data_url(capture_path)},
                    },
                ],
            }
        ],
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )
    data = K.extract_json(K.message_text(resp))

    a = Assessment()
    a.ok = bool(data.get("ok", False))
    a.orientation = data.get("orientation", "unknown")
    a.issues = data.get("issues", [])
    a.suggestions = data.get("suggestions", [])
    return a


# ---------------- CLI ----------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="target image")
    ap.add_argument("capture", help="captured image")
    ap.add_argument("--mode", choices=["rule", "ai"], default="rule")
    args = ap.parse_args(argv)

    if args.mode == "ai":
        a = assess_ai(args.target, args.capture)
    else:
        a = assess_rule(load_gray(args.target), load_gray(args.capture))

    print("=== assessment ===")
    print(a.summary())
    print(json.dumps(a.to_dict(), ensure_ascii=False))
    return 0 if a.ok else 1


if __name__ == "__main__":
    sys.exit(main())
