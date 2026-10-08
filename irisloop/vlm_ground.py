"""Grounding prompt and reply parsing for Qwen3-VL region picking.

Qwen3-VL answers grounding queries with 0-1000 normalized coordinates, e.g.
``[{"bbox_2d": [217, 112, 920, 956], "label": "cat"}]``. This module turns a
free-form user intent ("把光投到海报人物的头显上") plus a wall photo into camera
pixel geometry usable by ``irisloop.target_path``.
"""

from __future__ import annotations

import json
from typing import Any

GROUND_PROMPT = """这张图片是相机实拍的墙面，橙色四边形标出激光可投射的区域。
用户意图：{intent}
请找出意图所指的墙面目标，只输出一个 JSON 数组（不要输出任何其他文字）：
[{{"label": "目标的简短描述", "bbox_2d": [x1, y1, x2, y2], "outline": [[x, y], ...]}}]
要求：
- 所有坐标都是 0-1000 的归一化值（分别相对于图片宽和高）
- bbox_2d 必须给出，是目标的外接矩形
- outline 是沿目标轮廓的多边形点列（3-8 个点）；若目标就是矩形区域可与 bbox 相同
- 目标必须完整位于橙色四边形内，尽量贴合意图所指的对象"""


def extract_json_value(text: str) -> Any:
    """Pull the first JSON value out of a reply (tolerates prose and fences)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in reply: {text[:400]}")


def _norm_to_px(v: float, size: int) -> float:
    """0-1000 normalized coordinate -> pixel, clamped to the image."""
    return min(max(v / 1000.0 * size, 0.0), float(size - 1))


def regions_from_reply(text: str, width: int, height: int) -> list[dict]:
    """Parse a grounding reply into camera-pixel regions.

    Returns [{"label": str, "bbox": (x0, y0, x1, y1),
              "polygon": [(x, y), ...] | None}], best region first.
    """
    value = extract_json_value(text)
    if isinstance(value, dict):
        value = value.get("targets") or value.get("regions") or [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"expected a non-empty JSON array of regions: {text[:400]}")

    regions: list[dict] = []
    for item in value:
        if not isinstance(item, dict) or "bbox_2d" not in item:
            continue
        box = item["bbox_2d"]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x0 = _norm_to_px(float(box[0]), width)
        y0 = _norm_to_px(float(box[1]), height)
        x1 = _norm_to_px(float(box[2]), width)
        y1 = _norm_to_px(float(box[3]), height)
        if x1 <= x0 or y1 <= y0:
            continue
        polygon = None
        outline = item.get("outline")
        if isinstance(outline, list) and len(outline) >= 3:
            pts = []
            for p in outline:
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    pts.append((_norm_to_px(float(p[0]), width),
                                _norm_to_px(float(p[1]), height)))
            if len(pts) >= 3:
                polygon = pts
        regions.append({
            "label": str(item.get("label", "")),
            "bbox": (x0, y0, x1, y1),
            "polygon": polygon,
        })
    if not regions:
        raise ValueError(f"no usable region in reply: {text[:400]}")
    return regions


def refine_outline(frame, bbox, margin: int = 8, grabcut_iter: int = 5,
                   epsilon_ratio: float = 0.008) -> list[tuple[float, float]]:
    """Refine a coarse VLM bbox into an object silhouette polygon (GrabCut).

    Qwen3-VL at wall-photo resolution returns bbox-corner "outlines" rather
    than true contours, so the silhouette is extracted locally instead: crop
    around the bbox, GrabCut foreground, largest contour, simplified polygon
    in full-frame camera coordinates.
    """
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0, y0 = max(0, int(x0) - margin), max(0, int(y0) - margin)
    x1, y1 = min(w, int(x1) + margin + 1), min(h, int(y1) + margin + 1)
    crop = frame[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if cw < 4 * margin or ch < 4 * margin:
        raise ValueError(f"bbox too small to refine: {bbox}")

    mask = np.zeros((ch, cw), np.uint8)
    rect = (margin, margin, cw - 2 * margin, ch - 2 * margin)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(crop, mask, rect, bgd, fgd, grabcut_iter, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("refine found no foreground in the bbox")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        raise ValueError("refined foreground is too small")
    eps = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        raise ValueError("refined contour degenerated")
    return [(float(x + x0), float(y + y0)) for x, y in approx]
