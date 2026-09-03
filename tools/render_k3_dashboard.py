"""Render K3 director phase-check dashboard PNG for X posts."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmark" / "groundtruth" / "k3_report.json"
# X / social post images live under doc/
OUT = ROOT / "doc" / "k3-director-dashboard.png"
DATA_PATH_LABEL = r"IrisLoop\benchmark\groundtruth"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rs = report["results"]
    n = len(rs)
    ok = sum(1 for r in rs if r.get("recognizable"))
    confs = [float(r.get("confidence", 0)) for r in rs]
    avg_c = sum(confs) / n
    min_c = min(confs)
    max_c = max(confs)
    none_act = sum(
        1 for r in rs if set(r.get("content_actions") or ["none"]) == {"none"}
    )
    action_n = n - none_act

    W, H = 1600, 900
    bg = (13, 17, 16)
    card = (22, 28, 26)
    green = (61, 255, 122)
    white = (240, 244, 242)
    muted = (140, 150, 145)
    yellow = (255, 210, 90)

    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    f_title = font(54, True)
    f_sub = font(26)
    f_kpi = font(64, True)
    f_kpi_lbl = font(22)
    f_card_t = font(28, True)
    f_body = font(22)
    f_small = font(18)
    f_tiny = font(16)

    d.text((64, 40), "IrisLoop", font=f_title, fill=white)
    tw = d.textlength("IrisLoop", font=f_title)
    d.text((64 + tw + 18, 58), "· K3 director phase check", font=f_sub, fill=green)
    d.text(
        (64, 110),
        f"Data: {DATA_PATH_LABEL}  ·  wall + USB camera  ·  model: kimi-k3  ·  2026-09-02",
        font=f_small,
        fill=muted,
    )

    kpis = [
        (f"{ok}/{n}", "Recognizable", green),
        (f"{avg_c:.2f}", "Avg confidence", green),
        (f"{min_c:.2f}–{max_c:.2f}", "Confidence range", white),
        (f"{action_n}/{n}", "Need content fix", yellow),
    ]
    kx0, ky0, kw, kh, gap = 64, 160, 360, 150, 24
    for i, (val, lbl, col) in enumerate(kpis):
        x = kx0 + i * (kw + gap)
        d.rounded_rectangle((x, ky0, x + kw, ky0 + kh), radius=18, fill=card)
        d.text((x + 28, ky0 + 32), val, font=f_kpi, fill=col)
        d.text((x + 28, ky0 + 108), lbl, font=f_kpi_lbl, fill=muted)

    d.rounded_rectangle((64, 340, 780, 620), radius=18, fill=card)
    d.text((92, 368), "Phase verdict", font=f_card_t, fill=green)
    left = [
        ('Kimi K3 can own "good enough".', white),
        ("", muted),
        ("· Sees real green-laser projection", white),
        ("· Treats scan stripes as capture noise", white),
        ("· Does not ask to retune hardware", white),
        ("· Emits content actions when needed", white),
        ("  (thicken / simplify / drop bg)", muted),
    ]
    y = 420
    for text, col in left:
        if text:
            d.text((92, y), text, font=f_body, fill=col)
        y += 32

    d.rounded_rectangle((820, 340, 1536, 620), radius=18, fill=card)
    d.text((848, 368), "What this phase is / is not", font=f_card_t, fill=white)
    d.text((848, 420), "PASS — still understanding", font=f_body, fill=green)
    d.text(
        (848, 456),
        f"{DATA_PATH_LABEL} · 20 groups · still_01",
        font=f_small,
        fill=muted,
    )
    d.text((848, 500), "NEXT — hand notes to Kling", font=f_body, fill=yellow)
    d.text(
        (848, 536),
        "Not yet: multi-round video rewrite loop",
        font=f_small,
        fill=muted,
    )
    d.text(
        (848, 572),
        "Director seat validated before writer seat",
        font=f_small,
        fill=muted,
    )

    d.rounded_rectangle((64, 648, 1536, 840), radius=18, fill=card)
    d.text((92, 672), "Role split (ADR)", font=f_card_t, fill=white)
    d.rounded_rectangle((92, 730, 740, 810), radius=14, fill=(18, 40, 28))
    d.text((120, 748), "Kimi K3  =  director", font=f_body, fill=green)
    d.text(
        (120, 780),
        "watches the real beam · owns pass/fail",
        font=f_small,
        fill=muted,
    )
    d.rounded_rectangle((780, 730, 1508, 810), radius=14, fill=(40, 36, 18))
    d.text((808, 748), "Kling  =  writer", font=f_body, fill=yellow)
    d.text(
        (808, 780),
        "makes the next clip from director notes",
        font=f_small,
        fill=muted,
    )

    d.text(
        (64, 860),
        f"{DATA_PATH_LABEL}\\k3_report.json  ·  github.com/shenyaoqiang45/IrisLoop",
        font=f_tiny,
        fill=muted,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
