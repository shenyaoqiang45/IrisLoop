"""One IrisLoop round: director → writer → project → capture → director review.

Kimi K3 is the silhouette + creative director: it authors the content, then
looks at the real beam and decides whether it loves this piece. Wan2.2-T2V-Plus
on Bailian is only the writer. Frames go to IrisGreen group 1; the camera
records the beam; K3 reviews as an artist, not a recognizability checker.

Requires env (never commit keys):
    MOONSHOT_API_KEY or KIMI_API_KEY
    DASHSCOPE_API_KEY
    DASHSCOPE_WORKSPACE_ID

Usage:
    python tools/loop_round1.py "dramatic volcanic eruption"
    python tools/loop_round1.py "dramatic volcanic eruption" --rounds 3
    python tools/loop_round1.py "…" --skip-project   # stop after Wan frames (no BLE)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from irisloop import kimi_client as K
from irisloop import protocol as P
from irisloop import wan_bailian as W
from irisloop.ble_client import IrisBleClient
from irisloop.image_pack import binarize, build_stream, describe
from irisloop.projector import CHAR_FILE_DATA, CHAR_FILE_END, CHAR_FILE_START
from irisloop.video_frames import extract_frames

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"
ATT_HEADER_BYTES = 3
DEFAULT_ATT_MTU = 23
FILE_NAME_FIELD_MIN_BYTES = 16
START_DELAY_S = 0.1
PACKET_DELAY_S = 0.03
INTER_FILE_DELAY_S = 1.0

DIRECTOR_BRIEF_SYSTEM = (
    "You are IrisLoop's director: a silhouette master and a creative master. "
    "The medium is a green MEMS laser (640x480, 1bpp). You do not merely sanitize "
    "the user's wish — you invent a graphic idea: one iconic cut-paper / shadow-play "
    "figure, a readable gesture, a beat of motion that feels authored. "
    "Craft is non-negotiable (thick masses, black void, no texture, no text) because "
    "that is how silhouettes sing, not because you are a QA bot. "
    "Reply JSON only. wan_prompt must be English for Wan2.2."
)

DIRECTOR_BRIEF_USER = """User wish (raw intent, not the artwork): {wish}

Author a silhouette piece. Return JSON:
{{
  "concept": "one-sentence creative idea (what this piece is about)",
  "subject": "short English subject",
  "wan_prompt": "English T2V prompt: bold 1bpp silhouette, black void, thick masses, authored motion",
  "negative_prompt": "English negative prompt",
  "notes": "why this cut will feel like a finished graphic, not a stock clip"
}}"""

DIRECTOR_REVIEW_SYSTEM = (
    "You are IrisLoop's director: silhouette master + creative master. "
    "The images are USB-camera photos of your piece on a green MEMS laser. "
    "Green scan stripes and mild underexposure are capture artifacts, not content failure. "
    "Do not suggest changing focus, FOV, or hardware. "
    "recognizable is a factual check. pass is whether you love this silhouette "
    "as a work you would sign — graphic force, gesture, mass, drama. "
    "If you would only say 'a human can tell what it is', pass must be false. "
    "JSON only."
)

DIRECTOR_REVIEW_USER = """User wish: {wish}
Writer prompt used: {wan_prompt}

Look at the projection still(s) as the author of this piece. Return JSON:
{{
  "subject_seen": "what you see, or unknown",
  "recognizable": true/false,
  "confidence": 0.0-1.0,
  "loved": true/false,
  "issues": ["too thin|too fragmented|low contrast|subject unclear|gesture weak|generic|artifact interference|other"],
  "content_actions": ["thicken_strokes|simplify_silhouette|drop_background|recenter|slow_motion|bolder_gesture|none"],
  "pass": true/false,
  "summary": "one English sentence: do you love it, and why or why not",
  "next_intent": "round-2 creative goal in one sentence, or empty if pass",
  "next_wan_prompt": "new English T2V prompt that pursues next_intent; use same only if pass is true"
}}"""


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _next_writer_prompt(review: dict) -> str | None:
    raw = (review.get("next_wan_prompt") or "").strip()
    if not raw or raw.lower() in ("same", "unchanged", "n/a"):
        return None
    return raw


def brief_from_review(prev_brief: dict, review: dict) -> dict | None:
    nxt = _next_writer_prompt(review)
    if not nxt:
        return None
    return {
        "concept": review.get("next_intent") or prev_brief.get("concept"),
        "subject": prev_brief.get("subject"),
        "wan_prompt": nxt,
        "negative_prompt": prev_brief.get("negative_prompt"),
        "notes": "iteration: director next_wan_prompt from previous review",
        "_from_review": True,
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def director_brief(wish: str, *, effort: str = "low") -> dict:
    print("\n=== [1/5] DIRECTOR brief (Kimi K3) ===", flush=True)
    resp = K.chat(
        [
            {"role": "system", "content": DIRECTOR_BRIEF_SYSTEM},
            {
                "role": "user",
                "content": DIRECTOR_BRIEF_USER.format(wish=wish),
            },
        ],
        reasoning_effort=effort,
        response_format={"type": "json_object"},
    )
    data = K.extract_json(K.message_text(resp))
    data["_usage"] = resp.get("usage")
    if data.get("concept"):
        print(f"  concept : {data.get('concept')}", flush=True)
    print(f"  subject : {data.get('subject')}", flush=True)
    print(f"  prompt  : {data.get('wan_prompt')}", flush=True)
    return data


def writer_wan(brief: dict, out_dir: Path, *, frames: int, size: str) -> tuple[Path, list[Path]]:
    print("\n=== [2/5] WRITER Bailian Wan2.2-T2V-Plus ===", flush=True)
    prompt = brief.get("wan_prompt") or brief.get("subject")
    if not prompt:
        raise RuntimeError(f"director brief missing wan_prompt: {brief}")
    neg = brief.get("negative_prompt") or None
    writer_dir = out_dir / "writer"
    mp4, paths = W.generate_min_clip_and_frames(
        prompt,
        writer_dir,
        frame_count=frames,
        size=size,
        negative_prompt=neg,
    )
    # ensure numbered frames for push_group sort
    frame_dir = writer_dir / "frames"
    if not paths:
        paths = extract_frames(mp4, frame_dir, count=frames)
    print(f"  video   : {mp4}", flush=True)
    print(f"  frames  : {len(paths)} under {frame_dir}", flush=True)
    return mp4, paths


def build_name_packet(file_size: int, file_name: str) -> bytes:
    nb = file_name.encode("utf-8")
    field = max(FILE_NAME_FIELD_MIN_BYTES, len(nb))
    return file_size.to_bytes(4, "big") + nb.ljust(field, b"\x00")


def load_bw(path: Path):
    import cv2
    import numpy as np

    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read {path}")
    if img.shape != (480, 640):
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    return binarize(img, threshold=127)


async def push_one(client, file_name: str, stream: bytes) -> None:
    packet = build_name_packet(len(stream), file_name)
    mtu = getattr(client, "mtu_size", DEFAULT_ATT_MTU) or DEFAULT_ATT_MTU
    chunk = max(1, mtu - ATT_HEADER_BYTES)
    n = (len(stream) + chunk - 1) // chunk
    await client.write_gatt_char(CHAR_FILE_START, packet, response=True)
    await asyncio.sleep(START_DELAY_S)
    for i in range(n):
        await client.write_gatt_char(
            CHAR_FILE_DATA, stream[i * chunk : (i + 1) * chunk], response=False
        )
        await asyncio.sleep(PACKET_DELAY_S)
    await client.write_gatt_char(CHAR_FILE_END, packet, response=True)


async def project_and_capture(
    frame_paths: list[Path],
    capture_dir: Path,
    *,
    address: str,
    group: int,
    seconds: float,
    camera: int,
    exposure: float | None,
    interval_100ms: int,
    n_stills: int,
) -> dict:
    print("\n=== [3/5] PROJECT push group + play ===", flush=True)
    import importlib.util

    _pac = Path(__file__).resolve().parent / "project_and_capture.py"
    _spec = importlib.util.spec_from_file_location("project_and_capture", _pac)
    assert _spec and _spec.loader
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    run_group = _mod.run_group

    cli = IrisBleClient(address)
    await cli.connect()
    print(f"  connected mtu={cli.info.mtu}", flush=True)
    try:
        print(f"  delete group {group}…", flush=True)
        r = await cli.send_command(P.cmd_delete_group(group))
        print(f"  delete: {'ok' if r.ok else r.error}", flush=True)
        await asyncio.sleep(0.5)
        client = cli.client
        assert client is not None
        for i, path in enumerate(frame_paths, 1):
            bw = load_bw(path)
            stream = build_stream(bw)
            name = f"{group}_{i}.bmp"
            print(f"  [{i}/{len(frame_paths)}] {path.name} -> {name}  {describe(bw)}", flush=True)
            await push_one(client, name, stream)
            await asyncio.sleep(INTER_FILE_DELAY_S)
    finally:
        await cli.disconnect()

    print("\n=== [4/5] CAPTURE real projection ===", flush=True)
    capture_dir.mkdir(parents=True, exist_ok=True)
    result = await run_group(
        address,
        group,
        seconds,
        camera,
        exposure,
        False,
        interval_100ms,
        n_stills=n_stills,
        jpeg_quality=92,
        out_dir=str(capture_dir),
    )
    return result


def pick_review_stills(capture_dir: Path, limit: int = 3) -> list[Path]:
    stills = sorted(capture_dir.glob("frame_*.jpg"))
    if not stills:
        return []
    if len(stills) <= limit:
        return stills
    # evenly pick across capture
    idxs = [int(round(i * (len(stills) - 1) / (limit - 1))) for i in range(limit)]
    return [stills[i] for i in idxs]


def director_review(
    wish: str,
    wan_prompt: str,
    stills: list[Path],
    *,
    effort: str = "low",
) -> dict:
    print("\n=== [5/5] DIRECTOR review (Kimi K3 on capture) ===", flush=True)
    if not stills:
        return {
            "pass": False,
            "recognizable": False,
            "summary": "no capture stills",
            "error": "no_stills",
        }
    content: list = [
        {"type": "text", "text": DIRECTOR_REVIEW_USER.format(wish=wish, wan_prompt=wan_prompt)}
    ]
    for p in stills:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": K.b64_data_url(p, max_side=960, jpeg_quality=75)},
            }
        )
        print(f"  still: {p.name}", flush=True)
    resp = K.chat(
        [
            {"role": "system", "content": DIRECTOR_REVIEW_SYSTEM},
            {"role": "user", "content": content},
        ],
        reasoning_effort=effort,
        response_format={"type": "json_object"},
    )
    data = K.extract_json(K.message_text(resp))
    data["_usage"] = resp.get("usage")
    data["_stills"] = [str(p).replace("\\", "/") for p in stills]
    print(
        f"  recognizable={data.get('recognizable')}  loved={data.get('loved')}  "
        f"pass={data.get('pass')}  conf={data.get('confidence')}",
        flush=True,
    )
    print(f"  summary: {data.get('summary')}", flush=True)
    if data.get("next_intent"):
        print(f"  intent : {data.get('next_intent')}", flush=True)
    if data.get("next_wan_prompt"):
        print(f"  next   : {data.get('next_wan_prompt')}", flush=True)
    return data


async def run_one_round(
    args: argparse.Namespace,
    *,
    wish: str,
    brief: dict,
    out: Path,
    round_index: int,
) -> dict:
    report: dict = {
        "wish": wish,
        "round": round_index,
        "out": str(out).replace("\\", "/"),
        "model_writer": W.default_model(),
        "model_director": K.DEFAULT_MODEL,
        "size": args.size,
        "brief": brief,
    }
    _write_json(out / "01_director_brief.json", brief)

    mp4, frames = writer_wan(brief, out, frames=args.frames, size=args.size)
    report["writer"] = {
        "mp4": str(mp4).replace("\\", "/"),
        "frames": [str(p).replace("\\", "/") for p in frames],
    }
    _write_json(out / "02_writer.json", report["writer"])

    if args.skip_project:
        report["skipped"] = "project_capture_review"
        _write_json(out / "round_report.json", report)
        print("\n--skip-project: stopped after writer frames", flush=True)
        return report

    capture_dir = out / "capture"
    cap = await project_and_capture(
        frames,
        capture_dir,
        address=args.address,
        group=args.group,
        seconds=args.seconds,
        camera=args.camera,
        exposure=args.exposure,
        interval_100ms=args.interval,
        n_stills=args.n_stills,
    )
    report["capture"] = cap
    _write_json(out / "03_capture.json", cap)

    stills = pick_review_stills(capture_dir, limit=args.review_stills)
    review = director_review(
        wish,
        brief.get("wan_prompt", ""),
        stills,
        effort=args.effort,
    )
    report["review"] = review
    _write_json(out / "04_director_review.json", review)
    _write_json(out / "round_report.json", report)

    print("\n=== ROUND RESULT ===", flush=True)
    print(
        f"  pass={review.get('pass')} (loves the silhouette)  "
        f"loved={review.get('loved')}  recognizable={review.get('recognizable')}",
        flush=True,
    )
    print(f"  report: {out / 'round_report.json'}", flush=True)
    return report


async def async_main(args: argparse.Namespace) -> int:
    wish = args.wish.strip()
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    rounds = max(1, args.rounds)
    session = Path(args.out) if args.out else Path("captures") / "loop_rounds" / tag
    if rounds == 1 and not args.out:
        session = Path("captures") / "loop_round1" / tag
    session.mkdir(parents=True, exist_ok=True)

    print("=== IrisLoop loop ===", flush=True)
    print(f"  wish   : {wish}", flush=True)
    print(f"  rounds : {rounds} (stop early if director loves the cut)", flush=True)
    print(f"  out    : {session}", flush=True)

    if args.brief_json:
        brief_path = Path(args.brief_json)
        brief = json.loads(brief_path.read_text(encoding="utf-8"))
        print(f"\n=== [1/5] DIRECTOR brief (reused {brief_path}) ===", flush=True)
        print(f"  subject : {brief.get('subject')}", flush=True)
        print(f"  prompt  : {brief.get('wan_prompt')}", flush=True)
    else:
        brief = director_brief(wish, effort=args.effort)

    session_report: dict = {
        "wish": wish,
        "started": tag,
        "out": str(session).replace("\\", "/"),
        "max_rounds": rounds,
        "rounds": [],
        "stopped": None,
    }
    last_pass = False

    for i in range(1, rounds + 1):
        round_dir = session / f"r{i:02d}" if rounds > 1 else session
        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n######## ROUND {i}/{rounds} ########", flush=True)
        report = await run_one_round(
            args, wish=wish, brief=brief, out=round_dir, round_index=i
        )
        session_report["rounds"].append(
            {
                "round": i,
                "out": report.get("out"),
                "pass": (report.get("review") or {}).get("pass"),
                "loved": (report.get("review") or {}).get("loved"),
                "next_intent": (report.get("review") or {}).get("next_intent"),
            }
        )
        review = report.get("review") or {}
        last_pass = _truthy(review.get("pass")) or _truthy(review.get("loved"))
        if args.skip_project:
            session_report["stopped"] = "skip_project"
            break
        if last_pass:
            session_report["stopped"] = "director_loved"
            break
        if i >= rounds:
            session_report["stopped"] = "max_rounds"
            break
        nxt = brief_from_review(brief, review)
        if nxt is None:
            print("\n=== DIRECTOR re-author (no usable next_wan_prompt) ===", flush=True)
            brief = director_brief(wish, effort=args.effort)
        else:
            brief = nxt
            print("\n=== next writer prompt from director review ===", flush=True)
            print(f"  intent : {brief.get('concept')}", flush=True)
            print(f"  prompt : {brief.get('wan_prompt')}", flush=True)

    _write_json(session / "session_report.json", session_report)
    print("\n=== SESSION ===", flush=True)
    print(f"  stopped={session_report['stopped']}  last_pass={last_pass}", flush=True)
    print(f"  report: {session / 'session_report.json'}", flush=True)
    return 0 if last_pass else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wish", help='what you want to see, e.g. "dramatic volcanic eruption"')
    ap.add_argument("--frames", type=int, default=10, help="frames to extract / push (max 10 for group 1)")
    ap.add_argument("--size", default="832*480", help="Wan Bailian size (480P default)")
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--exposure", type=float, default=-7.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--interval", type=int, default=3, help="play interval units of 100ms")
    ap.add_argument("--n-stills", type=int, default=8)
    ap.add_argument("--review-stills", type=int, default=3)
    ap.add_argument("--effort", choices=["low", "high", "max"], default="low")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--skip-project",
        action="store_true",
        help="only director brief + Wan frames (no BLE / camera / review)",
    )
    ap.add_argument(
        "--brief-json",
        default=None,
        help="reuse an existing 01_director_brief.json (skip K3 brief call)",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="max director→writer→project cycles; stop early if pass",
    )
    args = ap.parse_args(argv)
    args.frames = max(1, min(10, args.frames))
    args.rounds = max(1, min(8, args.rounds))
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
