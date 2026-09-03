"""Run captured-still understanding cases with Kimi K3 (one still_01*.jpg per group).

Usage:
    set MOONSHOT_API_KEY=sk-...
    python tools/vu_k3_eval.py --root benchmark/groundtruth
    python tools/vu_k3_eval.py --root benchmark/groundtruth --groups 1 2 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from irisloop import protocol as P
from irisloop import kimi_client as K

SYSTEM = (
    "You are a visual-understanding QA inspector for MEMS green laser projection. "
    "The image is a USB-camera photo of a projection on a physical wall. "
    "Green scan stripes, slight underexposure, and color shift are capture artifacts, not content failure. "
    "Do not suggest changing focus, FOV, hardware brightness, or device parameters; only judge whether the projected content is recognizable. "
    "Output JSON only, no Markdown."
)

PROMPT = """Analyze this MEMS laser-projection still JPG. Material group ID={gid} ({gname}).

Return JSON only:
{{
  "group": {gid},
  "subject": "the subject you recognize (short English phrase, or unknown)",
  "recognizable": true/false,
  "confidence": 0.0 to 1.0,
  "issues": ["too thin|too fragmented|low contrast|subject unclear|artifact interference|other..."],
  "content_actions": ["thicken_strokes|simplify_silhouette|drop_background|recenter|none"],
  "summary": "one-sentence English conclusion"
}}"""


def find_still01(group_dir: Path) -> Path | None:
    hits = sorted(group_dir.glob("still_01*.jpg"))
    if hits:
        return hits[0]
    # Compatible with project_and_capture frame_01_*.jpg
    hits = sorted(group_dir.glob("frame_01_*.jpg"))
    return hits[0] if hits else None


def eval_still(gid: int, gname: str, still: Path, *, effort: str) -> dict:
    resp = K.chat(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": K.b64_data_url(still, max_side=960, jpeg_quality=75)
                        },
                    },
                    {
                        "type": "text",
                        "text": PROMPT.format(gid=gid, gname=gname),
                    },
                ],
            },
        ],
        reasoning_effort=effort,
        response_format={"type": "json_object"},
    )
    text = K.message_text(resp)
    data = K.extract_json(text)
    data["_raw"] = text
    data["_usage"] = resp.get("usage")
    data["_mode"] = "still_01"
    data["_file"] = str(still).replace("\\", "/")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="vu_testset directory")
    ap.add_argument("--groups", type=int, nargs="*", default=None)
    ap.add_argument("--effort", choices=["low", "high", "max"], default="low")
    ap.add_argument("--sleep", type=float, default=2.5,
                    help="Wait between groups (>=2 recommended when account concurrency=1)")
    ap.add_argument("--skip-ok", action="store_true",
                    help="Skip groups that already have a successful k3_eval.json")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"Directory does not exist: {root}")
        return 2

    print("=== Kimi K3 single-frame understanding eval (still_01) ===")
    print(f"  base={K.base_url()}  model={K.DEFAULT_MODEL}  effort={args.effort}")
    print(f"  root={root}")

    results: list[dict] = []
    for gdir in sorted(root.glob("group_*")):
        try:
            gid = int(gdir.name.split("_")[1])
        except Exception:
            continue
        if args.groups is not None and gid not in args.groups:
            continue

        gname = P.group_name(gid)
        prev = gdir / "k3_eval.json"
        if args.skip_ok and prev.exists():
            try:
                old = json.loads(prev.read_text(encoding="utf-8"))
                if old.get("ok") and "recognizable" in old:
                    print(f"\n--- group {gid:02d} ({gname}) skip already-ok ---")
                    results.append(old)
                    continue
            except Exception:
                pass

        still = find_still01(gdir)
        print(f"\n--- group {gid:02d} ({gname}) ---")
        if still is None:
            print("  skip: no still_01*.jpg / frame_01_*.jpg")
            results.append({"group": gid, "name": gname, "ok": False, "error": "no still_01"})
            continue

        print(f"  file={still.name}")
        t0 = time.perf_counter()
        try:
            data = eval_still(gid, gname, still, effort=args.effort)
            data["ok"] = True
            data["elapsed_s"] = round(time.perf_counter() - t0, 2)
            print(
                f"  subject={data.get('subject')}  "
                f"recognizable={data.get('recognizable')}  "
                f"conf={data.get('confidence')}  "
                f"{data.get('summary')}"
            )
        except Exception as e:
            data = {
                "group": gid,
                "name": gname,
                "ok": False,
                "error": str(e),
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "_file": str(still).replace("\\", "/"),
            }
            print(f"  [error] {e}")

        (gdir / "k3_eval.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append(data)
        time.sleep(args.sleep)

    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": K.DEFAULT_MODEL,
        "base_url": K.base_url(),
        "root": str(root).replace("\\", "/"),
        "mode": "still_01",
        "results": results,
        "recognizable": [r.get("group") for r in results if r.get("recognizable") is True],
        "unrecognizable": [r.get("group") for r in results if r.get("recognizable") is False],
        "errors": [r.get("group") for r in results if not r.get("ok")],
    }
    report_path = root / "k3_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"  report: {report_path}")
    for r in results:
        if r.get("ok"):
            print(
                f"  group {r.get('group'):02d}  "
                f"{'OK' if r.get('recognizable') else 'NO'}  "
                f"{r.get('subject')}  | {r.get('summary')}"
            )
        else:
            print(f"  group {r.get('group'):02d}  ERR  {r.get('error')}")
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
