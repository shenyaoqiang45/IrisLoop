"""Wan 2.2 T2V via SiliconFlow — preferred IrisLoop writer (cost).

Model: ``Wan-AI/Wan2.2-T2V-A14B`` (~5s clip, billed per video).
Auth: ``SILICONFLOW_API_KEY`` only (never commit).

Docs:
  https://api-docs.siliconflow.cn/docs/api/video-submit-post
  https://api-docs.siliconflow.cn/docs/api/video-status-post
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from irisloop.video_frames import DEFAULT_FRAME_COUNT, extract_frames

DEFAULT_MODEL = "Wan-AI/Wan2.2-T2V-A14B"
# Official submit enum only (api-docs.siliconflow.cn): 1280x720 | 720x1280 | 960x960
# No documented 480p size — use 1280x720 for 16:9 T2V. Override via SILICONFLOW_WAN_SIZE.
ALLOWED_IMAGE_SIZES = ("1280x720", "720x1280", "960x960")
DEFAULT_IMAGE_SIZE = "1280x720"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
# Wan2.2 on SiliconFlow is a fixed-length clip (~5s), not 1s/3s knobs.
NOMINAL_DURATION_S = 5


def api_key() -> str:
    key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SF_API_KEY")
    if not key:
        raise RuntimeError(
            "SILICONFLOW_API_KEY is not set. "
            "https://cloud.siliconflow.cn/account/ak → export SILICONFLOW_API_KEY."
        )
    return key


def base_url() -> str:
    return os.environ.get("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def default_model() -> str:
    return os.environ.get("SILICONFLOW_WAN_MODEL", DEFAULT_MODEL)


def default_image_size() -> str:
    size = os.environ.get("SILICONFLOW_WAN_SIZE", DEFAULT_IMAGE_SIZE)
    if size not in ALLOWED_IMAGE_SIZES:
        raise ValueError(
            f"image_size={size!r} not in official enum {ALLOWED_IMAGE_SIZES} "
            "(see https://api-docs.siliconflow.cn/docs/api/video-submit-post)"
        )
    return size


def _request(
    method: str,
    path: str,
    *,
    data: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    url = f"{base_url()}{path}"
    body = None if data is None else json.dumps(data).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SiliconFlow HTTP {e.code}: {err}") from e


def submit_text_video(
    prompt: str,
    *,
    model: str | None = None,
    image_size: str | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
) -> str:
    """Submit T2V job; returns requestId for polling."""
    size = image_size or default_image_size()
    if size not in ALLOWED_IMAGE_SIZES:
        raise ValueError(
            f"image_size={size!r} not in official enum {ALLOWED_IMAGE_SIZES}"
        )
    payload: dict[str, Any] = {
        "model": model or default_model(),
        "prompt": prompt,
        "image_size": size,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["seed"] = seed
    resp = _request("POST", "/video/submit", data=payload)
    rid = resp.get("requestId") or resp.get("request_id")
    if not rid:
        raise RuntimeError(f"submit did not return requestId: {resp}")
    return str(rid)


def get_status(request_id: str) -> dict[str, Any]:
    return _request("POST", "/video/status", data={"requestId": request_id})


def wait_video(
    request_id: str,
    *,
    poll_s: float = 10.0,
    timeout_s: float = 900.0,
    quiet: bool = False,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    t0 = time.time()
    while time.time() < deadline:
        last = get_status(request_id)
        status = last.get("status") or last.get("Status")
        if not quiet:
            print(
                f"  [{time.time() - t0:5.0f}s] status={status} requestId={request_id}",
                flush=True,
            )
        if status == "Succeed":
            return last
        if status == "Failed":
            raise RuntimeError(f"Wan generation failed: {last.get('reason')} | {last}")
        time.sleep(poll_s)
    raise TimeoutError(f"request {request_id} still running after {timeout_s}s: {last}")


def video_url_from_status(result: dict[str, Any]) -> str:
    results = result.get("results") or {}
    videos = results.get("videos") or []
    if not videos:
        raise RuntimeError(f"Succeed but no videos: {result}")
    url = videos[0].get("url")
    if not url:
        raise RuntimeError(f"Succeed but empty url: {result}")
    return str(url)


def download_video(url: str, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=300.0) as resp:
        dest.write_bytes(resp.read())
    return dest


def generate_and_download(
    prompt: str,
    dest: str | Path,
    *,
    image_size: str | None = None,
    negative_prompt: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> Path:
    size = image_size or default_image_size()
    print(f"  submit model={model or default_model()} size={size}", flush=True)
    rid = submit_text_video(
        prompt,
        model=model,
        image_size=size,
        negative_prompt=negative_prompt,
        seed=seed,
    )
    print(f"  requestId={rid} (Wan A14B often queues several minutes)", flush=True)
    result = wait_video(rid)
    print("  download…", flush=True)
    return download_video(video_url_from_status(result), dest)


def generate_min_clip_and_frames(
    prompt: str,
    out_dir: str | Path,
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
    image_size: str | None = None,
    negative_prompt: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> tuple[Path, list[Path]]:
    """Writer probe: one Wan clip (~5s) + evenly sampled stills."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "wan.mp4"
    generate_and_download(
        prompt,
        mp4,
        image_size=image_size or default_image_size(),
        negative_prompt=negative_prompt,
        model=model,
        seed=seed,
    )
    frames = extract_frames(mp4, out_dir / "frames", count=frame_count)
    return mp4, frames
