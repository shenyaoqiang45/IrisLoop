"""Kling v3 video generation via Alibaba Cloud Bailian (DashScope).

Preferred writer path for IrisLoop: open Kling on Bailian, call with
``DASHSCOPE_API_KEY`` (never commit the key).

Docs: https://help.aliyun.com/zh/model-studio/kling-video-generation-api-reference/

Env:
    DASHSCOPE_API_KEY          required (Beijing-region Bailian key)
    DASHSCOPE_WORKSPACE_ID     required (Bailian workspace / 业务空间 ID)
Optional:
    DASHSCOPE_KLING_MODEL      default kling/kling-v3-video-generation
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "kling/kling-v3-video-generation"
CREATE_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
# Bailian Kling v3 bills by second; API floor is 3s (1s is not offered).
MIN_DURATION_S = 3
DEFAULT_DURATION_S = MIN_DURATION_S
DEFAULT_FRAME_COUNT = 3


def api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. "
            "Bailian console → API-Key → export as DASHSCOPE_API_KEY."
        )
    return key


def workspace_id() -> str:
    wid = os.environ.get("DASHSCOPE_WORKSPACE_ID") or os.environ.get(
        "BAILIAN_WORKSPACE_ID"
    )
    if not wid:
        raise RuntimeError(
            "DASHSCOPE_WORKSPACE_ID is not set. "
            "Bailian console → workspace / 业务空间 ID "
            "(Beijing region, same region as the API key)."
        )
    return wid.strip()


def default_model() -> str:
    return os.environ.get("DASHSCOPE_KLING_MODEL", DEFAULT_MODEL)


def base_url() -> str:
    return f"https://{workspace_id()}.cn-beijing.maas.aliyuncs.com"


def _request(
    method: str,
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    hdrs = {
        "Authorization": f"Bearer {api_key()}",
        **(headers or {}),
    }
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            if not body:
                return {}
            return json.loads(body)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Bailian Kling HTTP {e.code}: {err}") from e


def create_text_video(
    prompt: str,
    *,
    model: str | None = None,
    duration: int = 5,
    mode: str = "std",
    aspect_ratio: str = "16:9",
    negative_prompt: str | None = None,
    audio: bool = False,
    watermark: bool = False,
) -> str:
    """Submit an async text-to-video job. Returns task_id."""
    if duration < MIN_DURATION_S or duration > 15:
        raise ValueError(
            f"duration must be an integer in [{MIN_DURATION_S}, 15] "
            "(Bailian Kling v3 minimum is 3s; use extract_frames for cheap stills)"
        )
    input_obj: dict[str, Any] = {"prompt": prompt}
    if negative_prompt:
        input_obj["negative_prompt"] = negative_prompt
    payload = {
        "model": model or default_model(),
        "input": input_obj,
        "parameters": {
            "mode": mode,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "audio": audio,
            "watermark": watermark,
        },
    }
    url = f"{base_url()}{CREATE_PATH}"
    resp = _request(
        "POST",
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        },
    )
    task_id = (resp.get("output") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"create did not return task_id: {resp}")
    return str(task_id)


def get_task(task_id: str) -> dict[str, Any]:
    url = f"{base_url()}/api/v1/tasks/{task_id}"
    return _request("GET", url)


def wait_task(
    task_id: str,
    *,
    poll_s: float = 15.0,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Poll until SUCCEEDED / FAILED / CANCELED / UNKNOWN or timeout."""
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = get_task(task_id)
        status = (last.get("output") or {}).get("task_status", "UNKNOWN")
        if status in ("SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"):
            return last
        time.sleep(poll_s)
    raise TimeoutError(f"task {task_id} still running after {timeout_s}s: {last}")


def video_url_from_result(result: dict[str, Any]) -> str:
    out = result.get("output") or {}
    status = out.get("task_status")
    if status != "SUCCEEDED":
        raise RuntimeError(
            f"task not succeeded ({status}): "
            f"{out.get('code')} {out.get('message')} | {result}"
        )
    url = out.get("video_url")
    if not url:
        raise RuntimeError(f"SUCCEEDED but no video_url: {result}")
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
    duration: int = DEFAULT_DURATION_S,
    mode: str = "std",
    negative_prompt: str | None = None,
    model: str | None = None,
) -> Path:
    """One-shot: create → wait → download MP4 (default duration = API minimum 3s)."""
    task_id = create_text_video(
        prompt,
        model=model,
        duration=duration,
        mode=mode,
        negative_prompt=negative_prompt,
    )
    result = wait_task(task_id)
    return download_video(video_url_from_result(result), dest)


def extract_frames(
    video_path: str | Path,
    dest_dir: str | Path,
    *,
    count: int = DEFAULT_FRAME_COUNT,
    prefix: str = "frame",
) -> list[Path]:
    """Sample ``count`` frames evenly in playback order (default 3).

    Always reads sequentially. Seeking with CAP_PROP_POS_FRAMES is unreliable
    on many H.264 MP4s (Kling exports included) and can scramble order.
    """
    import cv2

    if count < 1:
        raise ValueError("count must be >= 1")
    video_path = Path(video_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        # First pass: collect all frames (3s @24fps ≈ 72 — cheap) so indices
        # match real playback order even when FRAME_COUNT is wrong.
        frames_bgr = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_bgr.append(frame)
        n = len(frames_bgr)
        if n == 0:
            raise RuntimeError(f"no frames in {video_path}")
        if reported and abs(reported - n) > 2:
            # keep going; sequential length is source of truth
            pass
        idxs = _even_indices(n, count)
        out: list[Path] = []
        for i, idx in enumerate(idxs):
            path = dest_dir / f"{prefix}_{i:02d}.jpg"
            cv2.imencode(".jpg", frames_bgr[idx])[1].tofile(str(path))
            out.append(path)
        return out
    finally:
        cap.release()


def _even_indices(n: int, count: int) -> list[int]:
    if count == 1:
        return [n // 2]
    if count >= n:
        return list(range(n))
    # include endpoints; space the rest evenly
    return [int(round(i * (n - 1) / (count - 1))) for i in range(count)]


def generate_min_clip_and_frames(
    prompt: str,
    out_dir: str | Path,
    *,
    duration: int = DEFAULT_DURATION_S,
    frame_count: int = DEFAULT_FRAME_COUNT,
    mode: str = "std",
    negative_prompt: str | None = None,
    model: str | None = None,
) -> tuple[Path, list[Path]]:
    """Cheapest writer probe: min-length clip + a few stills for the director."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "kling.mp4"
    generate_and_download(
        prompt,
        mp4,
        duration=duration,
        mode=mode,
        negative_prompt=negative_prompt,
        model=model,
    )
    frames = extract_frames(mp4, out_dir / "frames", count=frame_count)
    return mp4, frames
