"""Wan text-to-video via Alibaba Cloud Bailian (DashScope).

Preferred cheap writer on Bailian:
  model ``wan2.2-t2v-plus``, 480P size ``832*480`` (official enum).

Docs: https://help.aliyun.com/zh/model-studio/legacy-wan-text-to-video-api-reference

Env:
    DASHSCOPE_API_KEY
    DASHSCOPE_WORKSPACE_ID   Beijing workspace / 业务空间 ID
Optional:
    DASHSCOPE_WAN_MODEL      default wan2.2-t2v-plus
    DASHSCOPE_WAN_SIZE       default 832*480
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

DEFAULT_MODEL = "wan2.2-t2v-plus"
# Official 480P 16:9 for wan2.2-t2v-plus (width*height with asterisk)
DEFAULT_SIZE = "832*480"
ALLOWED_SIZES_480P = ("832*480", "480*832", "624*624")
CREATE_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
NOMINAL_DURATION_S = 5


def api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")
    return key


def workspace_id() -> str:
    wid = os.environ.get("DASHSCOPE_WORKSPACE_ID") or os.environ.get(
        "BAILIAN_WORKSPACE_ID"
    )
    if not wid:
        raise RuntimeError("DASHSCOPE_WORKSPACE_ID is not set")
    return wid.strip()


def base_url() -> str:
    return f"https://{workspace_id()}.cn-beijing.maas.aliyuncs.com"


def default_model() -> str:
    return os.environ.get("DASHSCOPE_WAN_MODEL", DEFAULT_MODEL)


def default_size() -> str:
    return os.environ.get("DASHSCOPE_WAN_SIZE", DEFAULT_SIZE)


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
        raise RuntimeError(f"Bailian Wan HTTP {e.code}: {err}") from e


def create_text_video(
    prompt: str,
    *,
    model: str | None = None,
    size: str | None = None,
    negative_prompt: str | None = None,
    prompt_extend: bool = True,
) -> str:
    size = size or default_size()
    payload: dict[str, Any] = {
        "model": model or default_model(),
        "input": {"prompt": prompt},
        "parameters": {
            "size": size,
            "prompt_extend": prompt_extend,
        },
    }
    if negative_prompt:
        payload["input"]["negative_prompt"] = negative_prompt
    url = f"{base_url()}{CREATE_PATH}"
    print(f"  POST {url}", flush=True)
    print(f"  model={payload['model']} size={size}", flush=True)
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
    return _request("GET", f"{base_url()}/api/v1/tasks/{task_id}")


def wait_task(
    task_id: str,
    *,
    poll_s: float = 10.0,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    t0 = time.time()
    while time.time() < deadline:
        last = get_task(task_id)
        status = (last.get("output") or {}).get("task_status", "UNKNOWN")
        print(f"  [{time.time() - t0:5.0f}s] task_status={status}", flush=True)
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
    size: str | None = None,
    negative_prompt: str | None = None,
    model: str | None = None,
    prompt_extend: bool = True,
) -> Path:
    task_id = create_text_video(
        prompt,
        model=model,
        size=size,
        negative_prompt=negative_prompt,
        prompt_extend=prompt_extend,
    )
    print(f"  task_id={task_id}", flush=True)
    result = wait_task(task_id)
    print("  download…", flush=True)
    return download_video(video_url_from_result(result), dest)


def generate_min_clip_and_frames(
    prompt: str,
    out_dir: str | Path,
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
    size: str | None = None,
    negative_prompt: str | None = None,
    model: str | None = None,
) -> tuple[Path, list[Path]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "wan.mp4"
    generate_and_download(
        prompt,
        mp4,
        size=size or default_size(),
        negative_prompt=negative_prompt,
        model=model,
    )
    frames = extract_frames(mp4, out_dir / "frames", count=frame_count)
    return mp4, frames
