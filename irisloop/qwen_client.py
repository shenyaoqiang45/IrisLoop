"""Qwen3-VL via SiliconFlow — vision grounding client ("say where, project there").

Auth: ``SILICONFLOW_API_KEY`` / ``SF_API_KEY`` (shared with the Wan writer).
Default model: ``Qwen/Qwen3-VL-32B-Instruct`` — grounding is perception work;
the Thinking variant only adds latency/cost here. Override via
``SILICONFLOW_VL_MODEL``.

Qwen3-VL grounding replies use 0-1000 normalized coordinates in a JSON array
like ``[{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]``.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"


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
    return os.environ.get("SILICONFLOW_VL_MODEL", DEFAULT_MODEL)


def _request(
    method: str,
    path: str,
    *,
    data: dict[str, Any] | None = None,
    timeout: float = 180.0,
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


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: float = 180.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model or default_model(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return _request("POST", "/chat/completions", data=payload, timeout=timeout)


def message_text(resp: dict[str, Any]) -> str:
    choice = resp["choices"][0]["message"]
    content = choice.get("content") or ""
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)


def b64_data_url_bgr(frame, max_side: int = 1280, jpeg_quality: int = 85) -> str:
    """Encode an in-memory BGR frame as a JPEG data URL."""
    import cv2

    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def vision_message(text: str, data_url: str) -> list[dict]:
    """One user message with an image followed by text (OpenAI-compatible)."""
    return [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": text},
        ],
    }]
