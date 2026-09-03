"""Kimi K3 API client (image/video understanding).

Read the API key from the environment only; never commit it:
    MOONSHOT_API_KEY / KIMI_API_KEY
Optional:
    MOONSHOT_BASE_URL  inferred from the key prefix by default
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "kimi-k3"


def api_key() -> str:
    key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
    if not key:
        raise RuntimeError("MOONSHOT_API_KEY or KIMI_API_KEY is not set")
    return key


def base_url() -> str:
    explicit = os.environ.get("MOONSHOT_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    key = api_key()
    if key.startswith("sk-kimi-"):
        return "https://api.kimi.ai/v1"
    # China endpoint is typically moonshot.cn; international docs use moonshot.ai
    return os.environ.get("MOONSHOT_API_HOST", "https://api.moonshot.cn").rstrip("/") + "/v1"


def _request(
    method: str,
    path: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 180.0,
    retries: int = 5,
) -> Any:
    import time

    url = f"{base_url()}{path}"
    hdrs = {
        "Authorization": f"Bearer {api_key()}",
        **(headers or {}),
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if not body:
                    return {}
                return json.loads(body)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"Kimi API HTTP {e.code}: {err}")
            if e.code == 429 and attempt + 1 < retries:
                # org concurrency limit is 1; wait for the previous request to finish
                time.sleep(2.0 + attempt * 2.0)
                continue
            raise last_err from e
    assert last_err is not None
    raise last_err


def chat(
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 1.0,
    max_completion_tokens: int = 4096,
    reasoning_effort: str = "low",
    response_format: dict | None = None,
    timeout: float = 300.0,
) -> dict:
    # kimi-k3 only accepts temperature=1
    if model.startswith("kimi-k3"):
        temperature = 1.0
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "reasoning_effort": reasoning_effort,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    return _request(
        "POST",
        "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )


def message_text(resp: dict) -> str:
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


def b64_data_url(path: str | Path, max_side: int = 1280, jpeg_quality: int = 80) -> str:
    """Load a local image, shrink if needed, and return a data URL (keeps request bodies small)."""
    import cv2
    import numpy as np

    path = Path(path)
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def upload_file(path: str | Path, purpose: str = "video") -> str:
    """multipart upload; returns a file id (for ms://)."""
    path = Path(path)
    boundary = "----IrisLoopBoundary7MA4YWxkTrZu0gW"
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    file_bytes = path.read_bytes()
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="purpose"\r\n\r\n{purpose}\r\n'
    ).encode()
    body += f"--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    body += file_bytes + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    data = _request(
        "POST",
        "/files",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout=300.0,
    )
    fid = data.get("id")
    if not fid:
        raise RuntimeError(f"upload did not return an id: {data}")
    return str(fid)


def delete_file(file_id: str) -> None:
    try:
        _request("DELETE", f"/files/{file_id}", timeout=60.0)
    except Exception:
        pass


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"no JSON in reply: {text[:400]}")
    return json.loads(text[start:end])
