"""IrisLoop 视觉分析器 —— 比对目标图与摄像头实拍帧，输出结构化评估。

支持两种模式：
  - rule: 纯 OpenCV 规则评估（无需 API key，离线）
  - ai:   多模态大模型评估（需 MOONSHOT_API_KEY）

用法:
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
    """一次投影效果评估。"""

    ok: bool = False                    # 整体是否达标
    brightness_mean: float = 0.0        # 实拍平均亮度 0-255
    bright_ratio: float = 0.0           # 亮区占比 0-1
    sharpness: float = 0.0              # Laplacian 方差（清晰度）
    orientation: str = "unknown"        # normal / flip_v / flip_h / rot180 / unknown
    similarity: float = 0.0             # 与目标图的结构相似度 0-1
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


# ---------------- 图像加载 ----------------


def load_gray(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取 {path}")
    return img


# ---------------- 规则评估 ----------------


def assess_rule(target: np.ndarray, capture: np.ndarray) -> Assessment:
    """纯 OpenCV 规则评估。"""
    a = Assessment()

    # 统一尺寸
    t = cv2.resize(target, (640, 480)) if target.shape != (480, 640) else target
    c = capture
    if c.shape != (480, 640):
        c = cv2.resize(c, (640, 480))

    # 基础指标
    a.brightness_mean = float(c.mean())
    a.bright_ratio = float((c > 128).mean())
    a.sharpness = float(cv2.Laplacian(c, cv2.CV_64F).var())

    # 方向检测：对实拍图做 4 种变换，与目标图比对相关
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
        # 归一化相关
        score = float(np.corrcoef(t_bin.flatten(), v_bin.flatten())[0, 1])
        if score > best_score:
            best_score, best_ori = score, name
    a.orientation = best_ori
    a.similarity = max(0.0, best_score)

    # 判定
    if a.brightness_mean < 10:
        a.issues.append("画面几乎全黑")
        a.suggestions.append("检查是否投出 / 提高亮度 / 降低环境光")
    if a.bright_ratio < 0.01:
        a.issues.append("亮区占比过低")
    if a.sharpness < 50:
        a.issues.append("画面模糊")
        a.suggestions.append("调整焦距 / 缩短投射距离")
    if best_ori == "flip_v":
        a.issues.append("图像上下翻转")
        a.suggestions.append("打包前加 flipud 或下发翻转命令 0x12")
    elif best_ori == "flip_h":
        a.issues.append("图像左右镜像")
        a.suggestions.append("打包前加 fliplr 或下发镜像命令 0x1E")
    elif best_ori == "rot180":
        a.issues.append("图像旋转 180°")
        a.suggestions.append("打包前 rot180 或组合翻转命令")
    if a.similarity < 0.3 and a.brightness_mean >= 10:
        a.issues.append("与目标图差异大")

    a.ok = (
        a.brightness_mean >= 10
        and a.bright_ratio >= 0.01
        and a.sharpness >= 50
        and best_ori == "normal"
        and a.similarity >= 0.4
    )
    return a


# ---------------- AI 评估 ----------------


def _b64_image(path: str) -> str:
    """读图并压缩成 JPEG base64（PNG 原图易超 413 限制）。"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法读取 {path}")
    # 缩到最长边 800 并 JPEG 压缩
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode()


def assess_ai(target_path: str, capture_path: str) -> Assessment:
    """调用多模态大模型评估（Moonshot Kimi）。"""
    key = os.environ.get("MOONSHOT_API_KEY")
    if not key:
        raise RuntimeError("未设置 MOONSHOT_API_KEY")

    import urllib.request

    prompt = """你是 MEMS 激光投影质检员。第一张是目标图(应投出的内容)，第二张是摄像头实拍。
评估实拍效果，只回 JSON：
{
  "ok": bool,             // 整体是否可接受
  "orientation": "normal|flip_v|flip_h|rot180|unknown",
  "brightness": "too_dark|ok|too_bright",
  "sharpness": "blurry|ok",
  "distortion": "none|keystone|other",
  "issues": ["..."],      // 主要问题
  "suggestions": ["..."]  // 修正建议
}"""

    payload = {
        "model": "kimi-k2-0905-preview" if key.startswith("sk-kimi-")
        else "moonshot-v1-8k-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{_b64_image(target_path)}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{_b64_image(capture_path)}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.1,
    }

    req = urllib.request.Request(
        # Kimi 开放平台 key (sk-kimi-) 走 api.kimi.com，Moonshot key 走 api.moonshot.cn
        "https://api.kimi.ai/v1/chat/completions"
        if key.startswith("sk-kimi-")
        else "https://api.moonshot.cn/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())

    text = body["choices"][0]["message"]["content"]
    # 提取 JSON
    start = text.find("{")
    end = text.rfind("}") + 1
    data = json.loads(text[start:end])

    a = Assessment()
    a.ok = bool(data.get("ok", False))
    a.orientation = data.get("orientation", "unknown")
    a.issues = data.get("issues", [])
    a.suggestions = data.get("suggestions", [])
    # AI 模式下部分数值指标留 0，以定性判断为主
    return a


# ---------------- CLI ----------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="目标图")
    ap.add_argument("capture", help="实拍图")
    ap.add_argument("--mode", choices=["rule", "ai"], default="rule")
    args = ap.parse_args(argv)

    if args.mode == "ai":
        a = assess_ai(args.target, args.capture)
    else:
        a = assess_rule(load_gray(args.target), load_gray(args.capture))

    print("=== 评估结果 ===")
    print(a.summary())
    print(json.dumps(a.to_dict(), ensure_ascii=False))
    return 0 if a.ok else 1


if __name__ == "__main__":
    sys.exit(main())
