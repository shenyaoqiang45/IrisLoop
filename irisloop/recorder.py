"""视频录制：带编码回退的 VideoWriter 封装。

本机实测：mp4v / MJPG / XVID 可用，avc1(H.264) 因缺 OpenH264 不可用，
因此必须按扩展名逐个尝试，不能写死单一编码。
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

CODECS_BY_EXT = {
    ".mp4": ["mp4v", "avc1"],
    ".avi": ["MJPG", "XVID", "mp4v"],
    ".mkv": ["mp4v", "XVID"],
}


class VideoRecorder:
    def __init__(self, path: str, fps: float, size: Tuple[int, int], codec: Optional[str] = None):
        self.path = path
        self.fps = fps
        self.size = size  # (width, height)
        self.codec = codec
        self.writer: Optional[cv2.VideoWriter] = None
        self.used_codec: Optional[str] = None
        self.frames = 0
        self.dropped = 0

    def open(self) -> str:
        ext = os.path.splitext(self.path)[1].lower()
        candidates = [self.codec] if self.codec else CODECS_BY_EXT.get(ext, ["mp4v"])
        if not self.codec:
            candidates += [c for c in CODECS_BY_EXT.get(ext, []) if c not in candidates]

        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)

        errors: List[str] = []
        for codec in candidates:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
            if writer.isOpened():
                self.writer = writer
                self.used_codec = codec
                return codec
            writer.release()
            errors.append(f"{codec} 不可用")

        raise RuntimeError(f"无法创建视频文件 {self.path}（{'；'.join(errors)}）")

    def write(self, frame: np.ndarray) -> bool:
        if self.writer is None:
            return False
        h, w = frame.shape[:2]
        # VideoWriter 不会自动缩放，尺寸不符会静默丢帧
        if (w, h) != self.size:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)
        self.frames += 1
        return True

    def release(self) -> dict:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        size_mb = os.path.getsize(self.path) / (1024 * 1024) if os.path.exists(self.path) else 0.0
        return {
            "path": self.path,
            "codec": self.used_codec,
            "frames": self.frames,
            "size_mb": round(size_mb, 2),
            "resolution": f"{self.size[0]}x{self.size[1]}",
            "fps": round(self.fps, 2),
        }

    def __enter__(self) -> "VideoRecorder":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def default_output_path(output_dir: str, ext: str = ".mp4") -> str:
    os.makedirs(output_dir, exist_ok=True)
    name = time.strftime("capture_%Y%m%d_%H%M%S") + ext
    return os.path.join(output_dir, name)
