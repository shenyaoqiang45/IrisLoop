"""Sequential frame sampling for writer clips (Kling / Wan / …)."""

from __future__ import annotations

from pathlib import Path


DEFAULT_FRAME_COUNT = 3


def even_indices(n: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if count == 1:
        return [n // 2]
    if count >= n:
        return list(range(n))
    return [int(round(i * (n - 1) / (count - 1))) for i in range(count)]


def extract_frames(
    video_path: str | Path,
    dest_dir: str | Path,
    *,
    count: int = DEFAULT_FRAME_COUNT,
    prefix: str = "frame",
) -> list[Path]:
    """Sample ``count`` frames evenly in playback order.

    Always reads sequentially. Seeking with CAP_PROP_POS_FRAMES is unreliable
    on many H.264 MP4s and can scramble order.
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
        frames_bgr = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_bgr.append(frame)
        n = len(frames_bgr)
        if n == 0:
            raise RuntimeError(f"no frames in {video_path}")
        idxs = even_indices(n, count)
        out: list[Path] = []
        for i, idx in enumerate(idxs):
            path = dest_dir / f"{prefix}_{i:02d}.jpg"
            cv2.imencode(".jpg", frames_bgr[idx])[1].tofile(str(path))
            out.append(path)
        return out
    finally:
        cap.release()
