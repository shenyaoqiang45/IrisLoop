"""USB camera capture wrapper."""

from __future__ import annotations

import contextlib
import time
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

# On Windows, MSMF reports fps correctly; DSHOW reports 0, so try MSMF first
PREFERRED_BACKENDS = (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)
# External USB cams often use MJPG; built-in laptop cams are usually YUY2/NV12 — forcing MJPG can black-screen or fail to open
FOURCC_TRIES: Tuple[Optional[str], ...] = ("MJPG", None)


class UsbCamera:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        backend: Optional[int] = None,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.fourcc: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_size: Tuple[int, int] = (0, 0)
        self.actual_fps: float = 0.0

    def open(self) -> None:
        candidates = (self.backend,) if self.backend is not None else PREFERRED_BACKENDS
        errors: List[str] = []

        for backend in candidates:
            for fourcc in FOURCC_TRIES:
                cap = cv2.VideoCapture(self.index, backend)
                if not cap.isOpened():
                    cap.release()
                    errors.append(f"backend={backend} fourcc={fourcc or 'default'} open failed")
                    continue

                if fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
                cap.set(cv2.CAP_PROP_FPS, float(self.fps))

                ok, frame = cap.read()
                if not ok or frame is None or frame.size < 100:
                    cap.release()
                    errors.append(f"backend={backend} fourcc={fourcc or 'default'} first-frame read failed")
                    continue

                h, w = frame.shape[:2]
                self.cap = cap
                self.fourcc = fourcc
                self.actual_size = (w, h)
                # Backend-reported fps can be unreliable (DSHOW is always 0); calibrate later with measure_fps
                self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
                return

        raise RuntimeError(
            f"cannot open camera index={self.index} ({'; '.join(errors) or 'no usable backend'})"
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def measure_fps(self, seconds: float = 2.0) -> float:
        """Measure actual frame rate. Prefer this when the backend-reported value is untrustworthy."""
        if self.cap is None:
            return 0.0
        count = 0
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            ok, _ = self.read()
            count += int(ok)
        elapsed = time.perf_counter() - start
        measured = count / elapsed if elapsed > 0 else 0.0
        if measured > 1.0:
            self.actual_fps = measured
        return measured

    def backend_name(self) -> str:
        if self.cap is None:
            return "unknown"
        try:
            return self.cap.getBackendName()
        except Exception:
            return "unknown"

    def set_auto_exposure(self, enabled: bool) -> None:
        if self.cap is None:
            return
        # MSMF: 0.75 auto / 0.25 manual; ignore failures
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if enabled else 0.25)
        except Exception:
            pass

    def set_exposure(self, value: float) -> None:
        if self.cap is None:
            return
        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        except Exception:
            pass

    def info(self) -> str:
        w, h = self.actual_size
        fcc = self.fourcc or "default"
        return (
            f"cam#{self.index} {w}x{h} @{self.actual_fps:.1f}fps "
            f"backend={self.backend_name()} fourcc={fcc}"
        )

    def __enter__(self) -> "UsbCamera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


@contextlib.contextmanager
def quiet_opencv() -> Iterator[None]:
    """Mute OpenCV warnings that flood when probing missing device indices."""
    try:
        previous = cv2.setLogLevel(0)  # LOG_LEVEL_SILENT
    except Exception:
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            cv2.setLogLevel(previous)


def probe_cameras(max_index: int = 8) -> List[dict]:
    """Probe available cameras. Returned sizes are each camera's default resolution."""
    results: List[dict] = []
    with quiet_opencv():
        for index in range(max_index):
            for backend in PREFERRED_BACKENDS:
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    break
                h, w = frame.shape[:2]
                results.append(
                    {
                        "index": index,
                        "width": w,
                        "height": h,
                        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                        "backend": _backend_name(cap, backend),
                    }
                )
                cap.release()
                break
    return results


def _backend_name(cap: cv2.VideoCapture, backend: int) -> str:
    try:
        return cap.getBackendName()
    except Exception:
        return str(backend)
