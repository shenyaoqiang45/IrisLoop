"""640x480 1bpp image packing (reimplements im2Bytes.m, verified against a real-device BMP).

Packing semantics (verified against F:\\2026\\LE3AutoCam\\alignment_test_h.bmp):
    1. binarize (threshold 127)
    2. fliplr horizontal flip
    3. row-major bit flatten (MATLAB reshape(pic',1,[]) column-major == numpy row-major)
    4. pack 8 bits per byte, MSB first
    5. prepend a 62-byte BMP header

Note: the document's "124-byte header" is wrong; it is 62 bytes
      (14-byte file header + 40 DIB + 8 dual-color palette).
      The BMP header's file_size=38462 and data_offset=62 self-check.
"""

from __future__ import annotations

import struct

import cv2
import numpy as np

WIDTH = 640
HEIGHT = 480
ROW_BYTES = WIDTH // 8          # 80
DATA_BYTES = ROW_BYTES * HEIGHT # 38400
HEAD_BYTES = 62
STREAM_BYTES = HEAD_BYTES + DATA_BYTES  # 38462

# From bmp_head.mat (verified: 640x480, 1bpp, data_offset=62, size=38400)
BMP_HEAD = bytes.fromhex(
    "424D3E960000000000003E0000002800000080020000E00100000100010000"
    "00000000960000120B0000120B0000000000000000000000000000FFFFFF00"
)


def binarize(gray: np.ndarray, threshold: int = 127) -> np.ndarray:
    return (np.asarray(gray) > threshold).astype(np.uint8)


def pack(bw: np.ndarray) -> bytes:
    """Binary image -> 38400-byte device payload.

    The device reads BMP pixel data bottom-up (standard BMP storage), so pack()
    must flipud first so the first row the device reads is the last image row.
    Verified by reverse-engineering a real-device BMP (test-data/1_1.bmp):
    unpacking after flipud yields an upright image.
    """
    if bw.shape != (HEIGHT, WIDTH):
        raise ValueError(f"expected {HEIGHT}x{WIDTH}, got {bw.shape}")
    a = np.flipud(np.fliplr(bw.astype(np.uint8)))
    bits = a.reshape(-1).reshape(-1, 8)
    weights = (1 << np.arange(7, -1, -1)).astype(np.uint8)
    return (bits * weights).sum(axis=1).astype(np.uint8).tobytes()


def build_stream(bw: np.ndarray) -> bytes:
    """62-byte header + 38400-byte data = 38462 bytes."""
    return BMP_HEAD + pack(bw)


def save_bmp(path: str, bw: np.ndarray) -> None:
    """Write a BMP that can be viewed on a PC (bottom-up row order).

    pack() already includes flipud (device reads bottom-up); reshaping that
    payload and writing it out is a standard BMP.
    """
    rows = np.frombuffer(pack(bw), dtype=np.uint8).reshape(HEIGHT, ROW_BYTES)
    with open(path, "wb") as f:
        f.write(BMP_HEAD)
        f.write(rows.tobytes())


# ---------------- test-image generation ----------------


def make_test_image(kind: str = "iris") -> np.ndarray:
    """Build a 640x480 binary test image with orientation marks for spotting flip/mirror."""
    img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    if kind == "checker":
        sq = 40
        for r in range(HEIGHT // sq):
            for c in range(WIDTH // sq):
                if (r + c) % 2 == 0:
                    img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 1
    elif kind == "grid":
        for i in range(0, WIDTH, 40):
            img[:, i:i + 2] = 1
        for i in range(0, HEIGHT, 40):
            img[i:i + 2, :] = 1
    elif kind == "solid":
        img[:] = 1
    else:  # iris: text + border + corner marks
        cv2.rectangle(img, (8, 8), (WIDTH - 9, HEIGHT - 9), 1, 3)
        cv2.putText(img, "IRIS", (150, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    3.2, 1, 8, cv2.LINE_AA)
        cv2.putText(img, "L O O P", (140, 300), cv2.FONT_HERSHEY_SIMPLEX,
                    1.8, 1, 5, cv2.LINE_AA)
        # orientation mark: solid square top-left (unique, detects flip/mirror)
        cv2.rectangle(img, (20, 20), (80, 80), 1, -1)
        # hollow square top-right (detects horizontal mirror)
        cv2.rectangle(img, (WIDTH - 81, 20), (WIDTH - 21, 80), 1, 3)
        # short bar at bottom center (detects vertical flip)
        cv2.line(img, (WIDTH // 2 - 60, HEIGHT - 25),
                 (WIDTH // 2 + 60, HEIGHT - 25), 1, 4)
        # scale ticks, to judge scaling
        for x in range(40, WIDTH - 40, 10):
            h = 14 if (x // 10) % 5 == 0 else 7
            cv2.line(img, (x, HEIGHT - 60), (x, HEIGHT - 60 - h), 1, 2)

    return img


def describe(bw: np.ndarray) -> str:
    return (f"{bw.shape[1]}x{bw.shape[0]} 1bpp  "
            f"white_pixels={int(bw.sum())} ({bw.mean()*100:.1f}%)  "
            f"payload={DATA_BYTES}B")
