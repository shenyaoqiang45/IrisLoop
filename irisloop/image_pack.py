"""640x480 1bpp 图片打包（复刻 im2Bytes.m，已用真机 BMP 验证）。

打包语义（对照 F:\\2026\\LE3AutoCam\\alignment_test_h.bmp 验证）:
    1. 二值化（阈值 127）
    2. fliplr 水平翻转
    3. 行主序展开比特（MATLAB reshape(pic',1,[]) 列主序 == numpy 行主序）
    4. 每 8 bit 打包 1 字节，MSB 在前
    5. 前附 62 字节 BMP 头

注意: 文档写的「124 字节头」有误，实为 62 字节
      (14 文件头 + 40 DIB + 8 双色调色板)，
      BMP 头内 file_size=38462、data_offset=62 可自校验。
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

# 来自 bmp_head.mat（已验证: 640x480, 1bpp, data_offset=62, size=38400）
BMP_HEAD = bytes.fromhex(
    "424D3E960000000000003E0000002800000080020000E00100000100010000"
    "00000000960000120B0000120B0000000000000000000000000000FFFFFF00"
)


def binarize(gray: np.ndarray, threshold: int = 127) -> np.ndarray:
    return (np.asarray(gray) > threshold).astype(np.uint8)


def pack(bw: np.ndarray) -> bytes:
    """二值图 -> 38400 字节设备载荷。"""
    if bw.shape != (HEIGHT, WIDTH):
        raise ValueError(f"期望 {HEIGHT}x{WIDTH}, 实际 {bw.shape}")
    a = np.fliplr(bw.astype(np.uint8))
    bits = a.reshape(-1).reshape(-1, 8)
    weights = (1 << np.arange(7, -1, -1)).astype(np.uint8)
    return (bits * weights).sum(axis=1).astype(np.uint8).tobytes()


def build_stream(bw: np.ndarray) -> bytes:
    """62 字节头 + 38400 字节数据 = 38462 字节。"""
    return BMP_HEAD + pack(bw)


def save_bmp(path: str, bw: np.ndarray) -> None:
    """存成可在 PC 上直接查看的 BMP（自底向上行序）。"""
    rows = np.frombuffer(pack(bw), dtype=np.uint8).reshape(HEIGHT, ROW_BYTES)
    body = np.flipud(rows).tobytes()  # 自顶向下 -> BMP 的自底向上
    with open(path, "wb") as f:
        f.write(BMP_HEAD)
        f.write(body)


# ---------------- 测试图生成 ----------------


def make_test_image(kind: str = "iris") -> np.ndarray:
    """生成 640x480 二值测试图。带方向标记，便于肉眼判断翻转/镜像。"""
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
    else:  # iris: 文字 + 边框 + 角标
        cv2.rectangle(img, (8, 8), (WIDTH - 9, HEIGHT - 9), 1, 3)
        cv2.putText(img, "IRIS", (150, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    3.2, 1, 8, cv2.LINE_AA)
        cv2.putText(img, "L O O P", (140, 300), cv2.FONT_HERSHEY_SIMPLEX,
                    1.8, 1, 5, cv2.LINE_AA)
        # 方向标记：左上角实心方块（唯一，可判别翻转/镜像）
        cv2.rectangle(img, (20, 20), (80, 80), 1, -1)
        # 右上角空心方块（判别水平镜像）
        cv2.rectangle(img, (WIDTH - 81, 20), (WIDTH - 21, 80), 1, 3)
        # 底部中心短线（判别垂直翻转）
        cv2.line(img, (WIDTH // 2 - 60, HEIGHT - 25),
                 (WIDTH // 2 + 60, HEIGHT - 25), 1, 4)
        # 刻度尺，判断缩放
        for x in range(40, WIDTH - 40, 10):
            h = 14 if (x // 10) % 5 == 0 else 7
            cv2.line(img, (x, HEIGHT - 60), (x, HEIGHT - 60 - h), 1, 2)

    return img


def describe(bw: np.ndarray) -> str:
    return (f"{bw.shape[1]}x{bw.shape[0]} 1bpp  "
            f"白像素={int(bw.sum())} ({bw.mean()*100:.1f}%)  "
            f"载荷={DATA_BYTES}B")
