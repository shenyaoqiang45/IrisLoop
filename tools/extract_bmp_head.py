"""从 bmp_head.mat 提取 62 字节 BMP 头，并与仓库里的真 BMP 交叉验证。

同时校验 im2Bytes.m 的位打包规则（fliplr + MSB-first）。
"""

from __future__ import annotations

import sys

import numpy as np
import scipy.io as sio

MAT = r"F:\2026\LE3AutoCam\bmp_head.mat"
REF_BMP = r"F:\2026\LE3AutoCam\alignment_test_h.bmp"


def load_head(path: str) -> bytes:
    d = sio.loadmat(path)
    head = d["head"]
    arr = np.asarray(head).ravel()
    return bytes(int(str(x), 16) for x in arr)


def parse_bmp_header(h: bytes) -> dict:
    if h[:2] != b"BM":
        return {"error": "not BMP"}
    import struct

    (size, _r1, _r2, offset) = struct.unpack("<IHHI", h[2:14])
    # BITMAPINFOHEADER: I i i H H I I i i I I  -> 40 字节，显式拆两段避开对齐
    (dib, w, hh, planes, bpp) = struct.unpack("<IiiHH", h[14:30])   # 16B
    (comp, img_size, xpels, ypels, clrused, clrimp) = struct.unpack("<IIiiII", h[30:54])  # 24B
    return {
        "file_size": size,
        "data_offset": offset,
        "dib_size": dib,
        "width": w,
        "height": hh,
        "planes": planes,
        "bpp": bpp,
        "compression": comp,
        "image_size": img_size,
        "total_expected": offset + img_size,
    }


def pack_like_im2bytes(gray: np.ndarray) -> bytes:
    """复刻 im2Bytes.m: fliplr -> 按行展平 -> MSB-first 位打包。

    MATLAB: pic = fliplr(pic); bits = reshape(pic',1,[]); 按 8 位一组 MSB 在前。
    pic' 是转置，展开顺序等价于「按列优先」= 逐列扫描。
    """
    bw = (gray > 127).astype(np.uint8)
    flipped = np.fliplr(bw)
    # 转置后按行展开 == 原图按列优先展开
    bits = flipped.T.reshape(-1)
    pad = (-bits.size) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    bits = bits.reshape(-1, 8)
    weights = (1 << np.arange(7, -1, -1)).astype(np.uint8)
    return (bits * weights).sum(axis=1).astype(np.uint8).tobytes()


def main() -> int:
    head = load_head(MAT)
    print(f"=== bmp_head.mat -> {len(head)} bytes ===")
    print("  " + head.hex().upper())
    info = parse_bmp_header(head)
    print("\n=== 解析 ===")
    for k, v in info.items():
        print(f"  {k:<16}: {v}")

    try:
        with open(REF_BMP, "rb") as f:
            ref = f.read()
        print(f"\n=== 参照 {REF_BMP} ===")
        print(f"  文件长度: {len(ref)}")
        print(f"  前62字节: {ref[:62].hex().upper()}")
        print(f"  与 head 一致: {ref[:62] == head}")
        if len(ref) >= 62:
            payload = ref[62:]
            print(f"  数据长度: {len(payload)} (期望 38400)")
    except FileNotFoundError:
        print(f"\n[warn] 参照 BMP 不存在: {REF_BMP}")

    # 校验位打包：拿参照图重打包，应与原数据一致
    try:
        import cv2

        img = cv2.imread(REF_BMP, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"\n=== 位打包校验 (cv2 读图 {img.shape}) ===")
            packed = pack_like_im2bytes(img)
            print(f"  打包长度: {len(packed)}")
            with open(REF_BMP, "rb") as f:
                ref = f.read()
            ref_payload = ref[62 : 62 + len(packed)]
            same = packed == ref_payload
            print(f"  与参照BMP数据一致: {same}")
            if not same:
                n = min(len(packed), len(ref_payload))
                diff = sum(1 for i in range(n) if packed[i] != ref_payload[i])
                print(f"  不同字节数: {diff}/{n}")
    except ImportError:
        print("\n[warn] cv2 不可用，跳过打包校验")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
