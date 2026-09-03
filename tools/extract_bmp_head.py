"""Extract the 62-byte BMP header from bmp_head.mat and cross-check against a real BMP in the tree.

Also verify im2Bytes.m bit packing (fliplr + MSB-first).
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
    # BITMAPINFOHEADER: I i i H H I I i i I I  -> 40 bytes; split in two to avoid alignment issues
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
    """Replicate im2Bytes.m: fliplr -> flatten by row -> MSB-first bit pack.

    MATLAB: pic = fliplr(pic); bits = reshape(pic',1,[]); groups of 8 bits, MSB first.
    pic' is a transpose; unroll order equals column-major = scan by column.
    """
    bw = (gray > 127).astype(np.uint8)
    flipped = np.fliplr(bw)
    # Transpose then row-unroll == original image unrolled column-major
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
    print("\n=== parse ===")
    for k, v in info.items():
        print(f"  {k:<16}: {v}")

    try:
        with open(REF_BMP, "rb") as f:
            ref = f.read()
        print(f"\n=== reference {REF_BMP} ===")
        print(f"  file length: {len(ref)}")
        print(f"  first 62 bytes: {ref[:62].hex().upper()}")
        print(f"  matches head: {ref[:62] == head}")
        if len(ref) >= 62:
            payload = ref[62:]
            print(f"  data length: {len(payload)} (expected 38400)")
    except FileNotFoundError:
        print(f"\n[warn] reference BMP not found: {REF_BMP}")

    # Verify bit packing: re-pack the reference image; should match original payload
    try:
        import cv2

        img = cv2.imread(REF_BMP, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"\n=== bit-pack check (cv2 read {img.shape}) ===")
            packed = pack_like_im2bytes(img)
            print(f"  packed length: {len(packed)}")
            with open(REF_BMP, "rb") as f:
                ref = f.read()
            ref_payload = ref[62 : 62 + len(packed)]
            same = packed == ref_payload
            print(f"  matches reference BMP payload: {same}")
            if not same:
                n = min(len(packed), len(ref_payload))
                diff = sum(1 for i in range(n) if packed[i] != ref_payload[i])
                print(f"  differing bytes: {diff}/{n}")
    except ImportError:
        print("\n[warn] cv2 unavailable, skip pack check")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
