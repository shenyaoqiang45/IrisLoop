"""Pin down the exact bit-packing semantics of im2Bytes.m (reverse from a reference BMP).

MATLAB: pic=fliplr(pic); bits=reshape(pic',1,[]);
Identity: column-major unroll of A'  ==  row-major unroll of A
So numpy should use a.reshape(-1) (C order), not a.T.reshape(-1).
"""

from __future__ import annotations

import cv2
import numpy as np

REF = r"F:\2026\LE3AutoCam\alignment_test_h.bmp"
ROW_BYTES = 80  # 640 / 8
NROWS = 480

with open(REF, "rb") as f:
    raw = f.read()
payload = raw[62 : 62 + 38400]

img = cv2.imread(REF, cv2.IMREAD_GRAYSCALE)
bw = (img > 127).astype(np.uint8)
print(f"cv2 read: {img.shape}  white ratio {bw.mean():.3f}")


def pack(a: np.ndarray) -> bytes:
    bits = a.reshape(-1).astype(np.uint8)
    bits = bits.reshape(-1, 8)
    w = (1 << np.arange(7, -1, -1)).astype(np.uint8)
    return (bits * w).sum(axis=1).astype(np.uint8).tobytes()


def as_rows(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint8).reshape(NROWS, ROW_BYTES)


# im2Bytes semantics: horizontal flip + row-major MSB pack
cand = pack(np.fliplr(bw))
cand_rows = as_rows(cand)
pay_rows = as_rows(payload)

print("\n=== per-row diff (candidate vs original payload) ===")
diff_rows = [r for r in range(NROWS) if not np.array_equal(cand_rows[r], pay_rows[r])]
print(f"  direct compare: {len(diff_rows)} rows differ -> {diff_rows[:12]}")

flipped = np.flipud(cand_rows)
d2 = [r for r in range(NROWS) if not np.array_equal(flipped[r], pay_rows[r])]
print(f"  after vertical flip: {len(d2)} rows differ -> {d2[:12]}")

print("\n=== try row offset (-3..+3) ===")
best = None
for shift in range(-3, 4):
    for base, bname in ((cand_rows, "direct"), (flipped, "vflip")):
        rolled = np.roll(base, shift, axis=0)
        n = sum(1 for r in range(NROWS) if not np.array_equal(rolled[r], pay_rows[r]))
        mark = "  <== MATCH" if n == 0 else ""
        print(f"  shift={shift:+d} {bname:<7}: {n:>3} rows differ{mark}")
        if n == 0 and best is None:
            best = (shift, bname)

if best:
    print(f"\nEXACT MATCH: shift={best[0]} base={best[1]}")
else:
    print("\nNo exact match; inspecting remaining differing rows")
    base = flipped if len(d2) < len(diff_rows) else cand_rows
    worst = [r for r in range(NROWS) if not np.array_equal(base[r], pay_rows[r])]
    print(f"  rows: {worst[:20]}")
    if worst:
        r = worst[0]
        print(f"  row {r} payload: {pay_rows[r][:24].hex().upper()}")
        print(f"  row {r} ours   : {base[r][:24].hex().upper()}")
        b = int(pay_rows[r].sum())
        o = int(base[r].sum())
        print(f"  row {r} popcount payload={b} ours={o}")
