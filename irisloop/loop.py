"""IrisLoop minimal closed loop — iterative self-calibration on a single image.

Per-iteration flow:
    1. Push the current content to the device as 1_1.bmp and keep it on
    2. Capture one camera frame
    3. analyzer assessment (rule or AI)
    4. If it fails and a correction is actionable, apply it and continue
    5. Stop on convergence or max iterations

Usage:
    # rule mode (offline): push a marker image, up to 3 iterations
    python -m irisloop.loop data/01a_upload_alignment_h.jpg

    # AI mode (requires MOONSHOT_API_KEY)
    python -m irisloop.loop data/01a_upload_alignment_h.jpg --mode ai

    # capture only (device already showing the target content)
    python -m irisloop.loop data/01a_upload_alignment_h.jpg --no-push
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "..")

import cv2
import numpy as np

from irisloop import protocol as P
from irisloop.analyzer import Assessment, assess_ai, assess_rule, load_gray
from irisloop.ble_client import IrisBleClient
from irisloop.camera import UsbCamera
from irisloop.image_pack import binarize, build_stream
from irisloop.projector import CHAR_FILE_DATA, CHAR_FILE_END, CHAR_FILE_START

DEFAULT_ADDR = "F4:12:FA:B6:B7:CA"
CAPTURE_DIR = "captures/loop"

ATT_HEADER_BYTES = 3
DEFAULT_ATT_MTU = 23
FILE_NAME_FIELD_MIN_BYTES = 16
START_DELAY_S = 0.1
PACKET_DELAY_S = 0.03


def build_name_packet(file_size: int, file_name: str) -> bytes:
    nb = file_name.encode("utf-8")
    field = max(FILE_NAME_FIELD_MIN_BYTES, len(nb))
    return file_size.to_bytes(4, "big") + nb.ljust(field, b"\x00")


async def push_image(cli: IrisBleClient, stream: bytes, name: str) -> None:
    client = cli.client
    assert client is not None
    packet = build_name_packet(len(stream), name)
    await client.write_gatt_char(CHAR_FILE_START, packet, response=True)
    await asyncio.sleep(START_DELAY_S)
    mtu = getattr(client, "mtu_size", DEFAULT_ATT_MTU) or DEFAULT_ATT_MTU
    chunk = max(1, mtu - ATT_HEADER_BYTES)
    n = (len(stream) + chunk - 1) // chunk
    for i in range(n):
        await client.write_gatt_char(
            CHAR_FILE_DATA, stream[i * chunk:(i + 1) * chunk], response=False
        )
        await asyncio.sleep(PACKET_DELAY_S)
    await client.write_gatt_char(CHAR_FILE_END, packet, response=True)


def apply_correction(bw: np.ndarray, a: Assessment) -> tuple[np.ndarray, str]:
    """Apply a source-image correction from the assessment. Returns (image, note)."""
    if a.orientation == "flip_v":
        return np.flipud(bw), "flipud(vertical flip)"
    if a.orientation == "flip_h":
        return np.fliplr(bw), "fliplr(horizontal mirror)"
    if a.orientation == "rot180":
        return np.flipud(np.fliplr(bw)), "rot180(180° rotation)"
    return bw, "no correction"


async def run_loop(
    target_path: str,
    addr: str,
    mode: str,
    max_iters: int,
    push: bool,
    cam_index: int,
) -> int:
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    target_gray = load_gray(target_path)
    if target_gray.shape != (480, 640):
        target_gray = cv2.resize(target_gray, (640, 480))
    bw = binarize(target_gray)

    cli = IrisBleClient(addr) if push else None
    if cli is not None:
        print(f"=== CONNECT {addr} ===")
        await cli.connect()

    try:
        for it in range(1, max_iters + 1):
            print(f"\n========== iteration {it}/{max_iters} ==========")

            # 1. push image
            if cli is not None:
                stream = build_stream(bw)
                print(f"  pushing {len(stream)}B -> 1_1.bmp ...")
                await push_image(cli, stream, "1_1.bmp")
                # keep group 1 on continuously
                await cli.stop()
                await asyncio.sleep(0.3)
                r = await cli.play(group_id=1, loop=True,
                                   total_100ms=36000, interval_100ms=50)
                print(f"  play: {'ok' if r.ok else r.error}")
                await asyncio.sleep(1.0)  # wait for the projection to settle

            # 2. capture
            shot = os.path.join(CAPTURE_DIR, f"iter{it}_capture.png")
            print("  capturing ...")
            cam = UsbCamera(cam_index, 1280, 720, 30)
            cam.open()
            try:
                # manual exposure: EV=-7 balances stripe suppression vs brightness (best in tests)
                try:
                    cam.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    cam.cap.set(cv2.CAP_PROP_EXPOSURE, -7)
                except Exception:
                    pass
                await asyncio.sleep(1.0)  # let exposure settle
                frame = None
                for _ in range(10):
                    ok, f = cam.read()
                    if ok and f is not None:
                        frame = f
                if frame is None:
                    print("  [error] capture failed")
                    return 2
                cv2.imwrite(shot, frame)
            finally:
                cam.release()

            # 3. assess
            if mode == "ai":
                target_png = os.path.join(CAPTURE_DIR, "target.png")
                cv2.imwrite(target_png, (bw * 255).astype(np.uint8))
                a = assess_ai(target_png, shot)
            else:
                a = assess_rule(bw * 255, load_gray(shot))
            print("  assessment:")
            print(a.summary())

            # 4. converge or correct
            if a.ok:
                print(f"\n✅ iteration {it} converged; projection looks good")
                return 0
            if it < max_iters:
                bw2, note = apply_correction(bw, a)
                if note != "no correction":
                    print(f"  applied correction: {note}")
                    bw = bw2
                else:
                    print("  no auto-correctable issue; observing again")

        print(f"\n⚠️ reached max iterations ({max_iters}); did not fully converge")
        return 1
    finally:
        if cli is not None:
            await cli.disconnect()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="target image (e.g. data/01a_upload_alignment_h.jpg)")
    ap.add_argument("--address", default=DEFAULT_ADDR)
    ap.add_argument("--mode", choices=["rule", "ai"], default="rule")
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--no-push", action="store_true",
                    help="do not push an image; capture and assess only (device must already show the target)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args(argv)

    return asyncio.run(
        run_loop(
            args.target,
            args.address,
            args.mode,
            args.max_iters,
            push=not args.no_push,
            cam_index=args.camera,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
