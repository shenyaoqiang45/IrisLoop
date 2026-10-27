"""Experimental, camera-verified five-circle projection. See docs/five-circle-projection-pipeline.md."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from irisloop.calibrate import make_dot_image
from irisloop.hik_camera import HikCamera
from irisloop.image_pack import build_stream
from irisloop.procalib import BoardModel, detect_board_dots, detect_grid_dots, match_lattice

DEFAULT_BRIDGE = ROOT.parent / "rea-IrisGreen" / "agent" / "service.py"
DEFAULT_SERIAL = "K65564865"


class Bridge:
    def __init__(self, process: asyncio.subprocess.Process):
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Bridge requires stdin/stdout pipes")
        self.process = process
        self.stdin = process.stdin
        self.stdout = process.stdout
        self.counter = 0
        self.connected = False
        self.slot_prepared = False

    async def rpc(self, method: str, params: dict | None = None):
        self.counter += 1
        request = dict(jsonrpc="2.0", id=self.counter, method=method, params=params or {})
        self.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
        await self.stdin.drain()
        while True:
            line = await self.stdout.readline()
            if not line:
                raise RuntimeError(f"Bridge exited during {method}")
            response = json.loads(line)
            if "id" not in response:
                if response.get("method") == "event.disconnected":
                    self.connected = False
                    print("[warn] Bridge disconnected", flush=True)
                continue
            if response["id"] != self.counter:
                raise RuntimeError(f"Unexpected RPC response id: {response['id']}")
            if "error" in response:
                raise RuntimeError(f"{method}: {response['error']}")
            return response["result"]

    async def push_current(self, mask: np.ndarray):
        payload = base64.b64encode(build_stream(mask)).decode("ascii")
        await self.rpc("display.show_frame", dict(
            data_base64=payload, clear_slot=not self.slot_prepared, seconds=3600))
        self.slot_prepared = True

    async def close(self):
        try:
            await self.rpc("service.shutdown")
        finally:
            self.stdin.close()
            await self.process.wait()


async def open_bridge(service: Path, address: str | None) -> tuple[Bridge, dict]:
    if not service.is_file():
        raise FileNotFoundError(f"Bridge service not found: {service}; pass --bridge")
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-X", "utf8", "-u", str(service),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
    bridge = Bridge(process)
    try:
        result = await bridge.rpc("device.connect", {"address": address} if address else {})
        if result.get("connected") is not True:
            raise RuntimeError(f"Device did not connect: {result}")
        bridge.connected = True
        print("CONNECTED", result, flush=True)
        return bridge, result["state"]
    except Exception:
        await bridge.close()
        raise


def save_image(out: Path, name: str, frame: np.ndarray):
    if not cv2.imwrite(str(out / name), frame):
        raise RuntimeError(f"Cannot save {out / name}")


def grab(cam: HikCamera, serial: str, exposure: float = 40000,
         gain: float = 12, frames: int = 8) -> np.ndarray:
    cam.open(serial=serial)
    try:
        cam.set_gain(gain)
        cam.set_exposure_us(exposure)
        cam.flush(4)
        captured = []
        for _ in range(frames):
            ok, frame = cam.read(2000)
            if not ok or frame is None:
                raise RuntimeError("Camera capture failed")
            captured.append(frame)
        return np.mean(captured, axis=0).astype(np.uint8)
    finally:
        cam.close()


def board_points(frame, board):
    for block in (151, 101, 71, 51):
        dots = detect_board_dots(frame, block=block, cval=5)
        matched = match_lattice(dots, board)
        if matched and len(matched) >= 90 and board.fiducials <= matched.keys():
            return matched, dots
    raise RuntimeError("Cannot identify full board and all five large circles")


def transform(H, points):
    return cv2.perspectiveTransform(np.array(points, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def spots(frame, bg, hull):
    candidates = detect_grid_dots(frame, bg, thresh=8, min_area=6, max_area=2000)
    return [p for p in candidates if cv2.pointPolygonTest(hull, p, False) >= 0]


def large_circles(dots, hull):
    inside = [d for d in dots if cv2.pointPolygonTest(hull, (d["cx"], d["cy"]), False) >= 0]
    if not inside:
        raise RuntimeError("No circular dots inside board hull")
    median = np.median([d["area"] for d in inside])
    large = [d for d in inside if d["area"] > 2.5 * median]
    if len(large) != 5:
        raise RuntimeError(f"Expected exactly five measured large circles, found {len(large)}")
    large.sort(key=lambda d: (d["cy"], d["cx"]))
    return np.array([[d["cx"], d["cy"]] for d in large]), [
        float(np.sqrt(d["area"] / np.pi)) for d in large]


def evaluate(frame, bg, hull, targets, radii):
    found = np.array(spots(frame, bg, hull))
    if len(found) != 5:
        raise RuntimeError(f"Expected five actual spots, found {len(found)}")
    distance = np.linalg.norm(targets[:, None] - found[None, :], axis=2)
    indices = distance.argmin(axis=1)
    if len(set(indices)) != 5:
        raise RuntimeError("Spots do not map one-to-one to targets")
    gain = frame.astype(float).mean(axis=2) - bg.astype(float).mean(axis=2)
    yy, xx = np.indices(gain.shape)
    measurements = []
    for i, (target, radius) in enumerate(zip(targets, radii)):
        observed = found[indices[i]]
        error = float(np.linalg.norm(observed - target))
        rr = np.hypot(xx - target[0], yy - target[1])
        lit = (gain > 8) & (rr < radius * 1.8)
        total = int(lit.sum())
        inside = int((lit & (rr < radius)).sum())
        containment = inside / total if total else 0
        if error > min(5, radius * 0.25) or containment < 0.9 or inside < 6:
            raise RuntimeError(f"Target {i+1} failed: error={error:.2f}, containment={containment:.3f}")
        measurements.append(dict(target=i+1, observed=observed.tolist(), error_px=error,
                                 light_inside_circle_fraction=containment, lit_pixels_inside=inside))
    return measurements


async def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "address": args.address, "samples": []}
    black = np.zeros((480, 640), np.uint8)
    cam = HikCamera()
    bridge = None
    passed = False
    try:
        bridge, state = await open_bridge(args.bridge, args.address)
        report["address"] = state["address"]

        async def project(mask):
            await bridge.push_current(mask)
            await asyncio.sleep(2.5)

        board = BoardModel.load(str(ROOT / "cbinfo" / "CB-0300-W-001.layout.json"))
        for attempt in range(4):
            await project(black)
            bg = grab(cam, args.serial)
            save_image(out, f"black_{attempt}.png", bg)
            try:
                matched, dots = board_points(bg, board)
            except RuntimeError:
                print(f"[warn] Black-frame board detection failed, attempt {attempt + 1}", flush=True)
                if attempt == 3:
                    raise
                await asyncio.sleep(3)
            else:
                break
        hull = cv2.convexHull(np.array(list(matched.values()), np.float32))
        targets, radii = large_circles(dots, hull)
        report.update(targets=targets.tolist(), target_order="camera top-to-bottom", radii_px=radii)
        save_image(out, "background.png", bg)

        anchors = np.array([(320, 240), (230, 150), (410, 150), (410, 330), (230, 330)], float)
        observations = []
        if args.resume is None:
            for i, pt in enumerate(anchors):
                await project(make_dot_image(*pt, radius=7))
                frame = grab(cam, args.serial)
                save_image(out, f"anchor_{i}.png", frame)
                found = spots(frame, bg, hull)
                if len(found) != 1:
                    raise RuntimeError(f"Expected one anchor, observed {found}")
                if cv2.pointPolygonTest(hull, found[0], True) < 25:
                    raise RuntimeError("Anchor is not safely within the board")
                observations.append(found[0])
            H, _ = cv2.findHomography(anchors, np.array(observations), 0)
        else:
            with np.load(args.resume / "live_homography.npz") as seed:
                H = seed["H"]
                anchors, observations = seed["proj"], seed["cam"].tolist()
            previous = json.loads((args.resume / "report.json").read_text(encoding="utf-8"))
            motion = float(np.linalg.norm(targets - previous["targets"], axis=1).max())
            if motion > 1.5:
                raise RuntimeError(f"Saved target geometry changed by {motion:.2f}px")
            report.update(seed_from=str(args.resume), seed_target_movement_px=motion)
        if H is None or not np.isfinite(H).all():
            raise RuntimeError("Invalid live homography")
        np.savez(out / "live_homography.npz", H=H, proj=anchors, cam=observations)
        proj = transform(np.linalg.inv(H), targets)
        if args.resume is not None:
            if "projector_points" in previous:
                proj = np.asarray(previous["projector_points"], dtype=float)
            for sample in previous["samples"]:
                proj[sample["target"] - 1] = sample["projector"]
        else:
            report["anchor_residual_px"] = np.linalg.norm(transform(H, anchors) - observations, axis=1).tolist()
            for i, target in enumerate(targets):
                for iteration in range(8):
                    pt = np.rint(proj[i])
                    if not (8 <= pt[0] < 632 and 8 <= pt[1] < 472):
                        raise RuntimeError(f"Target {i+1} outside projector bounds: {pt}")
                    await project(make_dot_image(*pt, radius=6))
                    frame = grab(cam, args.serial)
                    save_image(out, f"target_{i+1}_iteration_{iteration}.png", frame)
                    found = spots(frame, bg, hull)
                    if len(found) != 1:
                        raise RuntimeError(f"Target {i+1}: expected one spot, observed {found}")
                    observed = np.array(found[0])
                    error = float(np.linalg.norm(observed - target))
                    sample = dict(target=i+1, iteration=iteration, projector=pt.tolist(),
                                  observed=observed.tolist(), error_px=error)
                    report["samples"].append(sample)
                    print("TARGET", sample, flush=True)
                    if error <= 3:
                        proj[i] = pt
                        break
                    base = transform(H, [pt])[0]
                    J = (transform(H, [pt + (1, 0), pt + (0, 1)]) - base).T
                    delta = np.linalg.solve(J, target - observed)
                    if np.linalg.norm(delta) > 40:
                        delta *= 40 / np.linalg.norm(delta)
                    proj[i] = pt + delta
                else:
                    raise RuntimeError(f"Target {i+1} failed to converge")

        if args.resume is None:
            await project(black)
        final_bg = grab(cam, args.serial)
        _, fresh_dots = board_points(final_bg, board)
        fresh_targets, _ = large_circles(fresh_dots, hull)
        movement = float(np.linalg.norm(fresh_targets - targets, axis=1).max())
        if movement > 1.5:
            raise RuntimeError(f"Board moved by {movement:.2f}px")
        save_image(out, "final_background.png", final_bg)
        if proj.shape != (5, 2) or not np.isfinite(proj).all() or (
                (proj < (8, 8)) | (proj >= (632, 472))).any():
            raise RuntimeError("Invalid or out-of-bounds final projector coordinates")
        mask = black.copy()
        for pt in proj:
            mask |= make_dot_image(*pt, radius=6)
        save_image(out, "projector_mask.png", mask * 255)
        await project(mask)
        report["verification"] = []
        for repetition in range(3):
            await asyncio.sleep(1)
            frame = grab(cam, args.serial)
            save_image(out, f"verification_{repetition+1}_raw.png", frame)
            measured = evaluate(frame, final_bg, hull, targets, radii)
            report["verification"].append(measured)
            print("VERIFY", repetition+1, measured, flush=True)
        await bridge.close()
        bridge = None
        frame = grab(cam, args.serial)
        save_image(out, "after_disconnect_raw.png", frame)
        report["after_disconnect_verification"] = evaluate(frame, final_bg, hull, targets, radii)
        report.update(status="passed", projector_points=proj.tolist(), board_movement_px=movement,
                      final_state="five spots playing", play_seconds=3600)
        passed = True
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        if bridge is None:
            print("[warn] No active BLE session; projection state cannot be changed.", flush=True)
        raise
    finally:
        try:
            if bridge is not None:
                try:
                    if not passed and bridge.connected:
                        await bridge.push_current(black)
                        report["final_state"] = "black requested"
                finally:
                    await bridge.close()
        finally:
            cam.close()
            (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("REPORT", out / "report.json", report["status"], flush=True)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--address", help="omit only when exactly one projector advertises")
    ap.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    ap.add_argument("--serial", default=DEFAULT_SERIAL)
    ap.add_argument("--resume", type=Path, help="reuse a measured run; still verifies actual light")
    ap.add_argument("--out", type=Path, default=ROOT / "captures" / "procalib" /
                    datetime.now().strftime("five_circles_%Y%m%d_%H%M%S"))
    return ap


if __name__ == "__main__":
    asyncio.run(run(parser().parse_args()))
