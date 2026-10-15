#!/usr/bin/env python
"""Run repeatable physical-reward A/B trials from one persisted target path."""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from irisloop.calibrate import (
    DEFAULT_OUT,
    Calibration,
    _black_image,
    _capture_background,
    _grab_frame,
    _open_camera,
    _project,
)
from irisloop.physical_reward import assess_physical
from irisloop.preferences import generate_pairs
from irisloop.reward_run import finalize_run, save_trial, start_run, write_json
from irisloop.scene_manifest import attach_run, register_scene
from irisloop.target_path import TargetPath, render_target
from tools.play_video_at import field_overlay, pick_lasso, pick_strokes


async def _load_or_draw_target(args, cli, cam, calibration) -> TargetPath | None:
    if os.path.exists(args.target_path) and not args.redraw:
        target = TargetPath.load(args.target_path)
        target.validate_calibration(args.calibration)
        print(f"=== TARGET {args.target_path} ({target.mode}) ===")
        return target

    background = await _capture_background(cli, cam)
    if background is None:
        raise RuntimeError("cannot capture target-picking background")
    wall = field_overlay(calibration, background)
    if args.target_mode == "stroke":
        selection = await pick_strokes(
            wall,
            "reward target",
            "draw reusable path; Enter=confirm, R=clear, Esc=cancel",
            args.preview_width,
        )
        paths = selection
    else:
        selection = await pick_lasso(
            wall,
            "reward target",
            "trace reusable shape; release to close; Esc=cancel",
        )
        paths = None if selection is None else [selection]
    if paths is None:
        return None
    target = TargetPath.create(
        paths,
        scene=args.scene,
        mode=args.target_mode,
        camera_size=(background.shape[1], background.shape[0]),
        calibration_path=args.calibration,
        note="Created by tools/reward_ab.py",
    )
    target.save(args.target_path)
    print(f"  reusable target -> {args.target_path}")
    return target


def _trial_order(widths: list[int], repeats: int, seed: int):
    rng = random.Random(seed)
    order = []
    for repeat in range(1, repeats + 1):
        arm = list(widths)
        rng.shuffle(arm)
        order.extend((width, repeat) for width in arm)
    return order


async def run(args) -> int:
    calibration = Calibration.load(args.calibration)
    from irisloop.session import open_slot

    opened = await open_slot(args.address)
    cli = opened.client
    print(f"=== CONNECT {opened.address} ===")
    cam = _open_camera(args.camera, args.cam_width, args.cam_height)
    print(f"  camera: {cam.info()}")
    await asyncio.sleep(1.0)
    try:
        target = await _load_or_draw_target(args, cli, cam, calibration)
        if target is None:
            print("  cancelled")
            return 1
        register_scene(
            args.manifest,
            scene=args.scene,
            calibration_path=args.calibration,
            target_path=args.target_path,
            surface=args.surface,
            distance_m=args.distance_m,
            ambient=args.ambient,
            camera=cam.info(),
        )

        variants = {
            width: render_target(target, calibration, width)
            for width in args.widths
        }
        config = {
            "widths_px": args.widths,
            "repeats": args.repeats,
            "trial_order": _trial_order(args.widths, args.repeats, args.seed),
            "threshold": args.threshold,
            "threshold_sweep": args.threshold_sweep,
            "tolerance_px": args.tolerance,
            "selected_tolerance_px": args.selected_tolerance,
            "halo_penalty": args.halo_penalty,
            "seed": args.seed,
            "camera": cam.info(),
            "camera_index": args.camera,
            "exposure": -7,
            "ble_mtu": cli.info.mtu,
        }
        run_dir = start_run(
            args.output_root,
            scene=args.scene,
            config=config,
            calibration_path=args.calibration,
            target_path=args.target_path,
        )
        print(f"=== RUN {run_dir} ===")

        trial_reports = []
        for index, (width, repeat) in enumerate(config["trial_order"], 1):
            trial_id = f"t{index:03d}_w{width:02d}_r{repeat}"
            rendered = variants[width]
            print(
                f"  [{index}/{len(config['trial_order'])}] "
                f"width={width}px repeat={repeat}"
            )
            background = await _capture_background(cli, cam)
            if background is None:
                raise RuntimeError(f"{trial_id}: background capture failed")
            await _project(cli, rendered.mask, settle=args.settle)
            capture = _grab_frame(cam)
            if capture is None:
                raise RuntimeError(f"{trial_id}: projected capture failed")
            metrics, observed = assess_physical(
                rendered.mask * 255,
                capture,
                background,
                calibration,
                threshold=args.threshold,
                tolerance_px=tuple(args.tolerance),
                selected_tolerance_px=args.selected_tolerance,
                halo_penalty=args.halo_penalty,
                threshold_sweep=tuple(args.threshold_sweep),
            )
            artifacts = f"trials/{trial_id}"
            trial = {
                "trial_id": trial_id,
                "variant_id": f"w{width:02d}",
                "requested_width_px": width,
                "repeat": repeat,
                "captured": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "target_pixels": rendered.mask_pixels,
                "metrics": metrics.to_dict(),
                "artifacts": {
                    "directory": artifacts,
                    "target": f"{artifacts}/target_mask.png",
                    "background": f"{artifacts}/background.png",
                    "capture": f"{artifacts}/capture.png",
                    "observed": f"{artifacts}/observed_mask.png",
                    "overlay": f"{artifacts}/overlay.png",
                },
            }
            save_trial(
                run_dir,
                trial_id,
                target=rendered.mask,
                background=background,
                capture=capture,
                observed=observed,
                report=trial,
            )
            trial_reports.append(trial)
            print(
                f"    reward={metrics.reward:.3f} "
                f"strict_iou={metrics.strict.iou:.3f} "
                f"centroid={metrics.centroid_error_px:.2f}px"
            )

        report = finalize_run(
            run_dir,
            scene=args.scene,
            config=config,
            trials=trial_reports,
        )
        winner = report["winner"]
        attach_run(args.manifest, args.scene, run_dir)
        write_json(os.path.join(run_dir, "preference_pairs.json"), generate_pairs(report))
        print(
            f"=== WINNER {winner['requested_width_px']}px "
            f"margin={winner['margin']:.3f} ==="
        )
        print(f"  report -> {os.path.join(run_dir, 'run_report.json')}")
        return 0
    finally:
        try:
            await _project(cli, _black_image(), settle=0.3)
        except Exception:
            pass
        cam.release()
        await opened.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="reward_ab")
    parser.add_argument("--address", default=None,
                        help="projector MAC; omit when only one IrisGreen is advertising")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cam-width", type=int, default=1280)
    parser.add_argument("--cam-height", type=int, default=720)
    parser.add_argument("--calibration", default=DEFAULT_OUT)
    parser.add_argument("--scene", default="current_wall")
    parser.add_argument("--manifest", default="scenes/manifest.json")
    parser.add_argument("--surface", default="wall poster")
    parser.add_argument("--distance-m", type=float)
    parser.add_argument("--ambient", default="")
    parser.add_argument("--target-path", default="scenes/current_wall/path.json")
    parser.add_argument("--target-mode", choices=["stroke", "shape"], default="stroke")
    parser.add_argument("--preview-width", type=int, default=8)
    parser.add_argument("--redraw", action="store_true")
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--settle", type=float, default=0.8)
    parser.add_argument("--threshold", type=int, default=40)
    parser.add_argument("--threshold-sweep", type=int, nargs="+", default=[30, 40, 50, 60])
    parser.add_argument("--tolerance", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--selected-tolerance", type=int, default=4)
    parser.add_argument("--halo-penalty", type=float, default=0.25)
    parser.add_argument("--output-root", default="captures/reward_runs")
    args = parser.parse_args(argv)
    if args.repeats < 1 or any(width < 1 for width in args.widths):
        parser.error("repeats and widths must be positive")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
