"""Filesystem archive and aggregation for physical reward experiments."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import defaultdict

import cv2
import numpy as np

RUN_SCHEMA = "irisloop.reward_run/1"
TRIAL_SCHEMA = "irisloop.reward_trial/1"


def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def write_png(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    encoded.tofile(path)


def allocate_run_dir(root: str, scene: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(root, f"{stamp}_{scene}")
    path = base
    suffix = 2
    while os.path.exists(path):
        path = f"{base}_{suffix}"
        suffix += 1
    os.makedirs(os.path.join(path, "trials"), exist_ok=False)
    return path


def start_run(
    root: str,
    *,
    scene: str,
    config: dict,
    calibration_path: str,
    target_path: str,
) -> str:
    run_dir = allocate_run_dir(root, scene)
    shutil.copy2(calibration_path, os.path.join(run_dir, "homography.npz"))
    shutil.copy2(target_path, os.path.join(run_dir, "target_path.json"))
    write_json(
        os.path.join(run_dir, "00_run_config.json"),
        {
            "schema": RUN_SCHEMA,
            "scene": scene,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **config,
        },
    )
    return run_dir


def overlay_masks(target: np.ndarray, observed: np.ndarray) -> np.ndarray:
    target = target.astype(bool)
    observed = observed.astype(bool)
    image = np.zeros((*target.shape, 3), dtype=np.uint8)
    image[target, 1] = 255
    image[observed, 2] = 255
    image[target & observed] = (0, 255, 255)
    return image


def save_trial(
    run_dir: str,
    trial_id: str,
    *,
    target: np.ndarray,
    background: np.ndarray,
    capture: np.ndarray,
    observed: np.ndarray,
    report: dict,
) -> str:
    trial_dir = os.path.join(run_dir, "trials", trial_id)
    os.makedirs(trial_dir, exist_ok=False)
    write_png(os.path.join(trial_dir, "target_mask.png"), target * 255)
    write_png(os.path.join(trial_dir, "background.png"), background)
    write_png(os.path.join(trial_dir, "capture.png"), capture)
    write_png(os.path.join(trial_dir, "observed_mask.png"), observed * 255)
    write_png(
        os.path.join(trial_dir, "overlay.png"),
        overlay_masks(target, observed),
    )
    report = {"schema": TRIAL_SCHEMA, **report}
    write_json(os.path.join(trial_dir, "trial.json"), report)
    return os.path.relpath(trial_dir, run_dir).replace("\\", "/")


def summarize_trials(trials: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for trial in trials:
        grouped[int(trial["requested_width_px"])].append(trial)
    summary = []
    for width, rows in sorted(grouped.items()):
        rewards = np.asarray([row["metrics"]["reward"] for row in rows], dtype=float)
        strict = np.asarray(
            [row["metrics"]["strict"]["iou"] for row in rows], dtype=float
        )
        halo = np.asarray([
            next(
                item["halo_ratio"]
                for item in row["metrics"]["tolerant"]
                if item["radius_px"] == row["metrics"]["selected_tolerance_px"]
            )
            for row in rows
        ])
        summary.append({
            "variant_id": f"w{width:02d}",
            "requested_width_px": width,
            "n": len(rows),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "strict_iou_mean": float(strict.mean()),
            "halo_ratio_mean": float(halo.mean()),
        })
    return summary


def finalize_run(
    run_dir: str,
    *,
    scene: str,
    config: dict,
    trials: list[dict],
) -> dict:
    summary = summarize_trials(trials)
    ranked = sorted(summary, key=lambda row: row["reward_mean"], reverse=True)
    winner = None
    if ranked:
        margin = (
            ranked[0]["reward_mean"] - ranked[1]["reward_mean"]
            if len(ranked) > 1 else 0.0
        )
        winner = {
            "variant_id": ranked[0]["variant_id"],
            "requested_width_px": ranked[0]["requested_width_px"],
            "margin": float(margin),
        }
    report = {
        "schema": RUN_SCHEMA,
        "scene": scene,
        "completed": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": config,
        "trials": trials,
        "summary": summary,
        "winner": winner,
    }
    write_json(os.path.join(run_dir, "run_report.json"), report)
    return report
