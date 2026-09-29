"""Tracked scene definitions for repeatable physical reward collection."""

from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np

from irisloop.target_path import file_sha256

SCHEMA = "irisloop.scene_manifest/1"


def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {"schema": SCHEMA, "updated": None, "scenes": []}
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"unsupported scene manifest schema: {data.get('schema')}")
    return data


def _save(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def calibration_metadata(path: str) -> dict:
    archive = np.load(path, allow_pickle=False)
    raw = json.loads(archive["meta"].item()) if "meta" in archive else {}
    return {
        "sha256": file_sha256(path),
        "created": raw.get("created"),
        "stats": raw.get("stats", {}),
    }


def register_scene(
    manifest_path: str,
    *,
    scene: str,
    calibration_path: str,
    target_path: str,
    surface: str = "",
    distance_m: float | None = None,
    ambient: str = "",
    camera: str = "",
) -> dict:
    scene_dir = os.path.join(os.path.dirname(manifest_path), scene)
    os.makedirs(scene_dir, exist_ok=True)
    calibration_copy = os.path.join(scene_dir, "homography.npz")
    path_copy = os.path.join(scene_dir, "path.json")
    shutil.copy2(calibration_path, calibration_copy)
    if os.path.abspath(target_path) != os.path.abspath(path_copy):
        shutil.copy2(target_path, path_copy)

    manifest = _load(manifest_path)
    record = {
        "scene": scene,
        "surface": surface,
        "distance_m": distance_m,
        "ambient": ambient,
        "camera": camera,
        "calibration": f"{scene}/homography.npz",
        "target_path": f"{scene}/path.json",
        "calibration_meta": calibration_metadata(calibration_path),
        "status": "ready",
        "runs": [],
    }
    previous = next(
        (item for item in manifest["scenes"] if item["scene"] == scene), None
    )
    if previous:
        record["runs"] = previous.get("runs", [])
        manifest["scenes"] = [
            record if item["scene"] == scene else item
            for item in manifest["scenes"]
        ]
    else:
        manifest["scenes"].append(record)
    _save(manifest_path, manifest)
    return record


def attach_run(manifest_path: str, scene: str, run_dir: str) -> None:
    manifest = _load(manifest_path)
    record = next(
        (item for item in manifest["scenes"] if item["scene"] == scene), None
    )
    if record is None:
        raise ValueError(f"scene is not registered: {scene}")
    normalized = run_dir.replace("\\", "/")
    if normalized not in record["runs"]:
        record["runs"].append(normalized)
    record["status"] = "collected"
    _save(manifest_path, manifest)
