#!/usr/bin/env python
"""Register calibration and target-path inputs as a reusable scene."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from irisloop.calibrate import DEFAULT_OUT
from irisloop.scene_manifest import register_scene


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="register_scene")
    parser.add_argument("scene")
    parser.add_argument("--calibration", default=DEFAULT_OUT)
    parser.add_argument("--target-path", required=True)
    parser.add_argument("--manifest", default="scenes/manifest.json")
    parser.add_argument("--surface", default="")
    parser.add_argument("--distance-m", type=float)
    parser.add_argument("--ambient", default="")
    parser.add_argument("--camera", default="")
    args = parser.parse_args(argv)
    record = register_scene(
        args.manifest,
        scene=args.scene,
        calibration_path=args.calibration,
        target_path=args.target_path,
        surface=args.surface,
        distance_m=args.distance_m,
        ambient=args.ambient,
        camera=args.camera,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
