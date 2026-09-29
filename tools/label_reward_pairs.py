#!/usr/bin/env python
"""Blindly label which physical projection capture looks better."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from irisloop.preferences import VOTE_SCHEMA, generate_pairs
from irisloop.reward_run import write_json


def _load_json(path: str):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _read(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def _panel(left: np.ndarray, right: np.ndarray, progress: str) -> np.ndarray:
    height = 520
    def resize(image):
        scale = height / image.shape[0]
        return cv2.resize(image, (round(image.shape[1] * scale), height))
    left, right = resize(left), resize(right)
    canvas = cv2.hconcat([left, right])
    cv2.putText(canvas, "LEFT [1]", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(canvas, "RIGHT [2]", (left.shape[1] + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(canvas, f"{progress} | 0=tie  q=save+quit", (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return canvas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="label_reward_pairs")
    parser.add_argument("run_dir")
    parser.add_argument("--annotator", default="user")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)

    report = _load_json(os.path.join(args.run_dir, "run_report.json"))
    pairs_path = os.path.join(args.run_dir, "preference_pairs.json")
    if os.path.exists(pairs_path):
        pairs = _load_json(pairs_path)
    else:
        pairs = generate_pairs(report, args.seed)
        write_json(pairs_path, pairs)

    votes_dir = os.path.join(args.run_dir, "votes")
    votes_path = os.path.join(votes_dir, f"{args.annotator}.json")
    if os.path.exists(votes_path):
        votes = _load_json(votes_path)
    else:
        votes = {
            "schema": VOTE_SCHEMA,
            "annotator": args.annotator,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "votes": [],
        }
    completed = {vote["pair_id"] for vote in votes["votes"]}
    remaining = [pair for pair in pairs["pairs"] if pair["pair_id"] not in completed]

    cv2.namedWindow("reward preference", cv2.WINDOW_NORMAL)
    try:
        for index, pair in enumerate(remaining, 1):
            left = _read(os.path.join(args.run_dir, pair["left_capture"]))
            right = _read(os.path.join(args.run_dir, pair["right_capture"]))
            cv2.imshow(
                "reward preference",
                _panel(left, right, f"{index}/{len(remaining)}"),
            )
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("1"), ord("2"), ord("0"), ord("q"), 27):
                    break
            if key in (ord("q"), 27):
                break
            winner = {"1": "left", "2": "right", "0": "tie"}[chr(key)]
            votes["votes"].append({
                "pair_id": pair["pair_id"],
                "winner": winner,
                "labeled": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            write_json(votes_path, votes)
    finally:
        cv2.destroyAllWindows()
    write_json(votes_path, votes)
    print(f"  votes: {len(votes['votes'])}/{len(pairs['pairs'])} -> {votes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
