#!/usr/bin/env python
"""Compare blind human pairwise votes with physical reward rankings."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from irisloop.preferences import evaluate_preferences
from irisloop.reward_run import write_json


def _load(path: str):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="eval_reward_preferences")
    parser.add_argument("run_dir")
    parser.add_argument("--annotator", default="user")
    parser.add_argument("--tie-epsilon", type=float, default=1e-6)
    args = parser.parse_args(argv)

    report = _load(os.path.join(args.run_dir, "run_report.json"))
    pairs = _load(os.path.join(args.run_dir, "preference_pairs.json"))
    votes = _load(os.path.join(args.run_dir, "votes", f"{args.annotator}.json"))
    result = evaluate_preferences(report, pairs, votes, args.tie_epsilon)
    output = os.path.join(
        args.run_dir, f"preference_agreement_{args.annotator}.json"
    )
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"  report -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
