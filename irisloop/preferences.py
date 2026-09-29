"""Blind pair generation and agreement metrics for physical reward runs."""

from __future__ import annotations

import itertools
import random

PAIR_SCHEMA = "irisloop.reward_pairs/1"
VOTE_SCHEMA = "irisloop.reward_votes/1"


def generate_pairs(run_report: dict, seed: int = 20260929) -> dict:
    by_repeat: dict[int, list[dict]] = {}
    for trial in run_report["trials"]:
        by_repeat.setdefault(int(trial["repeat"]), []).append(trial)
    pairs = []
    rng = random.Random(seed)
    index = 1
    for repeat, trials in sorted(by_repeat.items()):
        ordered = sorted(trials, key=lambda item: item["requested_width_px"])
        for first, second in itertools.combinations(ordered, 2):
            left, right = (first, second) if rng.random() < 0.5 else (second, first)
            pairs.append({
                "pair_id": f"p{index:03d}",
                "repeat": repeat,
                "left_trial": left["trial_id"],
                "right_trial": right["trial_id"],
                "left_capture": left["artifacts"]["capture"],
                "right_capture": right["artifacts"]["capture"],
            })
            index += 1
    rng.shuffle(pairs)
    return {"schema": PAIR_SCHEMA, "seed": seed, "pairs": pairs}


def evaluate_preferences(
    run_report: dict,
    pairs: dict,
    votes: dict,
    tie_epsilon: float = 1e-6,
) -> dict:
    rewards = {
        trial["trial_id"]: float(trial["metrics"]["reward"])
        for trial in run_report["trials"]
    }
    pair_map = {pair["pair_id"]: pair for pair in pairs["pairs"]}
    valid = 0
    agreements = 0
    human_ties = 0
    reward_ties = 0
    details = []
    for vote in votes["votes"]:
        pair = pair_map[vote["pair_id"]]
        human = vote["winner"]
        if human == "tie":
            human_ties += 1
            continue
        left_reward = rewards[pair["left_trial"]]
        right_reward = rewards[pair["right_trial"]]
        if abs(left_reward - right_reward) <= tie_epsilon:
            predicted = "tie"
            reward_ties += 1
        else:
            predicted = "left" if left_reward > right_reward else "right"
        valid += 1
        match = predicted == human
        agreements += int(match)
        details.append({
            "pair_id": pair["pair_id"],
            "human": human,
            "reward": predicted,
            "match": match,
            "left_reward": left_reward,
            "right_reward": right_reward,
        })
    total_votes = len(votes["votes"])
    return {
        "pairs_total": len(pairs["pairs"]),
        "votes_total": total_votes,
        "valid_non_tie": valid,
        "agreements": agreements,
        "agreement": agreements / valid if valid else None,
        "human_ties": human_ties,
        "human_tie_rate": human_ties / total_votes if total_votes else None,
        "reward_ties": reward_ties,
        "details": details,
    }
