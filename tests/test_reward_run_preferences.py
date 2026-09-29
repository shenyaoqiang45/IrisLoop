import json

import numpy as np

from irisloop.preferences import evaluate_preferences, generate_pairs
from irisloop.reward_run import finalize_run, save_trial, summarize_trials


def _trial(trial_id, width, repeat, reward):
    return {
        "trial_id": trial_id,
        "requested_width_px": width,
        "repeat": repeat,
        "metrics": {
            "reward": reward,
            "strict": {"iou": reward / 2},
            "selected_tolerance_px": 4,
            "tolerant": [{"radius_px": 4, "halo_ratio": 0.1}],
        },
        "artifacts": {"capture": f"trials/{trial_id}/capture.png"},
    }


def test_summary_and_winner(tmp_path):
    trials = [
        _trial("a1", 8, 1, 0.6),
        _trial("b1", 16, 1, 0.8),
        _trial("a2", 8, 2, 0.7),
        _trial("b2", 16, 2, 0.9),
    ]
    summary = summarize_trials(trials)
    assert [item["requested_width_px"] for item in summary] == [8, 16]
    report = finalize_run(
        str(tmp_path), scene="test", config={}, trials=trials
    )
    assert report["winner"]["requested_width_px"] == 16
    assert (tmp_path / "run_report.json").exists()


def test_blind_pairs_and_agreement():
    report = {
        "trials": [
            _trial("a1", 8, 1, 0.6),
            _trial("b1", 16, 1, 0.8),
            _trial("a2", 8, 2, 0.7),
            _trial("b2", 16, 2, 0.9),
        ]
    }
    pairs = generate_pairs(report, seed=1)
    votes = {"votes": []}
    for pair in pairs["pairs"]:
        winner = "left" if pair["left_trial"].startswith("b") else "right"
        votes["votes"].append({"pair_id": pair["pair_id"], "winner": winner})
    result = evaluate_preferences(report, pairs, votes)
    assert result["agreement"] == 1.0


def test_trial_archive_does_not_overwrite(tmp_path):
    image = np.zeros((20, 30), dtype=np.uint8)
    color = np.zeros((20, 30, 3), dtype=np.uint8)
    save_trial(
        str(tmp_path),
        "t001",
        target=image,
        background=color,
        capture=color,
        observed=image,
        report={"trial_id": "t001"},
    )
    assert (tmp_path / "trials" / "t001" / "trial.json").exists()
    try:
        save_trial(
            str(tmp_path),
            "t001",
            target=image,
            background=color,
            capture=color,
            observed=image,
            report={"trial_id": "t001"},
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("duplicate trial should not overwrite")
