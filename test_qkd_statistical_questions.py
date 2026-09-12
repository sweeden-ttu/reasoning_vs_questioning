"""Unit tests for QKDStatisticalQuestionBank and Standard Deviation Ranking."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RVQ_DIR = Path(__file__).resolve().parent
if str(RVQ_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest

from qkd_statistical_questions import (
    CANONICAL_QKD_QUESTIONS,
    QKDStatisticalQuestionBank,
)


def _generate_synthetic_game_trajectory(n_steps: int = 100) -> list:
    """Generate synthetic trajectory of game observations across diverse phases."""
    trajectory = []
    for step in range(n_steps):
        day = (step // 24) + 1
        hour = step % 24
        # Economic growth curve with fluctuations
        p0_m = 3000.0 + (step * 85.0) + (np.sin(step) * 500.0)
        p1_m = 3000.0 + (step * 70.0) + (np.cos(step) * 400.0)

        # Dynamic market prices
        p_tom = 25.0 + (np.sin(step / 5.0) * 10.0)
        p_str = 50.0 + (np.cos(step / 7.0) * 20.0)
        p_mel = 80.0 + (np.sin(step / 3.0) * 30.0)

        obs = {
            "day": day,
            "hour": hour,
            "player": 0,
            "farms": [
                {
                    "money": p0_m,
                    "hands": [[4, 5] for _ in range(min(4, step // 20))],
                    "hires_today": min(8, step // 30),
                    "unlocked_quadrants": list(range(min(4, 1 + step // 50))),
                    "tiles": [
                        [
                            {
                                "kind": "PLANT" if (r + c) % 2 == 0 else "SOIL",
                                "stage": min(4, (step + r + c) % 5),
                                "moisture": 0.3 if (step + r) % 3 == 0 else 0.8,
                            }
                            for c in range(10)
                        ]
                        for r in range(10)
                    ],
                },
                {
                    "money": p1_m,
                    "hands": [],
                    "hires_today": min(8, step // 25),
                    "unlocked_quadrants": list(range(min(4, 1 + step // 40))),
                    "tiles": [
                        [{"kind": "PLANT", "stage": 3} for _ in range(5)]
                        for _ in range(4)
                    ],
                },
            ],
            "private": {
                "seeds": {"WHEAT": 4, "TOMATO": 2},
                "shed": {"WHEAT": step % 20, "TOMATO": step % 15},
            },
            "market": {
                "prices": {"TOMATO": p_tom, "STRAWBERRY": p_str, "MELON": p_mel},
                "inventory": {"TOMATO": 100, "WHEAT": 200},
            },
        }
        trajectory.append(obs)
    return trajectory


def test_qkd_statistical_question_bank_initialization():
    bank = QKDStatisticalQuestionBank()
    assert len(bank.questions) == 3
    qids = [q.qid for q in bank.questions]
    assert "Q_DAYS_REMAINING" in qids
    assert "K_OPP_WALLET_BALANCE" in qids
    assert "D_SUBAGENTS_WALLET_BALANCE" in qids


def test_qkd_question_variance_ranking():
    bank = QKDStatisticalQuestionBank()
    trajectory = _generate_synthetic_game_trajectory(n_steps=120)

    ranked_questions = bank.fit_sample_distribution(trajectory)
    assert len(ranked_questions) == len(CANONICAL_QKD_QUESTIONS)

    # Standard deviations must be non-negative and sorted in descending order
    stds = [q.std for q in ranked_questions]
    for s in stds:
        assert s >= 0.0
    assert stds == sorted(stds, reverse=True)

    top_k = bank.get_top_k_significant_questions(k=3)
    assert len(top_k) == 3


def test_summary_table_generation():
    bank = QKDStatisticalQuestionBank()
    trajectory = _generate_synthetic_game_trajectory(n_steps=50)
    bank.fit_sample_distribution(trajectory)

    table = bank.summary_table()
    assert isinstance(table, list)
    assert len(table) == 3

    # First row should be Rank 1 with highest std_deviation
    assert table[0]["rank"] == 1
    assert table[0]["std_deviation"] >= table[-1]["std_deviation"]
    for row in table:
        assert "qid" in row
        assert "channel" in row
        assert "std_deviation" in row
        assert "question" in row


def test_map_observation_to_questions():
    from qkd_statistical_questions import map_observation_to_questions

    obs = {
        "day": 6,
        "hour": 12,
        "player": 0,
        "farms": [
            {"money": 10000.0, "hands": [[1, 2], [3, 4]]},
            {"money": 8000.0, "hands": []},
        ],
    }
    mapped = map_observation_to_questions(obs)
    assert "Q_DAYS_REMAINING" in mapped
    assert "K_OPP_WALLET_BALANCE" in mapped
    assert "D_SUBAGENTS_WALLET_BALANCE" in mapped

    # Day 6 -> (30 - 6) / 30 = 24 / 30 = 0.8
    assert abs(mapped["Q_DAYS_REMAINING"] - 0.8) < 1e-5
    # Opp money 8000 -> log1p(8000) / 12 ~ 0.7489
    assert mapped["K_OPP_WALLET_BALANCE"] > 0.5
    # My money 10000 with 2 hands -> (log1p(10000)/12) * 1.2
    assert mapped["D_SUBAGENTS_WALLET_BALANCE"] > mapped["K_OPP_WALLET_BALANCE"]


def test_map_observation_to_qkd_structure():
    from qkd_statistical_questions import QKDObservationMap, map_observation_to_qkd

    obs = {
        "day": 15,
        "hour": 8,
        "player": 1,
        "farms": [
            {"money": 20000.0, "hands": []},
            {"money": 15000.0, "hands": [[0, 0]]},
        ],
    }
    qkd_map = map_observation_to_qkd(obs)
    assert isinstance(qkd_map, QKDObservationMap)
    assert qkd_map.day == 15
    assert qkd_map.hour == 8
    assert qkd_map.player == 1
    assert qkd_map.days_remaining == 15.0
    assert qkd_map.opp_wallet_balance == 20000.0
    assert abs(qkd_map.q_norm - 0.5) < 1e-5
    assert qkd_map.vector.shape == (3,)
    assert isinstance(qkd_map.to_dict(), dict)


def test_map_observations_trajectory():
    from qkd_statistical_questions import map_observations_trajectory

    trajectory = _generate_synthetic_game_trajectory(n_steps=24)
    mat = map_observations_trajectory(trajectory)
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (24, 3)
    assert np.all(mat >= 0.0)


# ── Edge Case & Robustness Testing: Null, Empty Strings, Word "empty" ─────────

def test_qkd_evaluators_with_null_none_and_empty_dict():
    """Verify individual question evaluators handle None, empty dicts, and missing fields."""
    from qkd_statistical_questions import (
        _eval_d_subagents_wallet_balance,
        _eval_k_opp_wallet_balance,
        _eval_q_days_remaining,
    )

    dummy_vec = np.zeros(128, dtype=np.float32)

    # 1. Null / None observation
    assert _eval_q_days_remaining(None, dummy_vec, dummy_vec) == 1.0  # day 0 -> (30-0)/30 = 1.0
    assert _eval_k_opp_wallet_balance(None, dummy_vec, dummy_vec) == 0.0
    assert _eval_d_subagents_wallet_balance(None, dummy_vec, dummy_vec) == 0.0

    # 2. Empty dict observation
    assert _eval_q_days_remaining({}, dummy_vec, dummy_vec) == 1.0
    assert _eval_k_opp_wallet_balance({}, dummy_vec, dummy_vec) == 0.0
    assert _eval_d_subagents_wallet_balance({}, dummy_vec, dummy_vec) == 0.0

    # 3. Explicit None values inside observation
    null_obs = {
        "day": None,
        "hour": None,
        "player": None,
        "farms": None,
        "market": None,
        "private": None,
    }
    assert _eval_q_days_remaining(null_obs, dummy_vec, dummy_vec) == 1.0
    assert _eval_k_opp_wallet_balance(null_obs, dummy_vec, dummy_vec) == 0.0
    assert _eval_d_subagents_wallet_balance(null_obs, dummy_vec, dummy_vec) == 0.0

    # 4. Farms with None elements
    null_farms_obs = {
        "day": 10,
        "player": 0,
        "farms": [None, None],
    }
    assert _eval_q_days_remaining(null_farms_obs, dummy_vec, dummy_vec) == pytest.approx((30 - 10) / 30.0)
    assert _eval_k_opp_wallet_balance(null_farms_obs, dummy_vec, dummy_vec) == 0.0
    assert _eval_d_subagents_wallet_balance(null_farms_obs, dummy_vec, dummy_vec) == 0.0


def test_qkd_evaluators_with_empty_strings():
    """Verify individual evaluators handle empty strings ('') in all fields gracefully."""
    from qkd_statistical_questions import (
        _eval_d_subagents_wallet_balance,
        _eval_k_opp_wallet_balance,
        _eval_q_days_remaining,
    )

    dummy_vec = np.zeros(128, dtype=np.float32)

    empty_str_obs = {
        "day": "",
        "hour": "",
        "player": "",
        "farms": [
            {"money": "", "hands": "", "tiles": ""},
            {"money": "", "hands": "", "tiles": ""},
        ],
        "market": {"prices": "", "inventory": ""},
        "private": {"seeds": "", "shed": ""},
    }

    q_res = _eval_q_days_remaining(empty_str_obs, dummy_vec, dummy_vec)
    k_res = _eval_k_opp_wallet_balance(empty_str_obs, dummy_vec, dummy_vec)
    d_res = _eval_d_subagents_wallet_balance(empty_str_obs, dummy_vec, dummy_vec)

    assert q_res == 1.0  # day parsed as 0.0 -> normalized 1.0
    assert k_res == 0.0  # opp money parsed as 0.0
    assert d_res == 0.0  # self money parsed as 0.0


def test_qkd_evaluators_with_word_empty():
    """Verify individual evaluators handle the word 'empty' / 'EMPTY' / 'Empty' across fields."""
    from qkd_statistical_questions import (
        _eval_d_subagents_wallet_balance,
        _eval_k_opp_wallet_balance,
        _eval_q_days_remaining,
    )

    dummy_vec = np.zeros(128, dtype=np.float32)

    word_empty_obs = {
        "day": "empty",
        "hour": "EMPTY",
        "player": "Empty",
        "farms": [
            {
                "money": "empty",
                "hands": "empty",
                "hires_today": "empty",
                "unlocked_quadrants": "empty",
                "tiles": [
                    [{"kind": "empty", "stage": "empty", "moisture": "empty"}],
                    [{"kind": "EMPTY", "stage": "EMPTY", "moisture": "EMPTY"}],
                ],
            },
            {
                "money": "EMPTY",
                "hands": "EMPTY",
                "tiles": "empty",
            },
        ],
        "market": {
            "prices": {"TOMATO": "empty", "WHEAT": "EMPTY"},
            "inventory": {"TOMATO": "empty", "WHEAT": "EMPTY"},
        },
        "private": {
            "seeds": {"WHEAT": "empty", "TOMATO": "EMPTY"},
            "shed": {"WHEAT": "empty", "TOMATO": "EMPTY"},
        },
    }

    q_res = _eval_q_days_remaining(word_empty_obs, dummy_vec, dummy_vec)
    k_res = _eval_k_opp_wallet_balance(word_empty_obs, dummy_vec, dummy_vec)
    d_res = _eval_d_subagents_wallet_balance(word_empty_obs, dummy_vec, dummy_vec)

    assert q_res == 1.0  # 'empty' parsed to 0.0 -> (30-0)/30 = 1.0
    assert k_res == 0.0  # 'EMPTY' parsed to 0.0
    assert d_res == 0.0  # 'empty' parsed to 0.0


def test_map_observation_to_questions_with_null_and_empty():
    """Test map_observation_to_questions with None, empty strings, and word 'empty'."""
    from qkd_statistical_questions import map_observation_to_questions

    # None input
    res_none = map_observation_to_questions(None)
    assert res_none["Q_DAYS_REMAINING"] == 1.0
    assert res_none["K_OPP_WALLET_BALANCE"] == 0.0
    assert res_none["D_SUBAGENTS_WALLET_BALANCE"] == 0.0

    # Empty dict
    res_empty_dict = map_observation_to_questions({})
    assert res_empty_dict["Q_DAYS_REMAINING"] == 1.0
    assert res_empty_dict["K_OPP_WALLET_BALANCE"] == 0.0
    assert res_empty_dict["D_SUBAGENTS_WALLET_BALANCE"] == 0.0

    # Word 'empty' in all keys
    res_word_empty = map_observation_to_questions({
        "day": "empty",
        "player": "EMPTY",
        "farms": ["empty", "empty"],
    })
    assert res_word_empty["Q_DAYS_REMAINING"] == 1.0
    assert res_word_empty["K_OPP_WALLET_BALANCE"] == 0.0
    assert res_word_empty["D_SUBAGENTS_WALLET_BALANCE"] == 0.0

    # Empty string in all keys
    res_empty_str = map_observation_to_questions({
        "day": "",
        "player": "",
        "farms": ["", ""],
    })
    assert res_empty_str["Q_DAYS_REMAINING"] == 1.0
    assert res_empty_str["K_OPP_WALLET_BALANCE"] == 0.0
    assert res_empty_str["D_SUBAGENTS_WALLET_BALANCE"] == 0.0


def test_map_observation_to_qkd_with_null_and_empty():
    """Test map_observation_to_qkd with None, empty strings, and word 'empty'."""
    from qkd_statistical_questions import QKDObservationMap, map_observation_to_qkd

    # 1. None
    qkd_none = map_observation_to_qkd(None)
    assert isinstance(qkd_none, QKDObservationMap)
    assert qkd_none.day == 0
    assert qkd_none.hour == 0
    assert qkd_none.player == 0
    assert qkd_none.days_remaining == 30.0
    assert qkd_none.opp_wallet_balance == 0.0
    assert qkd_none.subagents_wallet_balance == 0.0
    assert qkd_none.q_norm == 1.0
    assert qkd_none.k_norm == 0.0
    assert qkd_none.d_norm == 0.0
    assert qkd_none.vector.shape == (3,)
    np.testing.assert_allclose(qkd_none.vector, [1.0, 0.0, 0.0])

    # 2. Word 'empty'
    qkd_word_empty = map_observation_to_qkd({
        "day": "empty",
        "hour": "empty",
        "player": "empty",
        "farms": [{"money": "empty", "hands": "empty"}, {"money": "empty", "hands": "empty"}],
    })
    assert qkd_word_empty.day == 0
    assert qkd_word_empty.days_remaining == 30.0
    assert qkd_word_empty.opp_wallet_balance == 0.0
    assert qkd_word_empty.subagents_wallet_balance == 0.0
    np.testing.assert_allclose(qkd_word_empty.vector, [1.0, 0.0, 0.0])

    # 3. Empty strings
    qkd_empty_str = map_observation_to_qkd({
        "day": "",
        "hour": "",
        "player": "",
        "farms": [{"money": "", "hands": ""}],
    })
    assert qkd_empty_str.day == 0
    assert qkd_empty_str.days_remaining == 30.0
    np.testing.assert_allclose(qkd_empty_str.vector, [1.0, 0.0, 0.0])


def test_map_trajectory_with_null_and_empty():
    """Test map_observations_trajectory with empty sequence, None entries, and corrupted items."""
    from qkd_statistical_questions import map_observations_trajectory

    # 1. Empty list
    mat_empty = map_observations_trajectory([])
    assert mat_empty.shape == (0, 3)

    # 2. None argument
    mat_none = map_observations_trajectory(None)
    assert mat_none.shape == (0, 3)

    # 3. Sequence of None, empty dict, and word 'empty'
    corrupted_seq = [
        None,
        {},
        {"day": "empty", "player": "empty", "farms": "empty"},
        {"day": "", "player": "", "farms": ["", None]},
        {"day": 15, "farms": [{"money": "empty"}, {"money": 5000.0}]},
    ]
    mat = map_observations_trajectory(corrupted_seq)
    assert mat.shape == (5, 3)
    assert not np.any(np.isnan(mat))
    assert not np.any(np.isinf(mat))
    # Last row should have day 15 -> (30-15)/30 = 0.5
    assert mat[4, 0] == pytest.approx(0.5)
    # Opponent money 5000 -> log1p(5000)/12 ~ 0.7098
    assert mat[4, 1] > 0.6


def test_qkd_bank_fit_distribution_with_null_and_empty():
    """Test QKDStatisticalQuestionBank fitting with None, empty list, and corrupted inputs."""
    bank = QKDStatisticalQuestionBank()

    # 1. Empty list
    ranked_empty = bank.fit_sample_distribution([])
    assert len(ranked_empty) == 3

    # 2. None
    ranked_none = bank.fit_sample_distribution(None)
    assert len(ranked_none) == 3

    # 3. Trajectory containing None, empty strings, and 'empty' words
    mixed_trajectory = [
        None,
        {},
        {"day": "empty", "player": "empty"},
        {"day": "", "farms": [{"money": ""}, {"money": ""}]},
        {"day": 5, "farms": [{"money": 1000.0}, {"money": 2000.0}]},
        {"day": 20, "farms": [{"money": 15000.0, "hands": [[1, 1]]}, {"money": 12000.0}]},
    ]
    ranked = bank.fit_sample_distribution(mixed_trajectory)
    assert len(ranked) == 3
    for q in ranked:
        assert not np.isnan(q.mean)
        assert not np.isnan(q.std)
        assert not np.isnan(q.variance)
        assert q.sample_count == len(mixed_trajectory)

    # Summary table must generate cleanly without crashing
    table = bank.summary_table()
    assert len(table) == 3
    for row in table:
        assert isinstance(row["rank"], int)
        assert isinstance(row["std_deviation"], float)


