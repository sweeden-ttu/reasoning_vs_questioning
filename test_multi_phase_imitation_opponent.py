"""Unit tests for MultiPhaseImitationOpponent."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
RVQ_DIR = Path(__file__).resolve().parent
if str(RVQ_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.multi_phase_imitation_opponent import (
    MultiPhaseImitationOpponent,
    MultiPhaseImitationOpponentAgent,
    get_multi_phase_opponent,
    multi_phase_imitation_agent,
)


def _make_mock_observation(
    day: int = 1,
    hour: int = 0,
    player: int = 0,
    money: float = 3000.0,
    weeds: bool = False,
) -> Dict[str, Any]:
    """Create a standardized mock observation dictionary."""
    tiles = [
        [
            {
                "kind": "WEED" if (weeds and r == 0 and c == 0) else "SOIL",
                "moisture": 0.8,
                "stage": 0,
            }
            for c in range(10)
        ]
        for r in range(10)
    ]
    return {
        "day": day,
        "hour": hour,
        "player": player,
        "farms": [
            {
                "money": money,
                "hands": [],
                "hires_today": 0,
                "unlocked_quadrants": [0],
                "tiles": tiles,
            },
            {
                "money": 3000.0,
                "hands": [],
                "hires_today": 0,
                "unlocked_quadrants": [0],
                "tiles": tiles,
            },
        ],
        "private": {
            "seeds": {"WHEAT": 5, "MELON": 2},
            "shed": {"WHEAT": 25, "MELON": 15},
        },
        "market": {
            "prices": {"WHEAT": 25.0, "MELON": 130.0},
            "inventory": {"WHEAT": 500, "MELON": 200},
        },
    }


def test_multi_phase_opponent_initialization():
    opp = MultiPhaseImitationOpponent()
    assert opp.current_iteration == 1
    assert len(opp.replay_p0) > 0
    assert len(opp.replay_p1) > 0
    assert opp.subagent_2 is not None
    assert opp.subagent_7 is not None
    assert opp.opponent_4_fn is not None


def test_phase_1_player_0_imitation():
    """Iteration 1: Imitates Player 0 from the first half of the replay buffer."""
    custom_p0 = [{"farmer": ["MOVE_N", i], "hands": [], "market": []} for i in range(20)]
    opp = MultiPhaseImitationOpponent(
        replay_buffer_p0=custom_p0,
        start_iteration=1,
    )
    phase_name, phase_idx = opp.determine_active_phase(0, 1)
    assert phase_name == "player_0_replay_half"
    assert phase_idx == 1

    # First half of 20 elements = 10 elements (indices 0..9)
    obs = _make_mock_observation(day=1, hour=0)
    action_0 = opp.act(obs)
    assert action_0["farmer"] == ["MOVE_N", 0]

    obs_step_5 = _make_mock_observation(day=1, hour=5)
    action_5 = opp.act(obs_step_5)
    # step_idx = (1 - 1) * 24 + 5 = 5 -> 5 % 10 = 5
    assert action_5["farmer"] == ["MOVE_N", 5]


def test_phase_2_player_1_imitation():
    """Iteration 2: Imitates Player 1 from the replay buffer."""
    custom_p1 = [{"farmer": ["MOVE_S", i], "hands": [], "market": []} for i in range(20)]
    opp = MultiPhaseImitationOpponent(
        replay_buffer_p1=custom_p1,
        start_iteration=2,
    )
    phase_name, phase_idx = opp.determine_active_phase(0, 1)
    assert phase_name == "player_1_replay"
    assert phase_idx == 2

    obs = _make_mock_observation(day=1, hour=3)
    action = opp.act(obs)
    # step_idx = (1 - 1) * 24 + 3 = 3 -> 3 % 20 = 3
    assert action["farmer"] == ["MOVE_S", 3]


def test_phase_3_subagent_2_imitation():
    """Iteration 3: Imitates Subagent 2 (Scott Weeden Referee/Auditor)."""
    opp = MultiPhaseImitationOpponent(start_iteration=3)
    phase_name, phase_idx = opp.determine_active_phase(0, 1)
    assert phase_name == "subagent_2_referee"
    assert phase_idx == 3

    # With weeds on (0, 0), Subagent 2 should target the weed with DIG
    obs_weed = _make_mock_observation(day=2, hour=4, weeds=True)
    action_weed = opp.act(obs_weed)
    assert action_weed["farmer"] == ["DIG", 0, 0]
    assert "advice" in action_weed
    assert action_weed["advice"]["slot"] == 2


def test_phase_4_interim_transition():
    """Iteration 4: Intermediate transition policy."""
    opp = MultiPhaseImitationOpponent(start_iteration=4)
    phase_name, phase_idx = opp.determine_active_phase(0, 1)
    assert phase_name == "interim_transition"
    assert phase_idx == 4

    obs = _make_mock_observation(day=1, hour=0)
    action = opp.act(obs)
    assert "farmer" in action
    assert "market" in action


def test_phase_5_subagent_7_imitation():
    """Iteration 5: Imitates Subagent 7 (Land Expansion & Weed Suppression)."""
    opp = MultiPhaseImitationOpponent(start_iteration=5)
    phase_name, phase_idx = opp.determine_active_phase(0, 1)
    assert phase_name == "subagent_7_expansion"
    assert phase_idx == 5

    # With money >= 1000, Subagent 7 should unlock NE quadrant
    obs_rich = _make_mock_observation(day=5, hour=8, money=1500.0)
    action_rich = opp.act(obs_rich)
    assert ["UNLOCK", "NE"] in action_rich["market"]
    assert "advice" in action_rich
    assert action_rich["advice"]["slot"] == 7


def test_phase_6_opponent_4_imitation():
    """Iteration 6: Imitates Opponent 4 (Melon Mateo Tier 4 policy)."""
    opp = MultiPhaseImitationOpponent(start_iteration=6)
    phase_name, phase_idx = opp.determine_active_phase(0, 1)
    assert phase_name == "opponent_4_melon"
    assert phase_idx == 6

    obs = _make_mock_observation(day=10, hour=12, money=2000.0)
    action = opp.act(obs)
    assert "farmer" in action
    assert "market" in action


def test_iteration_progression_and_cycling():
    """Verify set_iteration, next_iteration, and >6 cycling."""
    opp = MultiPhaseImitationOpponent(start_iteration=1)
    assert opp.get_current_phase()["active_phase"] == "player_0_replay_half"

    opp.next_iteration()
    assert opp.get_current_phase()["active_phase"] == "player_1_replay"

    opp.next_iteration()
    assert opp.get_current_phase()["active_phase"] == "subagent_2_referee"

    opp.set_iteration(5)
    assert opp.get_current_phase()["active_phase"] == "subagent_7_expansion"

    opp.set_iteration(6)
    assert opp.get_current_phase()["active_phase"] == "opponent_4_melon"

    # Iteration 7 cycles back to 1 (player 0)
    opp.set_iteration(7)
    assert opp.get_current_phase()["active_phase"] == "player_0_replay_half"


def test_step_phased_intra_match_mode():
    """Verify intra-match step-phased mode transitions across 720 turns."""
    opp = MultiPhaseImitationOpponent(mode="step_phased", total_match_steps=720)

    # Step 50 (< 180): Phase 1 (Player 0)
    p_50, idx_50 = opp.determine_active_phase(50, 2)
    assert p_50 == "player_0_replay_half"
    assert idx_50 == 1

    # Step 250 (180..360): Phase 2 (Player 1)
    p_250, idx_250 = opp.determine_active_phase(250, 10)
    assert p_250 == "player_1_replay"
    assert idx_250 == 2

    # Step 400 (360..504): Phase 3 (Subagent 2)
    p_400, idx_400 = opp.determine_active_phase(400, 16)
    assert p_400 == "subagent_2_referee"
    assert idx_400 == 3

    # Step 600 (576..648): Phase 5 (Subagent 7)
    p_600, idx_600 = opp.determine_active_phase(600, 25)
    assert p_600 == "subagent_7_expansion"
    assert idx_600 == 5

    # Step 700 (>= 648): Phase 6 (Opponent 4)
    p_700, idx_700 = opp.determine_active_phase(700, 29)
    assert p_700 == "opponent_4_melon"
    assert idx_700 == 6


def test_edge_cases_with_null_and_empty():
    """Verify robustness when observation is None, empty dict, or strings like 'empty'."""
    opp = MultiPhaseImitationOpponent(start_iteration=3)

    # 1. None observation
    act_none = opp.act(None)
    assert isinstance(act_none, dict)
    assert "farmer" in act_none

    # 2. Empty dict
    act_empty = opp.act({})
    assert isinstance(act_empty, dict)

    # 3. Observation with 'empty' string values
    act_str_empty = opp.act({
        "day": "empty",
        "hour": "empty",
        "player": "empty",
        "farms": "empty",
    })
    assert isinstance(act_str_empty, dict)


def test_callable_entrypoint():
    """Test global multi_phase_imitation_agent entrypoint."""
    obs = _make_mock_observation(day=3, hour=6)
    action = multi_phase_imitation_agent(obs)
    assert isinstance(action, dict)
    assert "farmer" in action
    assert "hands" in action
    assert "market" in action
