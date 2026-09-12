"""Unit tests for TripartiteReasoningAgent: Decision = f(Questioning * Observer * Reasoning)."""

from __future__ import annotations

import numpy as np
import pytest

from agents.tripartite_composite_agent import (
    TripartiteDecisionContext,
    TripartiteDecisionEngine,
    TripartiteReasoningAgent,
)
from vector_memory_bank import DEFAULT_EMBEDDING_DIM


def _make_dummy_obs(
    day: int = 5,
    hour: int = 2,  # prime hour < 11
    p0_money: float = 3500.0,
    opp_money: float = 4000.0,
) -> dict:
    return {
        "day": day,
        "hour": hour,
        "player": 0,
        "farms": [
            {
                "money": p0_money,
                "hires_today": 0,
                "unlocked_quadrants": [0],
                "tiles": [
                    [{"kind": "PLANT", "stage": 2, "moisture": 0.8, "crop": "WHEAT"} for _ in range(10)]
                    for _ in range(10)
                ],
            },
            {
                "money": opp_money,
                "hires_today": 1,
                "unlocked_quadrants": [0],
                "tiles": [
                    [{"kind": "PLANT", "stage": 3, "crop": "WHEAT"} for _ in range(5)]
                    for _ in range(4)
                ] + [[{"kind": "EMPTY"} for _ in range(10)] for _ in range(6)],
            },
        ],
        "private": {
            "seeds": {"WHEAT": 0, "CORN": 2},
            "shed": {"WHEAT": 20},
        },
        "market": {
            "prices": {"WHEAT": 18.0, "CORN": 28.0},
            "inventory": {"WHEAT": 50, "CORN": 50},
        },
    }


def test_tripartite_decision_engine_synthesis():
    engine = TripartiteDecisionEngine(q_weight=0.5, o_weight=0.7, r_weight=1.0)
    obs = _make_dummy_obs(day=5, hour=2)

    reasoning_act = {"farmer": ["WATER"], "hands": [["PASS"]], "market": [["BUY_SEED", "WHEAT", 4]]}
    observer_insight = {
        "confident": True,
        "matched_archetype": "aggressive_wheat_rusher",
        "q_dot_k_score": 0.88,
        "recommended_action": {"market": [["BUY_SEED", "CORN", 4]]},
        "description": "Opponent rushes wheat; diversify to corn",
    }
    questioning_act = {"farmer": ["PASS"], "hands": [["PASS"]], "market": [["SELL", "WHEAT", 10]]}
    questioning_summary = {"confidence": 0.75, "text": "Opponent expanding farm"}

    final_action, rationale = engine.synthesize_decision(
        obs=obs,
        reasoning_act=reasoning_act,
        observer_insight=observer_insight,
        questioning_act=questioning_act,
        questioning_summary=questioning_summary,
    )

    assert isinstance(final_action, dict)
    assert final_action["farmer"] == ["WATER"]  # Base reasoning water action preserved
    # Market orders should contain both Observer counter-move (CORN) and Reasoning move (WHEAT) and Questioning sell
    market_ops = [op[0] for op in final_action["market"]]
    assert "BUY_SEED" in market_ops
    assert "SELL" in market_ops
    assert "Tripartite[Day 5 H2]" in rationale


def test_tripartite_reasoning_agent_act_cycle():
    agent = TripartiteReasoningAgent(memory_slots=10)
    agent.commit_game_identity()

    obs = _make_dummy_obs(day=3, hour=2, p0_money=3200.0, opp_money=2800.0)
    action = agent.act(obs)

    assert isinstance(action, dict)
    assert "farmer" in action
    assert "market" in action

    metrics = agent.metrics()
    assert metrics["tripartite_decisions_count"] == 1
    assert "Tripartite[Day 3 H2]" in metrics["last_decision_rationale"]
    assert metrics["tripartite_weights"]["q_weight"] == 0.5
    assert metrics["tripartite_weights"]["o_weight"] == 0.7
    assert metrics["tripartite_weights"]["r_weight"] == 1.0


def test_tripartite_schedule_enforcement():
    agent = TripartiteReasoningAgent(memory_slots=10)
    agent.commit_game_identity()

    # Hour 1 is non-prime; agent should PASS
    obs_pass = _make_dummy_obs(day=3, hour=1)
    act_pass = agent.act(obs_pass)
    assert act_pass == {"farmer": ["PASS"], "hands": [], "market": []}

    # Hour 3 is prime < 11; agent acts
    obs_act = _make_dummy_obs(day=3, hour=3)
    act_prime = agent.act(obs_act)
    assert act_prime != {"farmer": ["PASS"], "hands": [], "market": []} or len(agent.tripartite_audit_log) > 0
