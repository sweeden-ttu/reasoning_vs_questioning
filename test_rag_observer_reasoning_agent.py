"""Unit tests for RAGObserverReasoningAgent with 2D K-Map Don't-Care Filtering."""

from __future__ import annotations

import numpy as np
import pytest

from agents.rag_observer_reasoning_agent import (
    RAGObserverReasoningAgent,
    build_2d_kmap_mask,
    build_opponent_kmap_mask,
)
from vector_memory_bank import DEFAULT_EMBEDDING_DIM, StateVectorEncoder


def _make_dummy_obs(
    day: int = 3,
    hour: int = 2,  # prime hour < 11
    opp_money: float = 2800.0,
    opp_hires: int = 3,
) -> dict:
    return {
        "day": day,
        "hour": hour,
        "player": 0,
        "farms": [
            {
                "money": 3000.0,
                "hires_today": 0,
                "unlocked_quadrants": [0],
                "tiles": [
                    [{"kind": "PLANT", "stage": 2, "moisture": 0.8} for _ in range(10)]
                    for _ in range(10)
                ],
            },
            {
                "money": opp_money,
                "hires_today": opp_hires,
                "unlocked_quadrants": [0, 1] if opp_money > 5000 else [0],
                "tiles": [
                    [{"kind": "PLANT", "stage": 3} for _ in range(5)]
                    for _ in range(4)
                ] + [[{"kind": "EMPTY"} for _ in range(10)] for _ in range(6)],
            },
        ],
        "private": {
            "seeds": {"WHEAT": 5, "CORN": 2},
            "shed": {"WHEAT": 10},
        },
        "market": {
            "prices": {"WHEAT": 15.0, "CORN": 25.0},
            "inventory": {"WHEAT": 100, "CORN": 50},
        },
    }


def test_2d_kmap_mask_structure():
    mask_2d = build_2d_kmap_mask(self_dim=128, opp_dim=128)
    assert mask_2d.shape == (128, 128)
    assert mask_2d.dtype == np.float32

    # Active cross-interaction blocks
    assert mask_2d[0, 0] == 1.0    # Calendar coupling
    assert mask_2d[8, 8] == 1.0    # Economic competition
    assert mask_2d[26, 16] == 1.0  # Shed inventory x Market price
    assert mask_2d[16, 36] == 1.0  # Seed purchase x Opponent plant stage
    assert mask_2d[36, 8] == 1.0   # Farm stage x Opponent land expansion

    # Don't-care cross cells (must be 0)
    assert mask_2d[100, 100] == 0.0 # Uncoupled spatial micro coordinates
    assert mask_2d[16, 8] == 0.0    # Seed count x Opponent money


def test_encoder_split_and_2d_kmap():
    encoder = StateVectorEncoder(dim=256, half_dim=128)
    obs = _make_dummy_obs(day=5, hour=4, opp_money=6000.0, opp_hires=2)

    s_self, s_opp = encoder.encode_split(obs)
    assert s_self.shape == (128,)
    assert s_opp.shape == (128,)

    # 2D K-map projection
    mask_2d = build_2d_kmap_mask(128, 128)
    fused = encoder.encode_2d_kmap(obs, mask_2d=mask_2d)
    assert fused.shape == (256,)
    assert np.isclose(np.linalg.norm(fused), 1.0, atol=1e-4)


def test_rag_observer_reasoning_agent_act():
    agent = RAGObserverReasoningAgent(memory_slots=10)
    agent.commit_game_identity()

    # Observation matching aggressive wheat rusher (day 3, hour 2 prime)
    obs = _make_dummy_obs(day=3, hour=2, opp_money=2800.0, opp_hires=3)

    action = agent.act(obs)
    assert isinstance(action, dict)
    assert "farmer" in action
    assert "market" in action

    # Verify RAG observation was logged
    metrics = agent.metrics()
    assert metrics["rag_observations_count"] >= 1
    assert metrics["vector_bank_size"] > 0
    assert metrics["kmap_2d_active_cells"] > 0
    assert metrics["kmap_2d_shape"] == [128, 128]

    # Verify memory protocol recorded a fact
    facts = sum(s.facts_stored for s in agent.bank.slots)
    assert facts >= 1

