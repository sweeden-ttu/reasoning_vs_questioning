"""Unit tests for Strategy 3: QuantizedVectorMemoryBank & Vector Retrieval."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from hard_limits import PROBABILISTIC_MODEL_MAX_BYTES, SUBMISSION_MAX_BYTES
from vector_memory_bank import (
    DEFAULT_EMBEDDING_DIM,
    QuantizedVectorMemoryBank,
    StateVectorEncoder,
    bootstrap_vector_bank_from_episodes,
)


def _make_dummy_obs(day: int = 1, hour: int = 6, p0_money: float = 3000.0) -> dict:
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
                    [{"kind": "PLANT", "stage": 2, "moisture": 0.8} for _ in range(10)]
                    for _ in range(10)
                ],
            },
            {
                "money": 3000.0,
                "hires_today": 0,
                "unlocked_quadrants": [0],
                "tiles": [[{"kind": "EMPTY"} for _ in range(10)] for _ in range(10)],
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


def test_state_vector_encoder():
    encoder = StateVectorEncoder(dim=DEFAULT_EMBEDDING_DIM)
    obs = _make_dummy_obs(day=5, hour=12, p0_money=4500.0)
    vec = encoder.encode(obs)

    assert vec.shape == (DEFAULT_EMBEDDING_DIM,)
    assert vec.dtype == np.float32
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_quantized_vector_bank_add_and_query():
    bank = QuantizedVectorMemoryBank(dim=DEFAULT_EMBEDDING_DIM)

    obs1 = _make_dummy_obs(day=1, hour=2, p0_money=3000.0)
    act1 = {"farmer": ["PLANT", "WHEAT"], "market": []}
    bank.add_entry(obs1, act1, strategic_value=1.5, archetype_tag="early_plant")

    obs2 = _make_dummy_obs(day=28, hour=22, p0_money=40000.0)
    act2 = {"farmer": ["PASS"], "market": [("SELL", "WHEAT", 100)]}
    bank.add_entry(obs2, act2, strategic_value=50.0, archetype_tag="late_sell")

    assert len(bank) == 2

    # Query with state identical to obs1
    results1 = bank.query_top_k(obs1, k=2)
    assert len(results1) == 2
    best_match, score = results1[0]
    assert best_match["archetype_tag"] == "early_plant"
    assert score > 0.95  # High cosine similarity

    # Query with state identical to obs2
    results2 = bank.query_top_k(obs2, k=2)
    best_match2, score2 = results2[0]
    assert best_match2["archetype_tag"] == "late_sell"
    assert score2 > 0.95


def test_vector_bank_serialization_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "policy_vector_bank.npz"
        bank = QuantizedVectorMemoryBank(dim=DEFAULT_EMBEDDING_DIM)

        for i in range(100):
            obs = _make_dummy_obs(day=i % 30, hour=i % 24, p0_money=1000.0 * (i + 1))
            act = {"farmer": ["PLANT" if i % 2 == 0 else "WATER"], "market": []}
            bank.add_entry(obs, act, strategic_value=float(i), archetype_tag=f"state_{i}")

        size_bytes = bank.save(save_path)
        assert save_path.exists()
        assert size_bytes <= SUBMISSION_MAX_BYTES  # Hard limit check (90 MB)

        # Load back
        loaded = QuantizedVectorMemoryBank.load(save_path)
        assert len(loaded) == 100
        assert loaded.dim == DEFAULT_EMBEDDING_DIM

        # Query loaded bank
        test_obs = _make_dummy_obs(day=10, hour=10, p0_money=11000.0)
        matches = loaded.query_top_k(test_obs, k=3)
        assert len(matches) == 3


def test_retrieval_speed_at_scale():
    """Verify that retrieval across 10,000 vectors completes well within < 5ms per turn."""
    bank = QuantizedVectorMemoryBank(dim=DEFAULT_EMBEDDING_DIM)
    n_vectors = 10_000
    random_vectors = np.random.randn(n_vectors, DEFAULT_EMBEDDING_DIM).astype(np.float32)
    dummy_actions = [{"farmer": ["PASS"], "market": []} for _ in range(n_vectors)]

    bank.add_batch(random_vectors, dummy_actions)
    assert len(bank) == n_vectors

    query_obs = _make_dummy_obs()

    # Warmup
    _ = bank.query_top_k(query_obs, k=5)

    # Benchmark 50 consecutive queries
    t0 = time.perf_counter()
    for _ in range(50):
        _ = bank.query_top_k(query_obs, k=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / 50.0

    print(f"\n10,000-vector query latency: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 5.0, f"Query latency {elapsed_ms:.2f}ms exceeds 5ms per-turn budget"
