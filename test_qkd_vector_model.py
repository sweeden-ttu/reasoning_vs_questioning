"""Unit tests for QKDVectorModel: Questions + K-Map Observations + Decisions."""

from __future__ import annotations

import os
import time
from pathlib import Path
import numpy as np
import pytest

from qkd_vector_model import QKDVectorModel, QKDQueryInsight
from hard_limits import SUBMISSION_MAX_BYTES


def test_qkd_vector_model_initialization():
    model = QKDVectorModel(dim=256)
    assert len(model.q_store) >= 4  # Seeded questions
    assert len(model.k_store) >= 8  # Seeded K-map archetypes
    assert len(model.d_store) >= 4  # Seeded decisions
    assert model.total_vectors() >= 16
    assert model.kmap_2d_mask.shape == (128, 128)
    assert np.sum(model.kmap_2d_mask) > 0


def test_qkd_query_joint_insight():
    model = QKDVectorModel(dim=256)
    obs = {
        "day": 12,
        "hour": 4,
        "farms": [{"money": 8000.0}, {"money": 11000.0, "hires_today": 8}],
        "market": {"prices": {"TOMATO": 35.0}},
    }

    insight = model.query_qkd(obs)
    assert isinstance(insight, QKDQueryInsight)
    assert insight.day == 12
    assert insight.hour == 4
    assert insight.q_confidence > 0.0
    assert insight.k_dot_score >= 0.0
    assert "QKD[" in insight.synthesized_rationale
    assert "K_2D(" in insight.synthesized_rationale


def test_qkd_serialization_roundtrip(tmp_path: Path):
    model = QKDVectorModel(dim=256)
    # Add an experience
    model.add_experience(
        obs={"day": 5, "hour": 2, "farms": [{"money": 4000.0}, {"money": 3000.0}]},
        action={"farmer": ["WATER"], "market": [["HIRE"]]},
        reward=2.5,
        opp_name="wheat_walter",
        question_text="Will Walter early-harvest?",
        hypothesis="walter_early_harvest",
        confidence=0.88,
    )

    save_path = tmp_path / "qkd_test_model.npz"
    model.save(save_path)
    assert save_path.exists()

    size_bytes = os.path.getsize(save_path)
    assert size_bytes <= SUBMISSION_MAX_BYTES  # < 90 MB

    # Restore
    loaded_model = QKDVectorModel.load(save_path)
    assert loaded_model.dim == 256
    assert loaded_model.kmap_2d_mask.shape == (128, 128)
    assert np.allclose(loaded_model.kmap_2d_mask, model.kmap_2d_mask)
    assert len(loaded_model.q_store) == len(model.q_store)
    assert len(loaded_model.k_store) == len(model.k_store)
    assert len(loaded_model.d_store) == len(model.d_store)
    assert loaded_model.total_vectors() == model.total_vectors()


def test_qkd_retrieval_latency():
    model = QKDVectorModel(dim=256)
    obs = {"day": 10, "hour": 6, "farms": [{"money": 5000.0}, {"money": 5000.0}]}

    t0 = time.perf_counter()
    for _ in range(500):
        _ = model.query_qkd(obs)
    t1 = time.perf_counter()
    latency_ms = ((t1 - t0) / 500) * 1000

    assert latency_ms < 5.0  # Must be strictly < 5 ms, target < 1 ms

