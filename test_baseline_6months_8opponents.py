"""Tests for 6-Month 8-Opponent Baseline Training & Metrics."""

from __future__ import annotations

from pathlib import Path
import pytest

from agents.tripartite_composite_agent import TripartiteReasoningAgent
from train_qkd_model_6months_8opponents import (
    QKDBaselineTrainer,
    TARGET_OPPONENTS_8,
)
from vector_memory_bank import QuantizedVectorMemoryBank


def test_target_opponents_8_list():
    assert len(TARGET_OPPONENTS_8) == 8
    expected = [
        "fallow_finn",
        "wheat_walter",
        "rotation_rosa",
        "homestead_hana",
        "melon_mateo",
        "rancher_rita",
        "broker_bea",
        "closer_cleo",
    ]
    for opp in expected:
        assert opp in TARGET_OPPONENTS_8


def test_tripartite_8opp_archetypes_loaded():
    agent = TripartiteReasoningAgent(act_all_hours=True)
    assert len(agent.vector_bank) >= 8
    tags = [entry.get("archetype_tag") for entry in agent.vector_bank.entries]
    assert "melon_monopolist" in tags or any("melon" in t for t in tags)
    assert "livestock_rancher" in tags or any("rancher" in t or "livestock" in t for t in tags)
    assert "meta_opportunistic_broker" in tags or any("broker" in t for t in tags)
    assert "meta_queue_closer" in tags or any("closer" in t for t in tags)


def test_qkd_baseline_trainer_smoke(tmp_path: Path):
    trainer = QKDBaselineTrainer(
        output_dir=tmp_path,
        seasons_per_opp=1,
        turns_per_season=24,  # Quick 1-day smoke
        base_seed=100,
    )
    report = trainer.train_baseline()
    assert report["training_summary"]["overall_win_rate"] >= 0.0
    assert report["training_summary"]["total_matches"] == 16  # 8 opps x 1 season x 2 seats
    assert (tmp_path / "qkd_model.npz").exists()
    assert (tmp_path / "baseline_metrics.json").exists()

