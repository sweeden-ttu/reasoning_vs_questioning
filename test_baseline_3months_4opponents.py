"""Tests for 3-Month 4-Opponent Baseline Training & Metrics."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from agents.tripartite_composite_agent import TripartiteReasoningAgent
from train_baseline_3months_4opponents import (
    BaselineTrainer,
    TARGET_OPPONENTS,
)
from vector_memory_bank import QuantizedVectorMemoryBank


def test_target_opponents_list():
    assert len(TARGET_OPPONENTS) == 4
    assert "fallow_finn" in TARGET_OPPONENTS
    assert "wheat_walter" in TARGET_OPPONENTS
    assert "rotation_rosa" in TARGET_OPPONENTS
    assert "homestead_hana" in TARGET_OPPONENTS


def test_trainer_quick_smoke(tmp_path: Path):
    trainer = BaselineTrainer(
        output_dir=tmp_path,
        seasons_per_opp=1,
        turns_per_season=48,  # Quick 2-day smoke
        base_seed=100,
    )
    report = trainer.train_baseline()
    
    assert "training_summary" in report
    assert "opponent_breakdown" in report
    assert "vector_memory_bank" in report
    assert "hard_limits_compliance" in report

    assert report["vector_memory_bank"]["under_payload_ceiling"] is True
    assert (tmp_path / "policy_vector_bank.npz").exists()
    assert (tmp_path / "baseline_metrics.json").exists()


def test_tripartite_act_all_hours_flag():
    agent_comp = TripartiteReasoningAgent(act_all_hours=True)
    assert agent_comp.schedule_hours is None
    for h in range(24):
        assert agent_comp.may_act(h) is True

    agent_prime = TripartiteReasoningAgent(act_all_hours=False)
    assert agent_prime.schedule_hours == {2, 3, 5, 7}
    assert agent_prime.may_act(1) is False
    assert agent_prime.may_act(2) is True
