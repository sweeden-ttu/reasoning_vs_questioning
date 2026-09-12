"""Unit and Integration Tests for 120-Day GBDT Scaling and 10x10x10 Voxel Tensor.

Verifies:
  1. VoxelStateExtractor generates (10, 10, 10) tensors and 1,035-dim state vectors.
  2. QKDGBDTPolicy multi-head ensemble fits, scales, and executes vector inference.
  3. QKDGBDTAgent enforces two-stage Att pipeline.
  4. GBDT120DayScalingTrainer harvests transitions and scales model payload.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
RVQ_DIR = Path(__file__).resolve().parent
if str(RVQ_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from market_config_suite import (
    InitialTerminalConfiguration,
    build_initial_terminal_configuration,
    build_market_functions,
    build_opponent_functions,
)
from voxel_state_extractor import (
    COMMODITIES_LIST,
    COMMODITY_TO_ID,
    VoxelStateExtractor,
)
from qkd_gbdt_vector_policy import (
    MACRO_ACTIONS,
    MACRO_ACTION_TO_ID,
    QKDGBDTPolicy,
)
from agents.qkd_gbdt_agent import QKDGBDTAgent, agent as qkd_gbdt_agent_fn
from gbdt_120day_scaling_trainer import GBDT120DayScalingTrainer


@pytest.fixture
def sample_120day_obs() -> dict:
    """Construct a rich 120-day observation for testing."""
    return {
        "player": 0,
        "step": 480,  # Day 20 of 120
        "farms": [
            {
                "money": 6500.0,
                "farmer_pos": [2, 3],
                "hands": [{"pos": [3, 3]}, {"pos": [2, 4]}],
                "warehouse": {"WHEAT": 50, "CARROT": 30, "MELON": 10},
                "grid": [
                    {"type": "SOIL", "crop": "WHEAT", "growth": 0.9, "moisture": 0.8, "has_weed": False}
                    for _ in range(100)
                ],
            },
            {
                "money": 4200.0,
                "farmer_pos": [5, 5],
                "hands": [],
                "warehouse": {"WHEAT": 20},
                "grid": [{"type": "SOIL"} for _ in range(100)],
            },
        ],
        "market": {
            "inventory": {"WHEAT": 8500, "CARROT": 9200, "MELON": 11000},
            "prices": {"WHEAT": 28.0, "CARROT": 38.0, "MELON": 230.0},
        },
        "town": {
            "unlocked_shops": ["BAKERY", "PET_CAFE", "FARMERS_MARKET"],
        },
    }


def test_voxel_state_extractor_dimensions(sample_120day_obs: dict):
    extractor = VoxelStateExtractor(grid_height=10, grid_width=10, depth_channels=10)
    config = InitialTerminalConfiguration(number_of_days=120, amount_of_money=3000.0)

    # 1. 3D Voxel Tensor
    voxel_tensor = extractor.extract_voxel_tensor(sample_120day_obs, config)
    assert voxel_tensor.shape == (10, 10, 10)
    assert voxel_tensor.dtype == np.float32
    assert np.all(voxel_tensor >= -1.0) and np.all(voxel_tensor <= 2.0)

    # 2. 3 QKD Canonical Probes
    qkd_probes = extractor.extract_qkd_probes(sample_120day_obs, config)
    assert qkd_probes.shape == (3,)
    assert qkd_probes.dtype == np.float32

    # 3. 32 Global Economic Features
    global_feats = extractor.extract_global_economic_features(sample_120day_obs, config)
    assert global_feats.shape == (32,)
    assert global_feats.dtype == np.float32

    # 4. Total 1035-dim State Vector
    full_vec = extractor.extract_full_state_vector(sample_120day_obs, config)
    assert full_vec.shape == (1035,)
    assert full_vec.dtype == np.float32


def test_qkd_gbdt_policy_lifecycle(sample_120day_obs: dict, tmp_path: Path):
    policy = QKDGBDTPolicy(n_estimators=15, max_depth=6, num_leaves=32)
    config = InitialTerminalConfiguration(number_of_days=120)

    # Synthetic training batch
    N = 100
    X = np.random.randn(N, 1035).astype(np.float32)
    y_act = np.random.randint(0, len(MACRO_ACTIONS), size=N).astype(np.int64)
    y_tile = np.random.rand(N).astype(np.float32)
    y_liq = np.random.rand(N).astype(np.float32)
    y_val = (10000.0 + np.random.randn(N) * 1000.0).astype(np.float32)

    policy.fit(X, y_act, y_tile, y_liq, y_val)
    assert policy.is_fitted

    # Action prediction
    act = policy.select_action(sample_120day_obs, config)
    assert "farmer" in act
    assert "hands" in act
    assert "market" in act
    assert len(act["hands"]) == 2

    # Save and reload
    save_file = str(tmp_path / "test_gbdt_policy.pkl")
    size_mb = policy.save(save_file)
    assert os.path.exists(save_file)
    assert size_mb > 0.0

    loaded_policy = QKDGBDTPolicy.load(save_file)
    assert loaded_policy.is_fitted
    act_loaded = loaded_policy.select_action(sample_120day_obs, config)
    assert "farmer" in act_loaded


def test_qkd_gbdt_agent_two_stage_att(sample_120day_obs: dict):
    agent_inst = QKDGBDTAgent()
    config = InitialTerminalConfiguration(number_of_days=120)
    market_fns = build_market_functions(sample_120day_obs, config)
    opp_fns = build_opponent_functions(sample_120day_obs, config)

    # Stage 1
    stage1_action = agent_inst.Att(sample_120day_obs, config, market_fns)
    assert "farmer" in stage1_action
    assert agent_inst._last_stage1_att is not None

    # Stage 2
    stage2_action = agent_inst.Att(sample_120day_obs, config, market_fns, opp_fns)
    assert "farmer" in stage2_action
    assert agent_inst._last_stage2_att is not None

    # Standard act
    direct_act = agent_inst.act(sample_120day_obs, config)
    assert "farmer" in direct_act


def test_gbdt_scaling_trainer_harvest(tmp_path: Path):
    model_path = str(tmp_path / "test_qkd_gbdt_scaled.pkl")
    trainer = GBDT120DayScalingTrainer(
        total_days=120,
        target_model_mb=3.0,
        model_save_path=model_path,
    )

    # Harvest from 2 opponents
    trainer.opponents = {
        k: v for i, (k, v) in enumerate(trainer.opponents.items()) if i < 2
    }

    X, y_act, y_tile, y_liq, y_val = trainer.harvest_120day_trajectories(
        episodes_per_opp=1,
        max_steps_per_match=48,  # Fast test match
        verbose=False,
    )

    assert len(X) > 0
    assert X.shape[1] == 1035
    assert len(y_act) == len(X)

    # Train and scale
    res = trainer.train_and_scale_gbdt_to_100mb(
        X, y_act, y_tile, y_liq, y_val, target_mb=3.0, verbose=False
    )
    assert os.path.exists(model_path)
    assert res["final_model_size_mb"] >= 2.5
    assert res["final_model_size_mb"] <= 100.0
