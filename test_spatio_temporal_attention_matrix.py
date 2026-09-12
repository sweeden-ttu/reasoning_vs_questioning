"""Unit tests for 3D Spatio-Temporal-Adversarial Attention Matrix Engine."""

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

from spatio_temporal_attention_matrix import (
    LEAGUE_OPPONENT_NAMES,
    OPPONENT_PROFILES,
    SpatioTemporalAttentionMatrix,
)
from market_config_suite import (
    build_initial_terminal_configuration,
    build_market_functions,
    build_opponent_functions,
)


def _make_mock_obs(step: int = 0) -> dict:
    day = step // 24
    hour = step % 24
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tiles[2][2] = {"kind": "PLANT", "crop": "WHEAT", "stage": 3, "moisture": 0.9}
    tiles[4][4] = {"kind": "PLANT", "crop": "TOMATO", "stage": 2, "moisture": 0.6}
    tiles[6][6] = {"kind": "WEED"}
    tiles[8][8] = {"kind": "PASTURE"}

    return {
        "step": step,
        "day": day,
        "hour": hour,
        "player": 0,
        "farms": [
            {
                "farmer": [2, 2],
                "hands": [[4, 4], [2, 3]],
                "money": 5400.0,
                "tiles": tiles,
                "unlocked_quadrants": ["NW", "NE"],
            },
            {
                "farmer": [1, 1],
                "hands": [],
                "money": 3200.0,
                "tiles": [[None for _ in range(10)] for _ in range(10)],
                "unlocked_quadrants": ["NW"],
            },
        ],
        "private": {
            "shed": {"TOMATO": 20, "WHEAT": 10},
            "seeds": {"TOMATO": 5, "WHEAT": 5},
        },
        "market": {
            "inventory": {"TOMATO": 9900, "WHEAT": 9800, "CARROT": 10000},
            "prices": {"TOMATO": 62, "WHEAT": 26, "CARROT": 35},
        },
        "town": {
            "unlocked_shops": ["PIZZA_SHOP", "FARMERS_MARKET"],
        },
    }


def test_attention_matrix_dimensions():
    # 30-day standard season: (10, 10, 30, 11) = 33,000 weights
    tensor_30 = SpatioTemporalAttentionMatrix(height=10, width=10, days=30, num_opponents=11)
    assert tensor_30.tensor_shape == (10, 10, 30, 11)
    assert tensor_30.flat_shape == (100, 30, 11)
    assert tensor_30.map_size == 100

    # 90-day macro curriculum: (10, 10, 90, 11) = 99,000 weights
    tensor_90 = SpatioTemporalAttentionMatrix(height=10, width=10, days=90, num_opponents=11)
    assert tensor_90.tensor_shape == (10, 10, 90, 11)
    assert tensor_90.flat_shape == (100, 90, 11)


def test_spatial_and_temporal_encodings():
    engine = SpatioTemporalAttentionMatrix(height=10, width=10, days=30, num_opponents=11, feature_dim=16)
    obs = _make_mock_obs(step=48)

    # Encode spatial tile query
    q_plant = engine.encode_spatial_query(obs, x=2, y=2)
    q_weed = engine.encode_spatial_query(obs, x=6, y=6)
    assert q_plant.shape == (16,)
    assert q_weed.shape == (16,)
    assert not np.allclose(q_plant, q_weed)

    # Encode temporal keys
    k_day0 = engine.encode_temporal_key(day=0, total_days=30)
    k_day28 = engine.encode_temporal_key(day=28, total_days=30)
    assert k_day0.shape == (16,)
    assert k_day28.shape == (16,)
    # Day 28 should have terminal liquidation feature active
    assert k_day28[2] > 0.0


def test_attention_tensor_computation_and_simplex_normalization():
    engine = SpatioTemporalAttentionMatrix(height=10, width=10, days=30, num_opponents=11)
    obs = _make_mock_obs(step=48)
    config = build_initial_terminal_configuration(obs)
    mkt_funcs = build_market_functions(obs, config)
    opp_funcs = build_opponent_functions(obs, config)

    att = engine.compute_attention_tensor(obs, config, mkt_funcs, opp_funcs)
    assert att.shape == (10, 10, 30, 11)
    assert np.all(att >= 0.0)

    # Spatial probability simplex check: for every (d, o), sum over (x, y) must equal 1.0
    for d in range(30):
        for o in range(11):
            spatial_sum = float(np.sum(att[:, :, d, o]))
            assert abs(spatial_sum - 1.0) < 1e-5, f"Spatial sum on day {d}, opp {o} was {spatial_sum}"


def test_adversarial_modulation_across_opponents():
    engine = SpatioTemporalAttentionMatrix(height=10, width=10, days=30, num_opponents=11)
    obs = _make_mock_obs(step=48)
    config = build_initial_terminal_configuration(obs)
    mkt_funcs = build_market_functions(obs, config)
    opp_funcs = build_opponent_functions(obs, config)

    att = engine.compute_attention_tensor(obs, config, mkt_funcs, opp_funcs)

    # Opponent 0 (Fallow Finn) should have lower adversarial pressure than Opponent 10 (Self-Play Anchor)
    adv_0 = engine.compute_adversarial_context(0, day=15, opponent_functions=opp_funcs)
    adv_10 = engine.compute_adversarial_context(10, day=15, opponent_functions=opp_funcs)
    assert adv_10 > adv_0


def test_attention_map_slicing_and_priority_tiles():
    engine = SpatioTemporalAttentionMatrix(height=10, width=10, days=30, num_opponents=11)
    obs = _make_mock_obs(step=48)
    config = build_initial_terminal_configuration(obs)
    mkt_funcs = build_market_functions(obs, config)
    opp_funcs = build_opponent_functions(obs, config)

    engine.compute_attention_tensor(obs, config, mkt_funcs, opp_funcs)

    # 1. Tile heatmap
    heatmap = engine.get_tile_attention_map(day=10, opponent_id=3)
    assert heatmap.shape == (10, 10)
    assert abs(float(np.sum(heatmap)) - 1.0) < 1e-5

    # 2. Temporal curve
    curve = engine.get_temporal_attention_curve(x=2, y=2, opponent_id=3)
    assert curve.shape == (30,)

    # 3. Opponent slice
    opp_slice = engine.get_opponent_attention_slice(opponent_id=5)
    assert opp_slice.shape == (10, 10, 30)

    # 4. Top-K priority tiles
    top_tiles = engine.get_highest_priority_tiles(day=10, opponent_id=3, top_k=5)
    assert len(top_tiles) == 5
    for x, y, weight in top_tiles:
        assert 0 <= x < 10
        assert 0 <= y < 10
        assert weight > 0.0

    # 5. Diagnostic summary
    summary = engine.summary()
    assert summary["is_computed"] is True
    assert summary["total_attention_weights"] == 33000
    assert len(summary["opponent_names"]) == 11
