"""Tests for single multi-head attention + 720×720 mask (30-day)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RVQ = Path(__file__).resolve().parent
ROOT = RVQ.parent
sys.path.insert(0, str(RVQ))
sys.path.insert(0, str(ROOT))

from single_mha_720_30day import (  # noqa: E402
    MASK_DIM,
    STATE_DIM,
    SingleMultiHeadAttention720,
    mask_row_to_spatial,
)


@pytest.fixture
def tiny_batch():
    rng = np.random.default_rng(0)
    n = 48
    X = rng.random((n, STATE_DIM), dtype=np.float32)
    y_act = rng.integers(0, 10, size=n)
    y_tile = rng.random(n, dtype=np.float32)
    y_liq = rng.random(n, dtype=np.float32)
    y_val = rng.random(n, dtype=np.float32) * 10000
    mask = rng.random((MASK_DIM, MASK_DIM), dtype=np.float32)
    return X, y_act, y_tile, y_liq, y_val, mask


def test_mask_row_to_spatial_shape():
    row = np.linspace(0, 1, MASK_DIM, dtype=np.float32)
    spatial = mask_row_to_spatial(row)
    assert spatial.shape == (10, 10)
    assert abs(float(spatial.sum()) - 1.0) < 1e-4


def test_fit_save_load_act(tiny_batch, tmp_path):
    X, y_act, y_tile, y_liq, y_val, mask = tiny_batch
    model = SingleMultiHeadAttention720()
    model.fit(X, y_act, y_tile, y_liq, y_val, mask, max_samples=48, verbose=False)
    assert model.is_fitted
    matrix = model.gated_attention_matrix(X[0], step=100)
    assert matrix.shape == (10, 10)
    assert abs(float(matrix.sum()) - 1.0) < 1e-4

    path = tmp_path / "single_mha.pkl"
    meta = model.save(path)
    assert meta["within_100mb"]
    assert meta["horizon_days"] == 30
    assert meta["final_model_size_mb"] < 10.0  # lean — no capacity bank

    loaded = SingleMultiHeadAttention720.load(path)
    obs = {
        "player": 0,
        "step": 100,
        "day": 4,
        "hour": 4,
        "money": [3000.0, 2800.0],
        "farms": [
            {"farmer": [0, 0], "hands": [[1, 0]], "tiles": []},
            {"farmer": [5, 5], "hands": [], "tiles": []},
        ],
        "private": {"shed": {"WHEAT": 2}, "seeds": {}},
        "market": {},
    }
    out = loaded.act(obs)
    assert "farmer" in out and "hands" in out and "market" in out
