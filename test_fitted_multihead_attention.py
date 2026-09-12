"""Tests for fitted multi-head attention matrix (120-day → ≤100 MB)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RVQ = Path(__file__).resolve().parent
ROOT = RVQ.parent
sys.path.insert(0, str(RVQ))
sys.path.insert(0, str(ROOT))

from fitted_multihead_attention import (  # noqa: E402
    STATE_DIM,
    FittedMultiHeadAttentionMatrix,
)


@pytest.fixture
def tiny_batch():
    rng = np.random.default_rng(0)
    n = 64
    X = rng.random((n, STATE_DIM), dtype=np.float32)
    y_act = rng.integers(0, 10, size=n)
    y_tile = rng.random(n, dtype=np.float32)
    y_liq = rng.random(n, dtype=np.float32)
    y_val = rng.random(n, dtype=np.float32) * 10000
    return X, y_act, y_tile, y_liq, y_val


def test_fit_and_attention_shapes(tiny_batch, tmp_path):
    X, y_act, y_tile, y_liq, y_val = tiny_batch
    mha = FittedMultiHeadAttentionMatrix(d_model=32, d_k=16)
    mha.fit(X, y_act, y_tile, y_liq, y_val, max_samples=64, verbose=False)
    maps = mha.attention_maps(X[0])
    assert maps.shape == (5, 10, 10)
    assert np.allclose(maps.sum(axis=(1, 2)), 1.0, atol=1e-4)
    fused = mha.fused_attention_matrix(X[0])
    assert fused.shape == (10, 10)
    assert abs(float(fused.sum()) - 1.0) < 1e-4


def test_scale_respects_100mb_ceiling(tiny_batch, tmp_path):
    X, y_act, y_tile, y_liq, y_val = tiny_batch
    mha = FittedMultiHeadAttentionMatrix(d_model=32, d_k=16)
    mha.fit(X, y_act, y_tile, y_liq, y_val, max_samples=64, verbose=False)
    path = tmp_path / "mha_test.pkl"
    # Small target for unit-test speed
    result = mha.scale_to_100mb(str(path), target_mb=1.5, verbose=False)
    assert path.exists()
    assert result["final_model_size_mb"] <= 100.0
    assert result["within_100mb"] is True
    loaded = FittedMultiHeadAttentionMatrix.load(path)
    assert loaded.is_fitted
    assert loaded.attention_maps(X[1]).shape == (5, 10, 10)
