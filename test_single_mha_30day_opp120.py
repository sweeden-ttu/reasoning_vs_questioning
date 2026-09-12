"""Tests: 30-day single MHA + 120×120 opponent confusion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RVQ = Path(__file__).resolve().parent
ROOT = RVQ.parent
sys.path.insert(0, str(RVQ))
sys.path.insert(0, str(ROOT))

from single_mha_30day_opp120 import (  # noqa: E402
    CM_DIM,
    N_OPP_CLASSES,
    STATE_DIM,
    SingleMultiHeadAttentionOpp120,
    build_opponent_confusion_120,
    confusion_row_to_spatial,
    opp_class_id,
)


def test_opp_class_space():
    assert N_OPP_CLASSES == 120
    assert opp_class_id(0, 0) == 0
    assert opp_class_id(29, 11) == 119
    cm = build_opponent_confusion_120()
    assert cm.shape == (120, 120)
    assert np.allclose(cm.sum(axis=1), 1.0, atol=1e-4)
    spatial = confusion_row_to_spatial(cm[0])
    assert spatial.shape == (10, 10)


def test_fit_and_act(tmp_path):
    rng = np.random.default_rng(0)
    n = 64
    X = rng.random((n, STATE_DIM), dtype=np.float32)
    y_act = rng.integers(0, 10, size=n)
    y_tile = rng.random(n, dtype=np.float32)
    y_liq = rng.random(n, dtype=np.float32)
    y_val = rng.random(n, dtype=np.float32) * 1000
    model = SingleMultiHeadAttentionOpp120()
    model.fit(X, y_act, y_tile, y_liq, y_val, max_samples=64, verbose=False)
    assert model.confusion_120.shape == (CM_DIM, CM_DIM)
    assert model.horizon_days == 30
    matrix = model.gated_attention_matrix(X[0])
    assert matrix.shape == (10, 10)
    path = tmp_path / "m.pkl"
    meta = model.save(path)
    assert meta["confusion_shape"] == [120, 120]
    loaded = SingleMultiHeadAttentionOpp120.load(path)
    obs = {
        "player": 0,
        "step": 50,
        "day": 2,
        "hour": 2,
        "money": [3000.0, 2800.0],
        "farms": [
            {"farmer": [0, 0], "hands": [[1, 0]], "tiles": []},
            {"farmer": [5, 5], "hands": [], "tiles": []},
        ],
        "private": {"shed": {"WHEAT": 1}, "seeds": {}},
        "market": {},
    }
    out = loaded.act(obs)
    assert "farmer" in out
