"""Unit tests for 720×720 Floating-Point Convergence K-Map Engine."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
RVQ_DIR = Path(__file__).resolve().parent
if str(RVQ_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from kmap_720x720 import (
    KMap720x720,
    TOTAL_INTERACTION_CHANNELS,
    TOTAL_SEASON_TURNS,
    build_and_train_720x720_kmap,
    load_trained_10x10_kmap,
)

SAMPLE_10X10_PATH = (
    Path("/Users/sweeden/.cursor/worktrees/kmap-approach1-4ff64ce5/kagg-7c2ea97d9d34/artifacts/kmap_10x10/kmap_10x10_trained.json")
    if Path("/Users/sweeden/.cursor/worktrees/kmap-approach1-4ff64ce5/kagg-7c2ea97d9d34/artifacts/kmap_10x10/kmap_10x10_trained.json").exists()
    else ROOT_DIR / "artifacts/kmap_10x10/kmap_10x10_trained.json"
)


def test_kmap_720x720_initialization():
    engine = KMap720x720()
    assert engine.rows == TOTAL_SEASON_TURNS
    assert engine.cols == TOTAL_INTERACTION_CHANNELS
    assert engine.matrix.shape == (720, 720)
    assert len(engine.row_labels) == 720


def test_load_trained_10x10_kmap():
    if not SAMPLE_10X10_PATH.exists():
        pytest.skip(f"{SAMPLE_10X10_PATH} not found")
    data = load_trained_10x10_kmap(SAMPLE_10X10_PATH)
    assert "matrix" in data
    assert len(data["matrix"]) == 10
    assert len(data["matrix"][0]) == 10
    assert "summary" in data


def test_kmap_720x720_training_and_convergence(tmp_path):
    if not SAMPLE_10X10_PATH.exists():
        pytest.skip(f"{SAMPLE_10X10_PATH} not found")

    data = load_trained_10x10_kmap(SAMPLE_10X10_PATH)
    engine = KMap720x720()
    matrix = engine.train_from_10x10_data(data, temperature=25.0)

    assert matrix.shape == (720, 720)
    assert matrix.dtype == np.float32
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)
    assert not np.any(np.isnan(matrix))
    assert not np.any(np.isinf(matrix))

    metrics = engine.compute_convergence_metrics()
    assert metrics["total_cells"] == 518400
    assert metrics["polarization_ratio"] > 0.40
    assert metrics["shannon_entropy_bits"] >= 0.0
    assert metrics["epistemic_variance"] >= 0.0
    assert metrics["active_minterms_count"] > 0


def test_kmap_720x720_heatmap_rendering(tmp_path):
    if not SAMPLE_10X10_PATH.exists():
        pytest.skip(f"{SAMPLE_10X10_PATH} not found")

    data = load_trained_10x10_kmap(SAMPLE_10X10_PATH)
    engine = KMap720x720()
    engine.train_from_10x10_data(data)

    log_path = tmp_path / "test_log_heatmap.png"
    pol_path = tmp_path / "test_pol_heatmap.png"

    log_res = engine.render_logarithmic_heatmap(save_path=log_path)
    pol_res = engine.render_polarization_heatmap(save_path=pol_path)

    assert Path(log_res).exists()
    assert Path(log_res).stat().st_size > 49902
    assert Path(pol_res).exists()
    assert Path(pol_res).stat().st_size > 49902


def test_kmap_720x720_serialization_and_reload(tmp_path):
    if not SAMPLE_10X10_PATH.exists():
        pytest.skip(f"{SAMPLE_10X10_PATH} not found")

    results = build_and_train_720x720_kmap(
        trained_10x10_path=SAMPLE_10X10_PATH,
        out_dir=tmp_path,
        temperature=25.0,
    )

    assert Path(results["json_path"]).exists()
    assert Path(results["matrix_path"]).exists()

    # Verify reloading matrix
    loaded = np.load(results["matrix_path"])
    mat = loaded["matrix"]
    assert mat.shape == (720, 720)
    assert mat.dtype == np.float32
