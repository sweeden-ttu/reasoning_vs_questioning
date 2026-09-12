"""Unit Tests for Continuous Floating-Point Convergence 2D Karnaugh Map (K-Map).

Verifies:
  1. Exact proportional scaling (100 farm tiles x 99 opponent-commodity channels).
  2. Continuous float32 storage (39,600 bytes for 9,900 continuous probability cells).
  3. Real-time asymptotic convergence metrics (Polarization ratio, Shannon entropy, Variance).
  4. Continuous updates (EMA, logistic temperature annealing, Bayesian log-odds).
  5. Soft continuous fuzzy logic operations (&, |, ~).
  6. Float32 binary serialization (to_bytes / from_bytes).
  7. High-resolution continuous logarithmic heatmap rendering.
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

from proportional_kmap import (
    COMMODITY_LABELS,
    OPPONENT_LABELS,
    FloatingPointConvergenceKMap,
    ProportionalKMap,
    build_proportional_kmap_mask,
    generate_gray_code,
)
from vector_memory_bank import StateVectorEncoder, save_kmap_heatmap


def test_proportional_kmap_dimensions_and_float32_storage():
    kmap = ProportionalKMap(spatial_rows=100, env_cols=99)
    assert kmap.shape == (100, 99)
    assert kmap.total_cells == 9900
    # 9,900 float32 elements = 9900 * 4 = 39,600 bytes
    assert kmap.byte_size == 39600
    assert kmap.data.dtype == np.float32
    assert len(kmap.row_labels) == 100
    assert len(kmap.col_labels) == 99


def test_continuous_convergence_metrics():
    kmap = ProportionalKMap(spatial_rows=100, env_cols=99)
    
    # 1. Uniform uncertain state (P = 0.5)
    kmap.set_matrix(np.full((100, 99), 0.5, dtype=np.float32))
    assert pytest.approx(kmap.polarization_ratio, abs=1e-4) == 0.0
    assert pytest.approx(kmap.entropy, abs=1e-4) == 1.0  # Max Shannon entropy
    assert pytest.approx(kmap.epistemic_variance, abs=1e-4) == 0.25

    # 2. Fully polarized state (P in {0.0, 1.0})
    pattern = np.zeros((100, 99), dtype=np.float32)
    pattern[10:30, 20:40] = 1.0
    kmap.set_matrix(pattern)
    assert pytest.approx(kmap.polarization_ratio, abs=1e-4) == 1.0
    assert pytest.approx(kmap.entropy, abs=1e-3) == 0.0  # Zero entropy
    assert pytest.approx(kmap.epistemic_variance, abs=1e-4) == 0.0
    assert kmap.active_minterms_count == 20 * 20


def test_ema_and_temperature_annealing():
    kmap = ProportionalKMap(spatial_rows=100, env_cols=99)
    kmap.set_matrix(np.full((100, 99), 0.5, dtype=np.float32))

    # Target high-confidence belief
    target = np.zeros((100, 99), dtype=np.float32)
    target[0:50, :] = 1.0

    # 1. EMA Step
    kmap.update_ema(target, alpha=0.2)
    assert kmap.step_count == 1
    assert len(kmap.convergence_history) == 1
    assert kmap.polarization_ratio > 0.0

    # 2. Temperature Annealing
    init_pol = kmap.polarization_ratio
    kmap.anneal_temperature(temperature=0.1)  # Low temp = sharpen
    assert kmap.polarization_ratio > init_pol


def test_soft_continuous_logic_operations():
    kmap_a = ProportionalKMap(spatial_rows=100, env_cols=99)
    kmap_b = ProportionalKMap(spatial_rows=100, env_cols=99)

    mat_a = np.full((100, 99), 0.8, dtype=np.float32)
    mat_b = np.full((100, 99), 0.3, dtype=np.float32)

    kmap_a.set_matrix(mat_a)
    kmap_b.set_matrix(mat_b)

    # Soft AND (min)
    kmap_and = kmap_a & kmap_b
    assert np.allclose(kmap_and.to_numpy(), 0.3)

    # Soft OR (max)
    kmap_or = kmap_a | kmap_b
    assert np.allclose(kmap_or.to_numpy(), 0.8)

    # Soft NOT (1 - A)
    kmap_not = ~kmap_a
    assert np.allclose(kmap_not.to_numpy(), 0.2)


def test_float32_binary_serialization_roundtrip(tmp_path: Path):
    kmap = build_proportional_kmap_mask(100, 99)
    data = kmap.to_bytes()

    # Header is 12 bytes + 39,600 float32 data = 39,612 bytes total
    assert len(data) == 12 + 39600

    loaded_kmap = ProportionalKMap.from_bytes(data)
    assert loaded_kmap.shape == (100, 99)
    assert loaded_kmap.byte_size == 39600
    assert np.allclose(loaded_kmap.data, kmap.data)


def test_proportional_continuous_heatmap_rendering(tmp_path: Path):
    kmap = build_proportional_kmap_mask(100, 99)
    heatmap_file = str(tmp_path / "test_kmap_log_heatmap.png")

    rendered_path = kmap.render_logarithmic_heatmap(save_path=heatmap_file, title="Test Continuous K-Map")
    assert os.path.exists(rendered_path)
    assert os.path.getsize(rendered_path) > 1000  # Non-empty image


def test_state_vector_encoder_proportional_kmap_integration():
    encoder = StateVectorEncoder(dim=256, half_dim=128)
    dummy_obs = {
        "player": 0,
        "day": 45,
        "hour": 6,
        "farms": [
            {
                "money": 45000.0,
                "grid": [{"type": "SOIL", "crop": "MELON", "growth": 0.8, "moisture": 0.7} for _ in range(100)],
            },
            {"money": 20000.0},
        ],
        "market": {"inventory": {"MELON": 11000}, "prices": {"MELON": 240.0}},
    }

    kmap = encoder.compute_proportional_kmap(dummy_obs, spatial_rows=100, env_cols=99)
    assert isinstance(kmap, ProportionalKMap)
    assert kmap.shape == (100, 99)
    assert kmap.byte_size == 39600
    assert kmap.data.dtype == np.float32
    assert 0.0 <= kmap.polarization_ratio <= 1.0
