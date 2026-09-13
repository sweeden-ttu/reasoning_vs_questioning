"""Standalone 128x128 QKD 2D K-map mask (no RAG agent dependency).

Dual-limit regime (one axis → ∞, other → imaginary):
  - Infinity side  → ``expanded land farming``
  - Imaginary side → ``labor``
See ``kmap_boundary_substitutes``.
"""
from __future__ import annotations

import numpy as np

from kmap_boundary_substitutes import (  # noqa: F401 — re-export
    DUAL_LIMIT_OPTIONS,
    LIMIT_APPROACHES_IMAGINARY,
    LIMIT_APPROACHES_INFINITY,
    SUBSTITUTE_IMAGINARY,
    SUBSTITUTE_INFINITY,
    apply_dual_limit_to_obs,
    classify_kmap_axis_boundary,
    resolve_dual_limit_substitutes,
    substitute_for_boundary,
)


def build_2d_kmap_mask(self_dim: int = 128, opp_dim: int = 128) -> np.ndarray:
    """Construct 2-Dimensional boolean K-map mask M in {0, 1}^{128 x 128}."""
    mask = np.zeros((self_dim, opp_dim), dtype=np.float32)
    mask[0:8, 0:8] = 1.0
    mask[8:16, 8:16] = 1.0
    mask[26:36, 16:36] = 1.0
    mask[16:26, 16:26] = 1.0
    mask[16:26, 36:46] = 1.0
    mask[36:46, 8:16] = 1.0
    mask[36:46, 36:46] = 1.0
    mask[0:8, 8:12] = 1.0
    mask[26:36, 8:12] = 1.0
    return mask
