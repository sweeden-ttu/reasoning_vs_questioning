"""Standalone 128x128 QKD 2D K-map mask (no RAG agent dependency)."""
from __future__ import annotations

import numpy as np


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
