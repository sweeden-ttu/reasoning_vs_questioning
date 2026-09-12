"""Spatio-Temporal-Adversarial Attention Matrix Engine.

Constructs, computes, and queries the 3D/4D Attention Tensor:
    Shape: (Map_Height, Map_Width, Number_of_Days, Number_of_Opponents)
    Default: (10, 10, 30, 11) = 33,000 Attention Weights (or 10, 10, 90, 11 = 99,000 Weights)

Mathematical Formulation:
    A(x, y, d, o) = Softmax_{(x,y)} ( (q_spatial(x, y)^T k_temporal(d)) / sqrt(d_k)
                                     + Phi_adv(o, d)
                                     + Psi_market(x, y, d) )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from market_config_suite import (
    InitialTerminalConfiguration,
    build_initial_terminal_configuration,
    build_market_functions,
    build_opponent_functions,
)
from qkd_statistical_questions import CANONICAL_QKD_QUESTIONS, QKDStatisticalQuestionBank

LEAGUE_OPPONENT_NAMES = [
    "fallow_finn",
    "wheat_walter",
    "rotation_rosa",
    "homestead_hana",
    "melon_mateo",
    "rancher_rita",
    "broker_bea",
    "slotter_silas",
    "ledger_lena",
    "closer_cleo",
    "self_play_anchor",
]

# Opponent Aggression & Sector Bias Offsets
OPPONENT_PROFILES = {
    0: {"name": "fallow_finn", "aggression": 0.0, "preferred_crop": "NONE", "threat": 0.0},
    1: {"name": "wheat_walter", "aggression": 0.4, "preferred_crop": "WHEAT", "threat": 0.2},
    2: {"name": "rotation_rosa", "aggression": 0.5, "preferred_crop": "CARROT", "threat": 0.3},
    3: {"name": "homestead_hana", "aggression": 0.6, "preferred_crop": "TOMATO", "threat": 0.4},
    4: {"name": "melon_mateo", "aggression": 0.7, "preferred_crop": "MELON", "threat": 0.5},
    5: {"name": "rancher_rita", "aggression": 0.75, "preferred_crop": "WOOL", "threat": 0.6},
    6: {"name": "broker_bea", "aggression": 0.85, "preferred_crop": "STRAWBERRY", "threat": 0.7},
    7: {"name": "slotter_silas", "aggression": 0.90, "preferred_crop": "MILK", "threat": 0.75},
    8: {"name": "ledger_lena", "aggression": 0.92, "preferred_crop": "EGG", "threat": 0.8},
    9: {"name": "closer_cleo", "aggression": 0.98, "preferred_crop": "MELON", "threat": 0.9},
    10: {"name": "self_play_anchor", "aggression": 1.0, "preferred_crop": "COMPOSITE", "threat": 1.0},
}


class SpatioTemporalAttentionMatrix:
    """3D/4D Spatio-Temporal-Adversarial Attention Tensor Engine.
    
    Tensor Dimensions:
      - Spatial Grid: (height, width) = (10, 10) = 100 tiles.
      - Temporal Horizon: days = 30 (or 90 for macro curriculum).
      - Adversarial Opponents: num_opponents = 11.
    """

    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        days: int = 30,
        num_opponents: int = 11,
        feature_dim: int = 16,
    ) -> None:
        self.height = height
        self.width = width
        self.map_size = height * width
        self.days = days
        self.num_opponents = num_opponents
        self.feature_dim = feature_dim
        self.question_bank = QKDStatisticalQuestionBank()

        # Cache for latest computed attention tensor
        self._tensor: Optional[np.ndarray] = None
        self._raw_logits: Optional[np.ndarray] = None

    @property
    def tensor_shape(self) -> Tuple[int, int, int, int]:
        """Return the exact 4D tensor shape (Height, Width, Days, Opponents)."""
        return (self.height, self.width, self.days, self.num_opponents)

    @property
    def flat_shape(self) -> Tuple[int, int, int]:
        """Return the flattened 3D tensor shape (Map_Tiles, Days, Opponents)."""
        return (self.map_size, self.days, self.num_opponents)

    def encode_spatial_query(
        self,
        obs: Dict[str, Any],
        x: int,
        y: int,
    ) -> np.ndarray:
        """Encode spatial tile properties into a feature vector in R^{feature_dim}."""
        q = np.zeros(self.feature_dim, dtype=np.float32)
        player = int(obs.get("player", 0) or 0)
        farms = obs.get("farms", []) or []
        farm = farms[player] if len(farms) > player else {}
        tiles = farm.get("tiles", []) or []

        tile = None
        if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
            tile = tiles[y][x]

        # Feature 0..3: Spatial position coordinates
        q[0] = x / max(1, self.width - 1)
        q[1] = y / max(1, self.height - 1)
        q[2] = 1.0 if (x < 5 and y < 5) else 0.0  # NW Quadrant (Initial)
        q[3] = 1.0 if (x >= 5 or y >= 5) else 0.0  # Unlocked Expansion Quadrants

        # Feature 4..8: Tile Kind & Agricultural Maturity
        if isinstance(tile, dict):
            kind = tile.get("kind", "SOIL")
            if kind == "PLANT":
                q[4] = 1.0
                q[5] = float(tile.get("stage", 0) or 0) / 3.0  # Maturity Stage (0..3)
                q[6] = float(tile.get("moisture", 0.0) or 0.0)  # Soil Moisture (0..1)
            elif kind == "WEED":
                q[7] = 1.0  # Weed disruption alert
            elif kind == "PASTURE":
                q[8] = 1.0  # Pasture tile
            else:
                q[9] = 1.0  # Tilled Soil
        else:
            q[9] = 0.5

        # Feature 10..12: Farmer & Hands Worker Proximity
        farmer_pos = farm.get("farmer", [0, 0]) or [0, 0]
        dist_farmer = abs(x - farmer_pos[0]) + abs(y - farmer_pos[1])
        q[10] = math.exp(-0.25 * dist_farmer)

        hands = farm.get("hands", []) or []
        if hands:
            min_hand_dist = min(abs(x - h[0]) + abs(y - h[1]) for h in hands if len(h) >= 2)
            q[11] = math.exp(-0.25 * min_hand_dist)
            q[12] = len(hands) / 10.0

        # Normalization
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q = q / norm
        return q

    def encode_temporal_key(
        self,
        day: int,
        total_days: int = 30,
    ) -> np.ndarray:
        """Encode competition timeline key into a feature vector in R^{feature_dim}."""
        k = np.zeros(self.feature_dim, dtype=np.float32)
        d_norm = float(day) / max(1.0, float(total_days - 1))
        q_days_rem = max(0.0, (float(total_days) - float(day)) / float(total_days))

        k[0] = d_norm
        k[1] = q_days_rem  # Q_DAYS_REMAINING channel
        k[2] = 1.0 if q_days_rem <= (2.0 / 30.0) else 0.0  # Terminal Liquidation Indicator
        k[3] = 1.0 if day < 7 else 0.0  # Early Growth Compounding Phase
        k[4] = math.sin(2.0 * math.pi * (day % 7) / 7.0)  # Weekly Agricultural Cycle
        k[5] = math.cos(2.0 * math.pi * (day % 7) / 7.0)
        k[6] = 1.0 if (day % 2 == 0) else 0.0  # Even-Hour & Even-Day Execution Rhythm

        # Normalization
        norm = np.linalg.norm(k)
        if norm > 1e-6:
            k = k / norm
        return k

    def compute_adversarial_context(
        self,
        opponent_id: int,
        day: int,
        opponent_functions: Optional[Dict[str, Callable]] = None,
    ) -> float:
        """Compute scalar adversarial bias Phi_adv(o, d) for opponent o on day d."""
        profile = OPPONENT_PROFILES.get(opponent_id, OPPONENT_PROFILES[0])
        base_threat = profile["threat"]
        aggression = profile["aggression"]

        # Opponent dynamic gap factor
        gap_bias = 0.0
        if opponent_functions and "compute_adversarial_lead_gap" in opponent_functions:
            try:
                lead_gap = opponent_functions["compute_adversarial_lead_gap"]()
                if lead_gap < 0.0:
                    gap_bias = min(1.0, abs(lead_gap) / 20000.0) * 0.5
            except Exception:
                pass

        # Opponent timeline scaling
        timeline_threat = base_threat * (1.0 + 0.3 * (day / max(1.0, float(self.days))))
        return float(timeline_threat + gap_bias + (0.2 * aggression))

    def compute_market_response(
        self,
        obs: Dict[str, Any],
        x: int,
        y: int,
        day: int,
        market_functions: Optional[Dict[str, Callable]] = None,
    ) -> float:
        """Compute scalar market response bias Psi_market(x, y, d) from market price curves."""
        player = int(obs.get("player", 0) or 0)
        farms = obs.get("farms", []) or []
        farm = farms[player] if len(farms) > player else {}
        tiles = farm.get("tiles", []) or []

        crop = "WHEAT"
        if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
            t = tiles[y][x]
            if isinstance(t, dict):
                crop = t.get("crop", "WHEAT") or "WHEAT"

        if market_functions and "predict_future_price" in market_functions:
            try:
                p_future = market_functions["predict_future_price"](crop, horizon_hours=min(24, (day % 5) * 6))
                # Normalize response price (baseline $25..$250 -> 0.1..1.0)
                return float(math.log1p(max(0.0, p_future)) / 6.0)
            except Exception:
                pass

        return 0.25

    def compute_attention_tensor(
        self,
        obs: Dict[str, Any],
        initial_terminal_configuration: Optional[Union[InitialTerminalConfiguration, Dict[str, Any]]] = None,
        market_functions: Optional[Dict[str, Callable]] = None,
        opponent_functions: Optional[Dict[str, Callable]] = None,
    ) -> np.ndarray:
        """Compute and normalize the complete 4D Attention Tensor.
        
        Output Shape: (10, 10, days, num_opponents)
        Normalization: Spatial Softmax across (10, 10) grid for each (d, o) slice.
        """
        if initial_terminal_configuration is None:
            initial_terminal_configuration = build_initial_terminal_configuration(obs)
        if market_functions is None:
            market_functions = build_market_functions(obs, initial_terminal_configuration)
        if opponent_functions is None:
            opponent_functions = build_opponent_functions(obs, initial_terminal_configuration)

        logits = np.zeros(self.tensor_shape, dtype=np.float32)
        d_k = float(self.feature_dim)
        scale = 1.0 / math.sqrt(d_k)

        # Precompute spatial queries: (10, 10, feature_dim)
        spatial_queries = np.zeros((self.height, self.width, self.feature_dim), dtype=np.float32)
        for y in range(self.height):
            for x in range(self.width):
                spatial_queries[y, x] = self.encode_spatial_query(obs, x, y)

        # Precompute temporal keys: (days, feature_dim)
        temporal_keys = np.zeros((self.days, self.feature_dim), dtype=np.float32)
        for d in range(self.days):
            temporal_keys[d] = self.encode_temporal_key(d, total_days=self.days)

        # Compute dot-product attention + adversarial & market modulations
        for d in range(self.days):
            k_d = temporal_keys[d]
            for o in range(self.num_opponents):
                phi_adv = self.compute_adversarial_context(o, d, opponent_functions)
                for y in range(self.height):
                    for x in range(self.width):
                        q_xy = spatial_queries[y, x]
                        dot_prod = float(np.dot(q_xy, k_d)) * scale
                        psi_mkt = self.compute_market_response(obs, x, y, d, market_functions)
                        logits[y, x, d, o] = dot_prod + phi_adv + psi_mkt

        self._raw_logits = logits

        # Spatial Softmax Normalization across (height, width) for every (d, o)
        att_tensor = np.zeros_like(logits)
        for d in range(self.days):
            for o in range(self.num_opponents):
                slice_logits = logits[:, :, d, o]
                max_val = np.max(slice_logits)
                exp_vals = np.exp(slice_logits - max_val)
                att_tensor[:, :, d, o] = exp_vals / np.sum(exp_vals)

        self._tensor = att_tensor
        return att_tensor

    def get_tile_attention_map(self, day: int = 0, opponent_id: int = 0) -> np.ndarray:
        """Retrieve the 10x10 spatial attention heatmap for a specific day and opponent."""
        if self._tensor is None:
            raise RuntimeError("Attention tensor not yet computed. Call compute_attention_tensor() first.")
        d = min(max(0, day), self.days - 1)
        o = min(max(0, opponent_id), self.num_opponents - 1)
        return self._tensor[:, :, d, o]

    def get_temporal_attention_curve(self, x: int, y: int, opponent_id: int = 0) -> np.ndarray:
        """Retrieve the timeline curve (length = days) for a specific tile and opponent."""
        if self._tensor is None:
            raise RuntimeError("Attention tensor not yet computed.")
        o = min(max(0, opponent_id), self.num_opponents - 1)
        return self._tensor[y, x, :, o]

    def get_opponent_attention_slice(self, opponent_id: int = 0) -> np.ndarray:
        """Retrieve the full (10, 10, days) tensor slice for an opponent."""
        if self._tensor is None:
            raise RuntimeError("Attention tensor not yet computed.")
        o = min(max(0, opponent_id), self.num_opponents - 1)
        return self._tensor[:, :, :, o]

    def get_highest_priority_tiles(
        self,
        day: int = 0,
        opponent_id: int = 0,
        top_k: int = 5,
    ) -> List[Tuple[int, int, float]]:
        """Return the top-K priority tile coordinates (x, y, weight) for a given day and opponent."""
        heatmap = self.get_tile_attention_map(day, opponent_id)
        flat_indices = np.argsort(heatmap.ravel())[::-1][:top_k]
        result = []
        for idx in flat_indices:
            y, x = divmod(int(idx), self.width)
            result.append((x, y, float(heatmap[y, x])))
        return result

    def summary(self) -> Dict[str, Any]:
        """Return diagnostic metrics and summary of the attention tensor."""
        is_computed = self._tensor is not None
        return {
            "tensor_shape": list(self.tensor_shape),
            "total_attention_weights": int(np.prod(self.tensor_shape)),
            "spatial_grid": f"{self.width}x{self.height} ({self.map_size} tiles)",
            "competition_days": self.days,
            "league_opponents": self.num_opponents,
            "is_computed": is_computed,
            "mean_weight": float(np.mean(self._tensor)) if is_computed else 0.0,
            "max_weight": float(np.max(self._tensor)) if is_computed else 0.0,
            "min_weight": float(np.min(self._tensor)) if is_computed else 0.0,
            "opponent_names": LEAGUE_OPPONENT_NAMES[:self.num_opponents],
        }
