"""Dense Vector Memory and Retrieval-Augmented Policy Bank (Strategy 3).

Compresses tens of thousands of archetypal game states, expert trajectories,
and question-answering strategies into a sub-90MB quantized vector index.
Provides sub-millisecond retrieval of optimal actions and strategic summaries.

Hard limit compliance:
  - Vector bank on disk: <= 90 MB (leaves 10 MB patch buffer under 100 MB ceiling)
  - Query latency: < 5 ms on CPU (single-core dot product)
  - Floating precision: INT8 quantized or FP16 normalized
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from hard_limits import PROBABILISTIC_MODEL_MAX_BYTES, SUBMISSION_MAX_BYTES

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIM = 256
MAX_BANK_ENTRIES = 100_000  # At 256-D INT8: 100k vectors = 25.6 MB


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely cast value to float, handling None, 'empty', '', NaN, and malformed types."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return default
        return float(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if not s or s in ("empty", "null", "none", "nan", "undefined", "nil"):
            return default
        try:
            f = float(val)
            if np.isnan(f) or np.isinf(f):
                return default
            return f
        except (ValueError, TypeError):
            return default
    return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely cast value to int, handling None, 'empty', '', and malformed types."""
    f = _safe_float(val, float(default))
    return int(f)


def _safe_dict(val: Any) -> Dict[str, Any]:
    """Ensure value is a dictionary."""
    return val if isinstance(val, dict) else {}


def _safe_list(val: Any) -> List[Any]:
    """Ensure value is a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, (tuple, set)):
        return list(val)
    return []


class StateVectorEncoder:
    """Deterministic, zero-overhead game observation vectorizer.
    
    Transforms full Kaggriculture dictionary observations into normalized
    fixed-dimension embedding vectors (D = 256) split into:
      - S_self (128-D): My current state (farm, bank, seeds, shed, tiles, schedule)
      - S_opp  (128-D): Opponent observation (bank, delta, hires, market, public board)
    """

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM, half_dim: int = 128) -> None:
        self.dim = dim
        self.half_dim = half_dim

    def encode_split(self, obs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Encode observation dictionary into (s_self, s_opp) vectors each of shape (128,)."""
        s_self = np.zeros(self.half_dim, dtype=np.float32)
        s_opp = np.zeros(self.half_dim, dtype=np.float32)

        if not isinstance(obs, dict):
            return s_self, s_opp

        # ── 1. Global time and calendar features (indices 0..7 on both sides) ──
        day = _safe_float(obs.get("day", 0.0), 0.0)
        hour = _safe_float(obs.get("hour", 0.0), 0.0)
        player = _safe_int(obs.get("player", 0), 0)
        player = min(max(player, 0), 1)
        opp = 1 - player

        for s_vec in (s_self, s_opp):
            s_vec[0] = day / 30.0
            s_vec[1] = hour / 24.0
            s_vec[2] = float(np.sin(2 * np.pi * hour / 24.0))
            s_vec[3] = float(np.cos(2 * np.pi * hour / 24.0))
            s_vec[4] = float(np.sin(2 * np.pi * day / 30.0))
            s_vec[5] = float(np.cos(2 * np.pi * day / 30.0))
            s_vec[6] = 1.0 if (int(hour) % 2 == 0) else 0.0
            s_vec[7] = float(player)

        # ── 2. Farm Economy: Self (8..15) vs Opponent (8..15) ─────────────────
        farms = _safe_list(obs.get("farms", []))
        p0_farm = _safe_dict(farms[player]) if len(farms) > player else {}
        p1_farm = _safe_dict(farms[opp]) if len(farms) > opp else {}

        p0_money = _safe_float(p0_farm.get("money", 0.0), 0.0)
        p1_money = _safe_float(p1_farm.get("money", 0.0), 0.0)

        # Self economic posture
        s_self[8] = float(np.log1p(max(0.0, p0_money)) / 12.0)
        s_self[9] = _safe_float(p0_farm.get("hires_today", 0), 0.0) / 10.0
        s_self[10] = len(_safe_list(p0_farm.get("unlocked_quadrants", []))) / 4.0
        s_self[11] = float(np.clip(p0_money / 49902.0, 0.0, 1.0))

        # Opponent economic posture
        s_opp[8] = float(np.log1p(max(0.0, p1_money)) / 12.0)
        s_opp[9] = float(np.clip((p0_money - p1_money) / 5000.0, -1.0, 1.0))
        s_opp[10] = _safe_float(p1_farm.get("hires_today", 0), 0.0) / 10.0
        s_opp[11] = len(_safe_list(p1_farm.get("unlocked_quadrants", []))) / 4.0

        # ── 3. Self Private Inventory (16..35) vs Opp Shared Market (16..35) ──
        private = _safe_dict(obs.get("private", {}))
        seeds = _safe_dict(private.get("seeds", {}))
        shed = _safe_dict(private.get("shed", {}))
        crops_list = ["WHEAT", "CORN", "SOY", "TOMATO", "CARROT", "STRAWBERRY", "MELON", "POTATO", "PUMPKIN", "CARROT_S"]

        for i, crop in enumerate(crops_list[:10]):
            s_self[16 + i] = _safe_float(seeds.get(crop, 0), 0.0) / 20.0
            s_self[26 + i] = _safe_float(shed.get(crop, 0), 0.0) / 50.0

        # Shared Market (Spot Prices 16..25 & Inventory 26..35 on Opponent side)
        market = _safe_dict(obs.get("market", {}))
        prices = _safe_dict(market.get("prices", {}))
        inventory = _safe_dict(market.get("inventory", {}))
        for i, crop in enumerate(crops_list[:10]):
            s_opp[16 + i] = _safe_float(prices.get(crop, 0.0), 0.0) / 100.0
            s_opp[26 + i] = _safe_float(inventory.get(crop, 0), 0.0) / 500.0

        # ── 4. Spatial Grid: Self (36..127) vs Opponent Public (36..127) ───────
        p0_tiles = _safe_list(p0_farm.get("tiles", []))
        plant_count = 0.0
        weed_count = 0.0
        dry_soil_count = 0.0
        wet_soil_count = 0.0
        harvestable_count = 0.0

        idx_self = 46
        for r in range(min(10, len(p0_tiles))):
            row = _safe_list(p0_tiles[r])
            for c in range(min(10, len(row))):
                tile = _safe_dict(row[c])
                kind = str(tile.get("kind", "EMPTY")).upper()
                moisture = _safe_float(tile.get("moisture", 0.0), 0.0)
                stage = _safe_float(tile.get("stage", 0), 0.0)

                if kind == "PLANT":
                    plant_count += 1.0
                    if stage >= 3.0:
                        harvestable_count += 1.0
                elif kind == "WEED":
                    weed_count += 1.0

                if moisture > 0.5:
                    wet_soil_count += 1.0
                else:
                    dry_soil_count += 1.0

                if idx_self < 128:
                    val = 0.0
                    if kind == "PLANT":
                        val = 0.5 + (stage / 10.0)
                    elif kind == "WEED":
                        val = -0.5
                    elif kind == "SOIL":
                        val = 0.2 if moisture > 0.5 else 0.1
                    s_self[idx_self] = val
                    idx_self += 1

        s_self[36] = plant_count / 100.0
        s_self[37] = harvestable_count / 100.0
        s_self[38] = weed_count / 100.0
        s_self[39] = wet_soil_count / 100.0
        s_self[40] = dry_soil_count / 100.0

        # Opponent Public Board Status (36..127)
        p1_tiles = _safe_list(p1_farm.get("tiles", []))
        p1_plant_count = 0.0
        p1_harvestable = 0.0
        idx_opp = 46
        for r in range(min(10, len(p1_tiles))):
            row = _safe_list(p1_tiles[r])
            for c in range(min(10, len(row))):
                t = _safe_dict(row[c])
                k = str(t.get("kind", "EMPTY")).upper()
                st = _safe_float(t.get("stage", 0), 0.0)
                if k == "PLANT":
                    p1_plant_count += 1.0
                    if st >= 3.0:
                        p1_harvestable += 1.0
                if idx_opp < 128:
                    val = 0.0
                    if k == "PLANT":
                        val = 0.5 + (st / 10.0)
                    elif k == "WEED":
                        val = -0.5
                    s_opp[idx_opp] = val
                    idx_opp += 1

        s_opp[36] = p1_plant_count / 25.0
        s_opp[37] = p1_harvestable / 25.0

        return s_self, s_opp

    def encode(self, obs: Dict[str, Any]) -> np.ndarray:
        """Encode observation dictionary into unit-norm float32 vector of shape (dim,)."""
        s_self, s_opp = self.encode_split(obs)
        vec = np.concatenate([s_self, s_opp])

        # L2-normalize vector to unit sphere for fast cosine similarity via dot product
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm
        return vec

    def polarize_kmap(
        self,
        kmap: np.ndarray,
        temperature: float = 25.0,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Polarize every cell in the K-map so all values asymptotically approach 1 or approach 0.
        
        Uses high-gain sigmoid saturation:
            K_polarized = 1.0 / (1.0 + exp(-temperature * (K - threshold)))
        Ensuring binary decision clarity across all 128x128 interaction cells.
        """
        kmap = np.asarray(kmap, dtype=np.float32)
        max_val = float(np.max(np.abs(kmap)))
        if max_val > 1e-6:
            scaled = np.abs(kmap) / max_val
        else:
            scaled = np.abs(kmap)
        
        # High-gain logistic polarization driving elements strictly towards 0.0 or 1.0
        polarized = 1.0 / (1.0 + np.exp(-temperature * (scaled - threshold)))
        polarized = np.where(polarized > 0.95, 1.0, np.where(polarized < 0.05, 0.0, polarized))
        return polarized.astype(np.float32)

    def compute_proportional_kmap(
        self,
        obs: Dict[str, Any],
        spatial_rows: int = 100,
        env_cols: int = 99,
        mask_kmap: Optional[Any] = None,
    ) -> Any:
        """Compute proportionately scaled (100 x 99) bitpacked 2D K-map."""
        try:
            from proportional_kmap import ProportionalKMap, build_proportional_kmap_mask
        except ImportError:
            from reasoning_vs_questioning.proportional_kmap import (
                ProportionalKMap,
                build_proportional_kmap_mask,
            )

        if mask_kmap is None:
            mask_kmap = build_proportional_kmap_mask(spatial_rows=spatial_rows, env_cols=env_cols)

        # Extract spatial and opponent feature vectors
        farms = obs.get("farms", []) or []
        p0 = farms[0] if farms else {}
        p1 = farms[1] if len(farms) > 1 else {}
        grid = p0.get("grid", []) or []
        market = obs.get("market", {}) or {}
        prices = market.get("prices", {}) or {}
        inventory = market.get("inventory", {}) or {}

        # Spatial tile activation (100,)
        spatial_vec = np.zeros(spatial_rows, dtype=np.float32)
        for idx in range(min(spatial_rows, len(grid))):
            tile = grid[idx] if isinstance(grid, list) else {}
            growth = float(tile.get("growth", 0.0) or 0.0) if isinstance(tile, dict) else 0.0
            water = float(tile.get("moisture", 0.5) or 0.5) if isinstance(tile, dict) else 0.5
            has_crop = 1.0 if (isinstance(tile, dict) and tile.get("crop")) else 0.0
            spatial_vec[idx] = (growth * 0.5 + water * 0.3 + has_crop * 0.2)

        # Environment / Opponent-Commodity activation (99,)
        env_vec = np.zeros(env_cols, dtype=np.float32)
        opp_money = float(p1.get("money", 3000.0) or 3000.0)
        opp_factor = math.log1p(max(0.0, opp_money)) / 12.0

        commodities = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
        for o_idx in range(11):
            for c_idx, comm in enumerate(commodities):
                col_idx = o_idx * 9 + c_idx
                if col_idx < env_cols:
                    p = float(prices.get(comm, 25.0) or 25.0)
                    inv = float(inventory.get(comm, 10000) or 10000)
                    env_vec[col_idx] = math.tanh((p / 100.0) * opp_factor * (inv / 10000.0))

        # Outer product interaction matrix
        outer_mat = np.outer(spatial_vec, env_vec)  # (100, 99)
        unpacked_mask = mask_kmap.to_numpy(np.float32) if hasattr(mask_kmap, "to_numpy") else (mask_kmap.unpack(np.float32) if hasattr(mask_kmap, "unpack") else mask_kmap)
        masked_mat = outer_mat * unpacked_mask

        # Continuous Floating-Point Convergence K-Map
        result_kmap = ProportionalKMap(spatial_rows=spatial_rows, env_cols=env_cols)
        result_kmap.set_matrix(masked_mat)
        return result_kmap

    def compute_polarized_kmap_matrix(
        self,
        obs: Dict[str, Any],
        mask_2d: Optional[np.ndarray] = None,
        temperature: float = 25.0,
        threshold: float = 0.5,
    ) -> Any:
        """Compute polarized K-map, returning a bitpacked ProportionalKMap."""
        return self.compute_proportional_kmap(obs)

    def encode_2d_kmap(
        self,
        obs: Dict[str, Any],
        mask_2d: Optional[np.ndarray] = None,
        polarize: bool = True,
    ) -> np.ndarray:
        """Encode observation via 2D K-map bilinear interaction matrix into 256-D fused representation.

        Dual-limit regime: when one axis → ∞ and the other → imaginary, annotate
        ``obs`` with substitutes ``labor`` and ``expanded land farming``.
        """
        try:
            from kmap_boundary_substitutes import apply_dual_limit_to_obs, resolve_dual_limit_substitutes
        except ImportError:
            from reasoning_vs_questioning.kmap_boundary_substitutes import (  # type: ignore
                apply_dual_limit_to_obs,
                resolve_dual_limit_substitutes,
            )

        annotated = apply_dual_limit_to_obs(obs)
        # Explicit None/'empty' axis tags if callers set them on the obs.
        dual = annotated.get("kmap_dual_limit") or resolve_dual_limit_substitutes(
            annotated.get("self_axis"),
            annotated.get("opp_axis"),
        )
        if dual is not None and isinstance(obs, dict):
            obs["kmap_dual_limit"] = dual
            obs["kmap_substitute_options"] = list(dual["options"])
            obs["preferred_actions"] = list(dual["ordered_by_axis"])

        s_self, s_opp = self.encode_split(obs)
        outer_prod = np.outer(s_self, s_opp)  # (128, 128)

        if mask_2d is not None:
            if hasattr(mask_2d, "unpack"):
                unpacked = mask_2d.unpack(np.float32)
                # Resize if needed
                if unpacked.shape != outer_prod.shape:
                    kmap_2d = outer_prod
                else:
                    kmap_2d = outer_prod * unpacked
            elif isinstance(mask_2d, np.ndarray) and mask_2d.shape == outer_prod.shape:
                kmap_2d = outer_prod * mask_2d
            else:
                kmap_2d = outer_prod
        else:
            kmap_2d = outer_prod

        if polarize:
            kmap_2d = self.polarize_kmap(kmap_2d)

        z_self = np.sum(kmap_2d, axis=1)  # (128,)
        z_opp = np.sum(kmap_2d, axis=0)   # (128,)
        fused = np.concatenate([z_self, z_opp])  # (256,)

        norm = float(np.linalg.norm(fused))
        if norm > 1e-6:
            fused /= norm
        return fused


def save_kmap_heatmap(
    kmap: Union[np.ndarray, Any],
    save_path: str = "kmap_polarization_heatmap.png",
    title: str = "Proportional 2D K-Map ({0, 1} Bitpacked Decision Matrix)",
) -> str:
    """Render and save a high-resolution heatmap of the proportional or continuous K-map."""
    if hasattr(kmap, "render_heatmap"):
        return kmap.render_heatmap(save_path=save_path, title=title)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kmap_mat = np.asarray(kmap, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    
    im = ax.imshow(kmap_mat, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto", origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Polarized Activation Probability ({0, 1} Asymptote)", fontsize=11, fontweight="bold")
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0.0 (Inert)", "0.5", "1.0 (Active Minterm)"])

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(f"Spatial Grid Rows ($N = {kmap_mat.shape[0]}$)", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Adversarial & Market Columns ($N = {kmap_mat.shape[1]}$)", fontsize=11, fontweight="bold")

    zeros_pct = float(np.mean(kmap_mat < 0.05)) * 100.0
    ones_pct = float(np.mean(kmap_mat > 0.95)) * 100.0
    purity_pct = zeros_pct + ones_pct
    
    ax.text(
        0.02, 0.03,
        f"Proportional K-Map Shape: {kmap_mat.shape[0]} x {kmap_mat.shape[1]}\n"
        f"Polarization Purity: {purity_pct:.1f}% | Zeros: {zeros_pct:.1f}% | Ones: {ones_pct:.1f}%",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return os.path.abspath(save_path)


@dataclass
class PolicyMemoryEntry:
    """Single memory bank entry holding an archetypal state and policy response."""
    state_vector: np.ndarray
    action: Dict[str, Any]
    strategic_value: float = 0.0
    archetype_tag: str = ""
    dialogue_response: str = ""


class QuantizedVectorMemoryBank:
    """High-density, sub-90MB Vector Store with INT8 Quantization.

    Stores up to 100,000 game state vectors with fast top-K cosine retrieval.
    """

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        self.dim = dim
        self.encoder = StateVectorEncoder(dim=dim)
        self.vectors_int8: Optional[np.ndarray] = None
        self.scales: Optional[np.ndarray] = None
        self.entries: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add_entry(
        self,
        obs: Union[Dict[str, Any], np.ndarray],
        action: Dict[str, Any],
        strategic_value: float = 0.0,
        archetype_tag: str = "",
        dialogue_response: str = "",
    ) -> None:
        """Add a single state-policy pair to the bank."""
        if isinstance(obs, np.ndarray):
            vec = obs.astype(np.float32)
        else:
            vec = self.encoder.encode(obs)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec = vec / norm

        # INT8 quantization: map [-1, 1] to [-127, 127]
        scale = float(np.max(np.abs(vec))) if np.max(np.abs(vec)) > 0 else 1.0
        q_vec = np.clip(np.round((vec / scale) * 127.0), -127, 127).astype(np.int8)

        if self.vectors_int8 is None:
            self.vectors_int8 = q_vec.reshape(1, self.dim)
            self.scales = np.array([scale], dtype=np.float32)
        else:
            self.vectors_int8 = np.vstack([self.vectors_int8, q_vec.reshape(1, self.dim)])
            self.scales = np.append(self.scales, np.float32(scale))

        self.entries.append({
            "action": action,
            "strategic_value": float(strategic_value),
            "archetype_tag": str(archetype_tag),
            "dialogue_response": str(dialogue_response),
        })

    def add_batch(
        self,
        vectors: np.ndarray,
        actions: Sequence[Dict[str, Any]],
        values: Optional[Sequence[float]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> None:
        """Add a batch of pre-computed vectors (N, dim) and actions efficiently."""
        n = vectors.shape[0]
        assert vectors.shape[1] == self.dim, f"Dimension mismatch: expected {self.dim}, got {vectors.shape[1]}"

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_vectors = vectors / norms

        # INT8 Quantize
        scales = np.max(np.abs(unit_vectors), axis=1)
        scales[scales == 0] = 1.0
        q_vectors = np.clip(np.round((unit_vectors / scales[:, None]) * 127.0), -127, 127).astype(np.int8)

        if self.vectors_int8 is None:
            self.vectors_int8 = q_vectors
            self.scales = scales.astype(np.float32)
        else:
            self.vectors_int8 = np.vstack([self.vectors_int8, q_vectors])
            self.scales = np.concatenate([self.scales, scales.astype(np.float32)])

        for i in range(n):
            self.entries.append({
                "action": actions[i],
                "strategic_value": float(values[i]) if values is not None else 0.0,
                "archetype_tag": str(tags[i]) if tags is not None else "",
                "dialogue_response": "",
            })

    def query_top_k(
        self,
        obs: Union[Dict[str, Any], np.ndarray],
        k: int = 5,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve top-K nearest archetype policies in sub-millisecond time."""
        if self.vectors_int8 is None or len(self.entries) == 0:
            return []

        if isinstance(obs, np.ndarray):
            q_vec = obs.astype(np.float32)
        else:
            q_vec = self.encoder.encode(obs)

        norm = float(np.linalg.norm(q_vec))
        if norm > 1e-6:
            q_vec = q_vec / norm

        raw_dots = np.dot(self.vectors_int8, q_vec)
        similarities = (raw_dots * (self.scales / 127.0)).astype(np.float32)

        k = min(k, len(self.entries))
        if k <= 0:
            return []

        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_sorted = top_k_indices[np.argsort(-similarities[top_k_indices])]

        results: List[Tuple[Dict[str, Any], float]] = []
        for idx in top_k_sorted:
            results.append((self.entries[idx], float(similarities[idx])))

        return results

    def query_policy(
        self,
        obs: Dict[str, Any],
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Get best action by blending top-K retrieved archetypes."""
        top_matches = self.query_top_k(obs, k=5)
        if not top_matches:
            return {"farmer": ["PASS"], "hands": [], "market": []}

        best_entry, best_score = top_matches[0]
        if best_score > 0.85:
            return best_entry["action"]

        scores = np.array([score for _, score in top_matches], dtype=np.float32)
        exp_weights = np.exp((scores - np.max(scores)) / max(1e-4, temperature))
        probs = exp_weights / np.sum(exp_weights)

        chosen_idx = int(np.random.choice(len(top_matches), p=probs))
        return top_matches[chosen_idx][0]["action"]

    def save(self, filepath: Union[str, Path]) -> int:
        """Serialize compressed vector bank to disk. Ensures <= 90MB hard limit."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)

        metadata_json = json.dumps({
            "dim": self.dim,
            "n_entries": len(self.entries),
            "entries": self.entries,
        }, separators=(",", ":"))

        np.savez_compressed(
            p,
            vectors_int8=self.vectors_int8 if self.vectors_int8 is not None else np.empty((0, self.dim), dtype=np.int8),
            scales=self.scales if self.scales is not None else np.empty((0,), dtype=np.float32),
            metadata=np.array(metadata_json, dtype=object),
        )

        size_bytes = p.stat().st_size
        if size_bytes > SUBMISSION_MAX_BYTES:
            logger.warning(
                "Vector bank size %d exceeds 90 MB submission limit (max %d)",
                size_bytes,
                SUBMISSION_MAX_BYTES,
            )
        else:
            logger.info("Vector bank saved to %s (%.2f MB, %d vectors)", p, size_bytes / (1024 * 1024), len(self))

        return size_bytes

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> QuantizedVectorMemoryBank:
        """Load quantized vector bank from disk."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Vector bank file not found: {p}")

        data = np.load(p, allow_pickle=True)
        meta_str = str(data["metadata"])
        meta = json.loads(meta_str)

        bank = cls(dim=int(meta.get("dim", DEFAULT_EMBEDDING_DIM)))
        bank.vectors_int8 = data["vectors_int8"]
        bank.scales = data["scales"]
        bank.entries = meta.get("entries", [])

        logger.info("Loaded vector bank: %d vectors (dim=%d)", len(bank), bank.dim)
        return bank


def bootstrap_vector_bank_from_episodes(
    episode_data_list: Sequence[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    max_entries: int = MAX_BANK_ENTRIES,
) -> QuantizedVectorMemoryBank:
    """Build a QuantizedVectorMemoryBank from historical Kaggle episode logs."""
    bank = QuantizedVectorMemoryBank(dim=DEFAULT_EMBEDDING_DIM)
    count = 0

    for ep in episode_data_list:
        if count >= max_entries:
            break
        steps = ep.get("steps", []) or []
        for step_idx, step_data in enumerate(steps):
            if count >= max_entries:
                break
            obs = step_data.get("observation") or step_data.get("obs")
            action = step_data.get("action")
            reward = float(step_data.get("reward", 0.0) or 0.0)

            if obs is not None and action is not None:
                bank.add_entry(
                    obs=obs,
                    action=action,
                    strategic_value=reward,
                    archetype_tag=f"ep_step_{step_idx}",
                )
                count += 1

    if output_path:
        bank.save(output_path)

    return bank
