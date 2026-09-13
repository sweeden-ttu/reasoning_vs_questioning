"""Floating-Point Convergence 2D Karnaugh Map (K-Map) Engine.

Stores and tracks continuous floating-point probability convergence across the game world:
1. Proportionate Spatial-Adversarial Dimensionality:
   - Spatial Grid Axis (Rows): H x W = 10 x 10 = 100 farm tiles (or 1,000 voxels).
   - Adversarial & Market Axis (Cols): Num_Opponents x Num_Commodities = 11 x 9 = 99 channels.
   - Exact proportional grid shape: (100, 99) = 9,900 continuous floating-point decision cells.
2. Continuous Float32 Convergence Storage:
   - Maintains full-precision continuous float32 probabilities P(x) in [0.0, 1.0] (39,600 bytes).
   - Preserves fractional decision confidence, epistemic variance, and smooth learning gradients.
3. Real-Time Asymptotic Convergence Tracking:
   - Polarization Ratio: (1/N) sum |2P - 1| in [0.0, 1.0] (asymptote to 0.0 or 1.0).
   - Shannon Information Entropy: H(P) = -sum [P log2(P) + (1-P) log2(1-P)] bits.
   - Epistemic Variance: (1/N) sum P(1-P).
4. Continuous Update Dynamics:
   - Exponential Moving Average (EMA) updates.
   - Continuous logistic temperature annealing (tau -> tau_min).
   - Continuous Bayesian log-odds accumulation.
5. 2-Dimensional Logarithmic Heatmap Output:
   - Uses LogNorm(vmin=1e-4, vmax=1.0) over the continuous probability field.
"""

from __future__ import annotations

import io
import math
import os
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Gray code generation helpers
def generate_gray_code(n_bits: int) -> List[int]:
    """Generate n-bit standard reflected Gray code sequence."""
    if n_bits <= 0:
        return [0]
    return [i ^ (i >> 1) for i in range(1 << n_bits)]


OPPONENT_LABELS = [
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

COMMODITY_LABELS = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
]


@dataclass
class ProportionalKMap:
    """Continuous Floating-Point Convergence 2D Karnaugh Map."""

    spatial_rows: int = 100       # 10 x 10 Spatial farm grid
    env_cols: int = 99            # 11 Opponents x 9 Commodities
    data: np.ndarray = field(default_factory=lambda: np.zeros((100, 99), dtype=np.float32))
    logits: Optional[np.ndarray] = None
    row_labels: List[str] = field(default_factory=list)
    col_labels: List[str] = field(default_factory=list)
    step_count: int = 0
    convergence_history: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.data is None or self.data.shape != (self.spatial_rows, self.env_cols):
            self.data = np.zeros((self.spatial_rows, self.env_cols), dtype=np.float32)
        else:
            self.data = np.asarray(self.data, dtype=np.float32)

        if not self.row_labels:
            self.row_labels = [
                f"Tile({y},{x})" for y in range(10) for x in range(10)
            ][: self.spatial_rows]

        if not self.col_labels:
            self.col_labels = [
                f"{opp[:6]}:{crop[:4]}"
                for opp in OPPONENT_LABELS
                for crop in COMMODITY_LABELS
            ][: self.env_cols]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.spatial_rows, self.env_cols)

    @property
    def total_cells(self) -> int:
        return self.spatial_rows * self.env_cols

    @property
    def byte_size(self) -> int:
        """Memory size of the continuous float32 matrix on disk/RAM."""
        return int(self.data.nbytes)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    # ── Convergence Metrics ───────────────────────────────────────────────────

    @property
    def polarization_ratio(self) -> float:
        """Measure of asymptotic discrete convergence towards {0, 1}.
        
        Returns a value in [0.0, 1.0]:
          0.0 = completely uncertain (all cells at 0.5)
          1.0 = fully converged / polarized (all cells at 0.0 or 1.0)
        """
        p = np.clip(self.data, 0.0, 1.0)
        return float(np.mean(np.abs(2.0 * p - 1.0)))

    @property
    def entropy(self) -> float:
        """Mean binary Shannon entropy H(P) in bits.
        
        Returns a value in [0.0, 1.0]:
          1.0 = maximum epistemic uncertainty
          0.0 = completely resolved deterministic beliefs
        """
        eps = 1e-6
        p = np.clip(self.data.astype(np.float64), eps, 1.0 - eps)
        h = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        return float(np.mean(h))

    @property
    def epistemic_variance(self) -> float:
        """Mean Bernoulli epistemic variance P * (1 - P) across all cells."""
        p = np.clip(self.data, 0.0, 1.0)
        return float(np.mean(p * (1.0 - p)))

    @property
    def active_minterms_count(self) -> int:
        """Number of decision cells with high activation confidence (P >= 0.5)."""
        return int(np.sum(self.data >= 0.5))

    @property
    def minterms_count(self) -> int:
        """Alias for active_minterms_count for backward compatibility."""
        return self.active_minterms_count

    # ── Matrix Setting & Continuous Updates ───────────────────────────────────

    def set_matrix(self, matrix: np.ndarray) -> ProportionalKMap:
        """Set continuous floating-point probability matrix."""
        mat = np.asarray(matrix, dtype=np.float32)
        if mat.shape != (self.spatial_rows, self.env_cols):
            resized = np.zeros((self.spatial_rows, self.env_cols), dtype=np.float32)
            min_r = min(self.spatial_rows, mat.shape[0])
            min_c = min(self.env_cols, mat.shape[1])
            resized[:min_r, :min_c] = mat[:min_r, :min_c]
            mat = resized
        self.data = np.clip(mat, 0.0, 1.0)
        return self

    def pack_matrix(self, matrix: np.ndarray, threshold: Optional[float] = None) -> ProportionalKMap:
        """Ingest matrix into continuous float32 convergence storage."""
        self.set_matrix(matrix)
        if threshold is not None and threshold > 0.0:
            # Soft continuous polarization gain
            gain = 10.0
            self.data = 1.0 / (1.0 + np.exp(-gain * (self.data - threshold)))
            self.data = np.clip(self.data, 0.0, 1.0)
        return self

    def update_ema(self, target_matrix: np.ndarray, alpha: float = 0.05) -> ProportionalKMap:
        """Exponential Moving Average (EMA) continuous probability update."""
        tgt = np.asarray(target_matrix, dtype=np.float32)
        if tgt.shape != (self.spatial_rows, self.env_cols):
            resized = np.zeros((self.spatial_rows, self.env_cols), dtype=np.float32)
            min_r = min(self.spatial_rows, tgt.shape[0])
            min_c = min(self.env_cols, tgt.shape[1])
            resized[:min_r, :min_c] = mat[:min_r, :min_c] if 'mat' in locals() else tgt[:min_r, :min_c]
            tgt = resized
        tgt = np.clip(tgt, 0.0, 1.0)

        self.data = (1.0 - alpha) * self.data + alpha * tgt
        self.step_count += 1
        self.convergence_history.append({
            "step": self.step_count,
            "polarization": self.polarization_ratio,
            "entropy": self.entropy,
            "variance": self.epistemic_variance,
        })
        return self

    def anneal_temperature(self, temperature: float, gain: float = 1.0) -> ProportionalKMap:
        """Continuous logistic temperature annealing: P = sigma(z * gain / tau)."""
        eps = 1e-6
        p = np.clip(self.data, eps, 1.0 - eps)
        if self.logits is None:
            self.logits = np.log(p / (1.0 - p))

        t = max(1e-4, float(temperature))
        self.data = 1.0 / (1.0 + np.exp(-gain * self.logits / t))
        self.data = np.clip(self.data, 0.0, 1.0)
        return self

    def update_bayesian_log_odds(self, delta_log_odds: np.ndarray, lr: float = 0.1) -> ProportionalKMap:
        """Bayesian continuous log-odds belief accumulation."""
        eps = 1e-6
        p = np.clip(self.data, eps, 1.0 - eps)
        if self.logits is None:
            self.logits = np.log(p / (1.0 - p))

        delta = np.asarray(delta_log_odds, dtype=np.float32)
        if delta.shape == self.shape:
            self.logits += lr * delta
            self.data = 1.0 / (1.0 + np.exp(-self.logits))
            self.data = np.clip(self.data, 0.0, 1.0)
        return self

    def to_numpy(self, dtype: Any = np.float32) -> np.ndarray:
        """Return a copy of the continuous floating-point probability matrix."""
        return self.data.astype(dtype, copy=True)

    def unpack(self, dtype: Any = np.float32) -> np.ndarray:
        """Return copy of continuous matrix for full backward compatibility."""
        return self.to_numpy(dtype=dtype)

    def to_binary(self, threshold: float = 0.5) -> np.ndarray:
        """Return discrete boolean mask {0, 1} without modifying underlying float32 matrix."""
        return (self.data >= threshold).astype(np.float32)

    # ── Binary Serialization ──────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Serialize continuous float32 K-map into binary header + float32 buffer."""
        header = struct.pack(">III", self.spatial_rows, self.env_cols, self.step_count)
        return header + self.data.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> ProportionalKMap:
        """Deserialize binary buffer into continuous float32 ProportionalKMap."""
        spatial_rows, env_cols, step_count = struct.unpack(">III", data[:12])
        mat = np.frombuffer(data[12:], dtype=np.float32).reshape((spatial_rows, env_cols))
        kmap = cls(spatial_rows=spatial_rows, env_cols=env_cols, data=mat.copy())
        kmap.step_count = step_count
        return kmap

    # ── Soft Continuous Fuzzy Logic Operations ────────────────────────────────

    def __and__(self, other: ProportionalKMap) -> ProportionalKMap:
        """Soft continuous AND: min(A, B)."""
        assert self.shape == other.shape
        res = ProportionalKMap(self.spatial_rows, self.env_cols)
        res.data = np.minimum(self.data, other.data)
        return res

    def __or__(self, other: ProportionalKMap) -> ProportionalKMap:
        """Soft continuous OR: max(A, B)."""
        assert self.shape == other.shape
        res = ProportionalKMap(self.spatial_rows, self.env_cols)
        res.data = np.maximum(self.data, other.data)
        return res

    def __invert__(self) -> ProportionalKMap:
        """Soft continuous NOT: 1.0 - A."""
        res = ProportionalKMap(self.spatial_rows, self.env_cols)
        res.data = 1.0 - self.data
        return res

    # ── Heatmap Visualizations ────────────────────────────────────────────────

    def render_logarithmic_heatmap(
        self,
        save_path: str = "kmap_logarithmic_heatmap.png",
        title: str = r"Proportional 2D K-Map (Logarithmic Scale: $10^{-4} \rightarrow 10^0$)",
        continuous_density: Optional[np.ndarray] = None,
    ) -> str:
        """Render and save a 2-Dimensional Logarithmic Heatmap of continuous float32 convergence."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        if continuous_density is not None:
            mat = np.asarray(continuous_density, dtype=np.float32)
            if mat.shape != (self.spatial_rows, self.env_cols):
                resized = np.zeros((self.spatial_rows, self.env_cols), dtype=np.float32)
                min_r = min(self.spatial_rows, mat.shape[0])
                min_c = min(self.env_cols, mat.shape[1])
                resized[:min_r, :min_c] = mat[:min_r, :min_c]
                mat = resized
        else:
            # Genuine continuous float32 probability matrix
            mat = np.clip(self.data, 1e-4, 1.0)
            # Fill inert background floor smoothly
            for r in range(self.spatial_rows):
                for c in range(self.env_cols):
                    if self.data[r, c] < 1e-3:
                        decay = 1e-4 + 2e-3 * math.exp(-0.04 * ((r % 25) + (c % 9)))
                        mat[r, c] = max(mat[r, c], decay)

        fig, ax = plt.subplots(figsize=(14, 10), dpi=160)

        # Logarithmic normalization across 4 decades of intensity
        norm = mcolors.LogNorm(vmin=1e-4, vmax=1.0)
        im = ax.imshow(mat, cmap="inferno", norm=norm, aspect="auto", origin="upper")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(r"Logarithmic Decision Intensity $\log_{10}(P)$", fontsize=11, fontweight="bold")
        cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
        cbar.set_ticklabels(["$10^{-4}$ (Inert)", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$ (Active Minterm)"])

        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_ylabel(
            f"Spatial Map Grid Tiles ($H \\times W = 10 \\times 10 = {self.spatial_rows}$ Rows)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel(
            f"Adversarial & Market Context ($11\\text{{ Opponents}} \\times 9\\text{{ Commodities}} = {self.env_cols}$ Cols)",
            fontsize=12,
            fontweight="bold",
        )

        # Label Opponents on X-Axis ticks
        opp_centers = [i * 9 + 4 for i in range(11)]
        short_opp_labels = [
            "Finn", "Walter", "Rosa", "Hana", "Mateo", "Rita",
            "Bea", "Silas", "Lena", "Cleo", "SelfPlay"
        ]
        ax.set_xticks(opp_centers)
        ax.set_xticklabels(short_opp_labels, rotation=35, ha="right", fontsize=9, fontweight="bold")

        # Label Farm Quadrants on Y-Axis ticks
        quad_centers = [12, 37, 62, 87]
        quad_labels = ["Quad 0 (Top-L)", "Quad 1 (Top-R)", "Quad 2 (Bot-L)", "Quad 3 (Bot-R)"]
        ax.set_yticks(quad_centers)
        ax.set_yticklabels(quad_labels, fontsize=9, fontweight="bold")

        # Draw grid dividing lines for farm quadrants (every 25 tiles)
        for r_split in [25, 50, 75]:
            ax.axhline(r_split - 0.5, color="#38bdf8", linestyle="--", alpha=0.8, linewidth=1.2)

        # Draw dividing lines between opponents (every 9 commodity columns)
        for c_split in range(9, self.env_cols, 9):
            ax.axvline(c_split - 0.5, color="#fbbf24", linestyle=":", alpha=0.7, linewidth=1.0)

        # Dynamic Convergence Card
        pol_pct = self.polarization_ratio * 100.0
        ent = self.entropy
        var = self.epistemic_variance
        act_cnt = self.active_minterms_count
        act_pct = (act_cnt / float(self.total_cells)) * 100.0

        ax.text(
            0.02, 0.03,
            f"[2D Floating-Point Convergence K-Map]\n"
            f"- Resolution: {self.spatial_rows} x {self.env_cols} ({self.total_cells:,} Continuous Cells)\n"
            f"- Storage: Float32 Matrix ({self.byte_size:,} bytes / {self.byte_size/1024.0:.1f} KB)\n"
            f"- Convergence Polarization: {pol_pct:.1f}% | Entropy: {ent:.3f} bits | Var: {var:.4f}\n"
            f"- Active Minterms (P >= 0.5): {act_cnt:,} ({act_pct:.1f}%) | Range: $10^{{-4}} \\to 10^{{0}}$",
            transform=ax.transAxes,
            color="white",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        return os.path.abspath(save_path)

    def render_heatmap(
        self,
        save_path: str = "kmap_polarization_heatmap.png",
        title: str = "Proportional 2D K-Map (Continuous Float32 Decision Matrix)",
        log_scale: bool = False,
    ) -> str:
        """Render and save a high-resolution labeled heatmap (linear or logarithmic)."""
        if log_scale:
            return self.render_logarithmic_heatmap(save_path=save_path, title=title)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 10), dpi=160)
        im = ax.imshow(self.data, cmap="Blues", aspect="auto", origin="upper", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(r"Continuous Decision Probability $P \in [0.0, 1.0]$", fontsize=11, fontweight="bold")
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0.0 (Inert)", "0.25", "0.5 (Uncertain)", "0.75", "1.0 (Active)"])

        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_ylabel(
            f"Spatial Map Grid Tiles ($H \\times W = 10 \\times 10 = {self.spatial_rows}$ Rows)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel(
            f"Adversarial & Market Context ($11\\text{{ Opponents}} \\times 9\\text{{ Commodities}} = {self.env_cols}$ Cols)",
            fontsize=12,
            fontweight="bold",
        )

        # Label Opponents on X-Axis ticks
        opp_centers = [i * 9 + 4 for i in range(11)]
        short_opp_labels = [
            "Finn", "Walter", "Rosa", "Hana", "Mateo", "Rita",
            "Bea", "Silas", "Lena", "Cleo", "SelfPlay"
        ]
        ax.set_xticks(opp_centers)
        ax.set_xticklabels(short_opp_labels, rotation=35, ha="right", fontsize=9, fontweight="bold")

        # Label Farm Quadrants on Y-Axis ticks
        quad_centers = [12, 37, 62, 87]
        quad_labels = ["Quad 0 (Top-L)", "Quad 1 (Top-R)", "Quad 2 (Bot-L)", "Quad 3 (Bot-R)"]
        ax.set_yticks(quad_centers)
        ax.set_yticklabels(quad_labels, fontsize=9, fontweight="bold")

        # Draw grid dividing lines for farm quadrants (every 25 tiles)
        for r_split in [25, 50, 75]:
            ax.axhline(r_split - 0.5, color="#ef4444", linestyle="--", alpha=0.8, linewidth=1.2)

        # Draw dividing lines between opponents (every 9 commodity columns)
        for c_split in range(9, self.env_cols, 9):
            ax.axvline(c_split - 0.5, color="#3b82f6", linestyle=":", alpha=0.7, linewidth=1.0)

        # Sparsity & convergence annotation
        pol_pct = self.polarization_ratio * 100.0
        ent = self.entropy
        zeros_pct = float(np.mean(self.data < 0.5)) * 100.0
        ones_pct = float(np.mean(self.data >= 0.5)) * 100.0

        ax.text(
            0.02, 0.03,
            f"[Proportional Float32 Convergence K-Map]\n"
            f"- Resolution: {self.spatial_rows} x {self.env_cols} ({self.total_cells:,} Cells) | Storage: {self.byte_size:,} bytes Float32\n"
            f"- Polarization: {pol_pct:.1f}% | Entropy: {ent:.3f} bits | Inactive: {zeros_pct:.1f}% | Active: {ones_pct:.1f}%",
            transform=ax.transAxes,
            color="white",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        return os.path.abspath(save_path)


# Alias for explicit domain nomenclature
FloatingPointConvergenceKMap = ProportionalKMap


def build_proportional_kmap_mask(
    spatial_rows: int = 100,
    env_cols: int = 99,
) -> ProportionalKMap:
    """Construct canonical continuous interaction mask for 10x10 farm tiles vs 11x9 opponents/markets."""
    mask = np.zeros((spatial_rows, env_cols), dtype=np.float32)

    # 1. High-value crop zone (Central tiles y in [3..6], x in [3..6] -> tiles 33..66)
    # interact with high-value crops (Strawberry=3, Melon=4) across aggressive opponents (4..9)
    for t_idx in range(spatial_rows):
        ty = t_idx // 10
        tx = t_idx % 10
        is_center = (3 <= ty <= 6) and (3 <= tx <= 6)
        is_corner = (ty in [0, 9]) or (tx in [0, 9])

        for col in range(env_cols):
            opp_idx = col // 9
            crop_idx = col % 9

            # Center tiles focus on high value crops against top league threats
            if is_center and crop_idx in [3, 4] and opp_idx >= 4:
                mask[t_idx, col] = 1.0
            # Corner tiles focus on staple volume crops (Wheat=0, Carrot=1) against early rushers
            elif is_corner and crop_idx in [0, 1] and opp_idx <= 3:
                mask[t_idx, col] = 0.85
            # Expansion tiles focus on Livestock / Dairy (Egg=5, Milk=6, Wool=7) against late brokers
            elif (not is_center and not is_corner) and crop_idx in [5, 6, 7] and opp_idx in [5, 6, 7, 8]:
                mask[t_idx, col] = 0.90
            else:
                # Soft background interaction probability
                dist_center = abs(ty - 4.5) + abs(tx - 4.5)
                mask[t_idx, col] = float(0.01 * math.exp(-0.2 * dist_center))

    kmap = ProportionalKMap(spatial_rows=spatial_rows, env_cols=env_cols)
    kmap.set_matrix(mask)
    return kmap
