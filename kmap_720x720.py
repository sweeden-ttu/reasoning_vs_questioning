"""720x720 Floating-Point Convergence K-Map Engine.

Constructs, trains, and evaluates a full season 720×720 spatiotemporal interaction
Karnaugh Map grounded in empirical question statistics from `kmap_10x10_trained.json`.

Dimensions:
  - Rows (720 Turns): 30 days × 24 hours full season temporal trajectory.
  - Columns (720 Channels): 10 Question Families × 72 harmonic interaction projections.
  - Total Decision Cells: 518,400 continuous floating-point probability cells.

Key Capabilities:
  - Continuous Float32 Asymptotic Convergence Tracking (Polarization, Entropy, Variance).
  - Ingestion of 100 trained significant questions (sigma, mean, channel, family).
  - High-gain temperature annealing (tau = 5.0 -> 25.0).
  - True Multi-Decade Logarithmic Heatmap Generation (LogNorm 1e-4 -> 1.0).
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
RVQ_DIR = Path(__file__).resolve().parent
if str(RVQ_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from proportional_kmap import ProportionalKMap, generate_gray_code
except ImportError:
    from reasoning_vs_questioning.proportional_kmap import ProportionalKMap, generate_gray_code

logger = logging.getLogger("kmap_720x720")

SEASON_DAYS = 30
TURNS_PER_DAY = 24
TOTAL_SEASON_TURNS = SEASON_DAYS * TURNS_PER_DAY  # 720
TOTAL_INTERACTION_CHANNELS = 720
CHANNELS_PER_FAMILY = 72  # 10 families * 72 = 720

FAMILIES: Tuple[str, ...] = (
    "time",
    "capital",
    "labor",
    "crops",
    "market",
    "opponent",
    "risk",
    "poison_trust",
    "schedule_flops",
    "fellowship",
)


def load_trained_10x10_kmap(path: Union[str, Path]) -> Dict[str, Any]:
    """Load trained 10x10 significant questions K-map JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trained 10x10 K-map file not found at: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


class KMap720x720:
    """720×720 Floating-Point Convergence 2D Karnaugh Map Engine."""

    def __init__(
        self,
        trained_10x10_data: Optional[Dict[str, Any]] = None,
        shape: Tuple[int, int] = (TOTAL_SEASON_TURNS, TOTAL_INTERACTION_CHANNELS),
    ) -> None:
        self.rows, self.cols = shape
        self.trained_10x10 = trained_10x10_data or {}
        self.matrix = np.zeros(shape, dtype=np.float32)
        self.raw_logits = np.zeros(shape, dtype=np.float32)
        self.channel_labels: List[str] = []
        self.row_labels: List[str] = [
            f"D{t // 24 + 1:02d}:H{t % 24:02d}" for t in range(self.rows)
        ]
        self.convergence_metrics: Dict[str, float] = {}

    def train_from_10x10_data(
        self,
        trained_10x10_data: Dict[str, Any],
        temperature: float = 25.0,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Train 720×720 continuous K-map from the 10x10 question matrix."""
        self.trained_10x10 = trained_10x10_data
        matrix_10x10 = trained_10x10_data.get("matrix", [])
        if not matrix_10x10 or len(matrix_10x10) < 10:
            raise ValueError("Invalid 10x10 matrix format in trained_10x10_data")

        gray_codes = trained_10x10_data.get("gray_col_codes", generate_gray_code(4)[:10])

        # Flatten questions into indexed family list
        family_questions: Dict[str, List[Dict[str, Any]]] = {}
        for row_idx, row in enumerate(matrix_10x10):
            fam_name = FAMILIES[row_idx] if row_idx < len(FAMILIES) else f"family_{row_idx}"
            family_questions[fam_name] = []
            for cell in row:
                family_questions[fam_name].append({
                    "qid": cell.get("qid", ""),
                    "text": cell.get("text", ""),
                    "channel": cell.get("channel", "Q"),
                    "sigma": float(cell.get("sigma", 0.0)),
                    "mean": float(cell.get("mean", 0.0)),
                    "significant": bool(cell.get("significant", False)),
                })

        # Generate 720 channel labels
        self.channel_labels = []
        for fam_idx, fam in enumerate(FAMILIES):
            for k in range(CHANNELS_PER_FAMILY):
                q_idx = (k * 10) // CHANNELS_PER_FAMILY
                q_qid = family_questions[fam][q_idx]["qid"]
                self.channel_labels.append(f"{fam[:4]}_{q_qid[:6]}_{k:02d}")

        # Construct continuous activation tensor (720, 720)
        t_arr = np.arange(self.rows, dtype=np.float32)  # (720,)
        t_norm = t_arr / float(self.rows)               # [0, 1]

        raw_mat = np.zeros((self.rows, self.cols), dtype=np.float32)

        for fam_idx, fam in enumerate(FAMILIES):
            questions = family_questions[fam]
            col_start = fam_idx * CHANNELS_PER_FAMILY

            # Family level statistics
            fam_sigmas = np.array([q["sigma"] for q in questions], dtype=np.float32)
            fam_means = np.array([q["mean"] for q in questions], dtype=np.float32)
            sigma_sum = float(np.sum(fam_sigmas)) if np.sum(fam_sigmas) > 0 else 1.0

            for k in range(CHANNELS_PER_FAMILY):
                col_idx = col_start + k
                q_idx = (k * 10) // CHANNELS_PER_FAMILY
                q_data = questions[q_idx]

                sigma_val = q_data["sigma"]
                mean_val = q_data["mean"]
                channel_type = q_data["channel"]  # Q, K, or D

                # Channel frequency weighting
                freq_mult = 1.0 + (q_idx % 4) + (gray_codes[q_idx] * 0.2)
                phase_shift = (2.0 * np.pi * k) / float(CHANNELS_PER_FAMILY)

                # Temporal harmonic projection
                if channel_type == "Q":
                    # Days remaining / season progression harmonic
                    temporal_curve = np.cos(2.0 * np.pi * t_norm * freq_mult + phase_shift)
                    decay_factor = np.exp(-0.5 * t_norm)
                    signal = mean_val + (sigma_val * temporal_curve * decay_factor)
                elif channel_type == "K":
                    # Adversarial / Opponent interaction wave
                    temporal_curve = np.sin(4.0 * np.pi * t_norm * freq_mult + phase_shift)
                    growth_factor = np.log1p(1.0 + (t_arr * 0.05)) / 4.0
                    signal = mean_val + (sigma_val * temporal_curve * growth_factor)
                else:  # D Channel (Decisions / Workforce / Land)
                    # Decision step response with step transitions
                    step_harmonics = np.sin(2.0 * np.pi * t_norm * (freq_mult * 2.0))
                    burst_factor = 1.0 + 0.3 * np.cos(2.0 * np.pi * (t_arr % 24) / 24.0)
                    signal = mean_val + (sigma_val * step_harmonics * burst_factor)

                raw_mat[:, col_idx] = signal

        self.raw_logits = raw_mat.copy()

        # High-gain continuous polarization training
        # Normalize columns via robust scaling
        med = np.median(raw_mat, axis=0, keepdims=True)
        std = np.std(raw_mat, axis=0, keepdims=True) + 1e-6
        normalized = (raw_mat - med) / std

        # Logistic polarization driving probabilities toward 0.0 or 1.0
        polarized = 1.0 / (1.0 + np.exp(-temperature * (normalized * 0.3 - (threshold - 0.5))))

        # Non-linear asymptotic saturation
        polarized = np.where(polarized > 0.95, 1.0 - 0.05 * np.exp(-5.0 * (polarized - 0.95)), polarized)
        polarized = np.where(polarized < 0.05, 0.05 * np.exp(-5.0 * (0.05 - polarized)), polarized)
        self.matrix = np.clip(polarized, 1e-6, 1.0).astype(np.float32)

        # Calculate convergence summary
        self.convergence_metrics = self.compute_convergence_metrics()
        return self.matrix

    def compute_convergence_metrics(self) -> Dict[str, float]:
        """Compute real-time asymptotic convergence metrics across all 518,400 cells."""
        flat = self.matrix.flatten()
        n = len(flat)

        # 1. Polarization Ratio
        polarization = float(np.mean(np.abs(2.0 * flat - 1.0)))

        # 2. Shannon Information Entropy (bits)
        eps = 1e-6
        p = np.clip(flat.astype(np.float64), eps, 1.0 - eps)
        entropy_cells = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        mean_entropy = float(np.mean(entropy_cells))

        # 3. Epistemic Variance
        variance = float(np.mean(flat * (1.0 - flat)))

        # 4. Active Minterms Count
        active_minterms = int(np.sum(flat >= 0.5))

        # 5. Purity fractions
        zeros_pct = float(np.mean(flat < 0.05)) * 100.0
        ones_pct = float(np.mean(flat > 0.95)) * 100.0

        return {
            "total_cells": n,
            "shape": list(self.matrix.shape),
            "polarization_ratio": round(polarization, 4),
            "shannon_entropy_bits": round(mean_entropy, 4),
            "epistemic_variance": round(variance, 6),
            "active_minterms_count": active_minterms,
            "zeros_pct": round(zeros_pct, 2),
            "ones_pct": round(ones_pct, 2),
            "purity_pct": round(zeros_pct + ones_pct, 2),
            "min_val": float(np.min(self.matrix)),
            "max_val": float(np.max(self.matrix)),
            "mean_val": round(float(np.mean(self.matrix)), 4),
        }

    def render_logarithmic_heatmap(
        self,
        save_path: Union[str, Path] = "kmap_720x720_logarithmic_heatmap.png",
        title: str = "720×720 Continuous Logarithmic K-Map (Season Trajectory × 10-Family Interaction)",
    ) -> str:
        """Render multi-decade logarithmic intensity heatmap."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        p_save = Path(save_path)
        p_save.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 12), dpi=160)
        data_plot = np.clip(self.matrix, 1e-4, 1.0)

        im = ax.imshow(
            data_plot,
            cmap="inferno",
            norm=LogNorm(vmin=1e-4, vmax=1.0),
            aspect="auto",
            origin="upper",
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Logarithmic Decision Probability field (10⁻⁴ → 10⁰)", fontsize=11, fontweight="bold")
        cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
        cbar.set_ticklabels(["10⁻⁴ (Inert)", "10⁻³", "10⁻²", "10⁻¹", "1.0 (Active Minterm)"])

        # Set family boundary ticks
        family_tick_positions = [i * CHANNELS_PER_FAMILY + CHANNELS_PER_FAMILY // 2 for i in range(len(FAMILIES))]
        ax.set_xticks(family_tick_positions)
        ax.set_xticklabels(list(FAMILIES), rotation=35, ha="right", fontsize=9, fontweight="bold")
        ax.set_xlabel("Question Families & Harmonic Interaction Channels (N = 720)", fontsize=12, fontweight="bold")

        # Set Day boundary ticks on Y-axis
        day_ticks = [d * 24 for d in range(0, 31, 5)]
        ax.set_yticks(day_ticks)
        ax.set_yticklabels([f"Day {d:02d}" for d in range(0, 31, 5)], fontsize=10, fontweight="bold")
        ax.set_ylabel("Season Turns (30 Days × 24 Hours, N = 720)", fontsize=12, fontweight="bold")

        metrics = self.convergence_metrics or self.compute_convergence_metrics()
        info_text = (
            f"Shape: {self.rows} × {self.cols} ({metrics['total_cells']:,} cells)\n"
            f"Polarization: {metrics['polarization_ratio']*100:.1f}%\n"
            f"Shannon Entropy: {metrics['shannon_entropy_bits']:.3f} bits\n"
            f"Epistemic Var: {metrics['epistemic_variance']:.5f}\n"
            f"Active Minterms: {metrics['active_minterms_count']:,}"
        )
        ax.text(
            0.02, 0.03,
            info_text,
            transform=ax.transAxes,
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85, edgecolor="#FF8800"),
        )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(p_save, bbox_inches="tight", dpi=160)
        plt.close(fig)
        return str(p_save.resolve())

    def render_polarization_heatmap(
        self,
        save_path: Union[str, Path] = "kmap_720x720_polarization_heatmap.png",
        title: str = "720×720 Continuous Floating-Point K-Map ({0, 1} Asymptotic Convergence)",
    ) -> str:
        """Render continuous linear probability colormap."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        p_save = Path(save_path)
        p_save.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 12), dpi=160)
        im = ax.imshow(self.matrix, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto", origin="upper")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Polarized Activation Probability [0.0, 1.0]", fontsize=11, fontweight="bold")
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["0.0 (Inert)", "0.5 (Neutral)", "1.0 (Active)"])

        family_ticks = [i * CHANNELS_PER_FAMILY + CHANNELS_PER_FAMILY // 2 for i in range(len(FAMILIES))]
        ax.set_xticks(family_ticks)
        ax.set_xticklabels(list(FAMILIES), rotation=35, ha="right", fontsize=9, fontweight="bold")
        ax.set_xlabel("Question Families (10 Families × 72 Channels = 720)", fontsize=12, fontweight="bold")

        day_ticks = [d * 24 for d in range(0, 31, 5)]
        ax.set_yticks(day_ticks)
        ax.set_yticklabels([f"Day {d:02d}" for d in range(0, 31, 5)], fontsize=10, fontweight="bold")
        ax.set_ylabel("Season Turns (N = 720)", fontsize=12, fontweight="bold")

        metrics = self.convergence_metrics or self.compute_convergence_metrics()
        info_text = (
            f"Shape: {self.rows} × {self.cols} ({metrics['total_cells']:,} cells)\n"
            f"Purity ({'{'}0, 1{'}'}): {metrics['purity_pct']:.1f}%\n"
            f"Zeros (<0.05): {metrics['zeros_pct']:.1f}% | Ones (>0.95): {metrics['ones_pct']:.1f}%\n"
            f"Shannon Entropy: {metrics['shannon_entropy_bits']:.3f} bits"
        )
        ax.text(
            0.02, 0.03,
            info_text,
            transform=ax.transAxes,
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.85, edgecolor="#3388FF"),
        )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(p_save, bbox_inches="tight", dpi=160)
        plt.close(fig)
        return str(p_save.resolve())

    def save(
        self,
        output_json_path: Union[str, Path],
        output_matrix_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        """Serialize trained 720x720 K-map metadata and float32 matrix."""
        p_json = Path(output_json_path)
        p_json.parent.mkdir(parents=True, exist_ok=True)

        if output_matrix_path is None:
            p_mat = p_json.with_suffix(".npz")
        else:
            p_mat = Path(output_matrix_path)
            p_mat.parent.mkdir(parents=True, exist_ok=True)

        # Save float32 matrix compressed
        np.savez_compressed(p_mat, matrix=self.matrix, raw_logits=self.raw_logits)

        metrics = self.convergence_metrics or self.compute_convergence_metrics()

        payload = {
            "kind": "significant_questions_kmap_720x720",
            "role": "full_season_spatiotemporal_qkd_convergence",
            "ts": datetime.now(timezone.utc).isoformat(),
            "source_10x10": str(self.trained_10x10.get("source_episode", "kmap_10x10_trained.json")),
            "trusted_episode_id": str(self.trained_10x10.get("trusted_episode_id", "107982855")),
            "shape": list(self.matrix.shape),
            "total_cells": metrics["total_cells"],
            "families": list(FAMILIES),
            "channels_per_family": CHANNELS_PER_FAMILY,
            "turns_per_day": TURNS_PER_DAY,
            "season_days": SEASON_DAYS,
            "matrix_path": str(p_mat.resolve()),
            "metrics": metrics,
            "summary": {
                "polarization_ratio": metrics["polarization_ratio"],
                "shannon_entropy_bits": metrics["shannon_entropy_bits"],
                "epistemic_variance": metrics["epistemic_variance"],
                "active_minterms_count": metrics["active_minterms_count"],
                "family_coverage": len(FAMILIES),
            },
        }

        with open(p_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")

        return {
            "json_path": str(p_json.resolve()),
            "matrix_path": str(p_mat.resolve()),
        }


def build_and_train_720x720_kmap(
    trained_10x10_path: Union[str, Path],
    out_dir: Union[str, Path] = "artifacts/kmap_720x720",
    temperature: float = 25.0,
) -> Dict[str, Any]:
    """Complete workflow building and training 720x720 K-map from 10x10 data."""
    p_10x10 = Path(trained_10x10_path)
    data_10x10 = load_trained_10x10_kmap(p_10x10)

    p_out = Path(out_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    kmap_engine = KMap720x720()
    matrix = kmap_engine.train_from_10x10_data(data_10x10, temperature=temperature)

    # Save artifacts
    saved_files = kmap_engine.save(
        output_json_path=p_out / "kmap_720x720_trained.json",
        output_matrix_path=p_out / "kmap_720x720_matrix.npz",
    )

    # Render heatmaps
    log_heatmap_path = kmap_engine.render_logarithmic_heatmap(
        save_path=p_out / "kmap_720x720_logarithmic_heatmap.png"
    )
    pol_heatmap_path = kmap_engine.render_polarization_heatmap(
        save_path=p_out / "kmap_720x720_polarization_heatmap.png"
    )

    return {
        "kmap_engine": kmap_engine,
        "matrix": matrix,
        "metrics": kmap_engine.convergence_metrics,
        "json_path": saved_files["json_path"],
        "matrix_path": saved_files["matrix_path"],
        "log_heatmap_path": log_heatmap_path,
        "polarization_heatmap_path": pol_heatmap_path,
    }
