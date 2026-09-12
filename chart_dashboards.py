"""Canonical chart/dashboard renderers extracted for unit testing.

These mirror the savefig/heatmap paths used by notebooks and scratch builders:
- 90-day self-training analytics dashboard
- 120-day GBDT scaling dashboard
- 4-year macro market dashboard
- BC pretrain loss curve
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np

PathLike = Union[str, Path]


def _ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def render_90day_analytics_dashboard(
    save_path: PathLike,
    *,
    loss_history: Sequence[float],
    val_loss_history: Sequence[float],
    final_moneys: Sequence[float],
    opponent_names: Sequence[str],
    q_traj: Sequence[float],
    k_traj: Sequence[float],
    d_traj: Sequence[float],
    win_flags: Sequence[bool],
) -> str:
    """4-panel dashboard used by qkd_90day / scratch_build_notebook."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _ensure_parent(Path(save_path))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    axes[0, 0].plot(loss_history, color="#1f77b4", lw=2.5, marker="o", label="Policy Cross-Entropy")
    axes[0, 0].plot(val_loss_history, color="#ff7f0e", lw=2.0, marker="s", label="Value Function MSE")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.4)

    x = np.arange(len(opponent_names))
    axes[0, 1].bar(x, final_moneys, color="#2ca02c", alpha=0.85)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(list(opponent_names), rotation=35, ha="right", fontsize=8)
    axes[0, 1].set_title("Final Money vs Opponents")
    axes[0, 1].grid(True, alpha=0.4)

    steps = np.arange(len(q_traj))
    axes[1, 0].plot(steps, q_traj, color="#9467bd", lw=2.0, label="Q: Days Remaining")
    axes[1, 0].plot(steps, k_traj, color="#8c564b", lw=2.0, label="K: Opponent Wallet")
    axes[1, 0].plot(steps, d_traj, color="#17becf", lw=2.0, label="D: Subagents' Wallets")
    axes[1, 0].set_title("QKD Probe Trajectories")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.4)

    wins = sum(1 for w in win_flags if w)
    losses = len(win_flags) - wins
    axes[1, 1].bar(["Win", "Loss"], [wins, losses], color=["#2ecc71", "#e74c3c"])
    axes[1, 1].set_title("Win/Loss Counts")
    axes[1, 1].grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out.resolve())


def render_120day_gbdt_dashboard(
    save_path: PathLike,
    *,
    scaling_iterations: Sequence[int],
    model_sizes_mb: Sequence[float],
    opponent_names: Sequence[str],
    player_scores: Sequence[float],
    opponent_scores: Sequence[float],
    spatial_density: np.ndarray,
    unpacked_kmap: np.ndarray,
) -> str:
    """4-panel dashboard used by qkd_120day / scratch_build_120day_notebook."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    out = _ensure_parent(Path(save_path))
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=150)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    ax1 = axes[0, 0]
    ax1.plot(list(scaling_iterations), list(model_sizes_mb), marker="o", color="#3b82f6", linewidth=2.5)
    ax1.axhline(100.0, color="#ef4444", linestyle="--", linewidth=1.8)
    ax1.axhline(99.0, color="#10b981", linestyle=":", linewidth=1.5)
    ax1.set_title("GBDT Ensemble Capacity Scaling vs Iterations")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = axes[0, 1]
    x_pos = np.arange(len(opponent_names))
    w = 0.38
    ax2.bar(x_pos - w / 2, player_scores, width=w, color="#10b981", alpha=0.9)
    ax2.bar(x_pos + w / 2, opponent_scores, width=w, color="#ef4444", alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(list(opponent_names), rotation=35, ha="right", fontsize=9)
    ax2.set_title("120-Day Final Net Worth vs League Opponents")
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax3 = axes[1, 0]
    dens = np.asarray(spatial_density, dtype=np.float64)
    im3 = ax3.imshow(dens, cmap="YlGn", aspect="auto")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title("10x10 Spatial Farm Tile Allocation")

    ax4 = axes[1, 1]
    unpacked = np.asarray(unpacked_kmap, dtype=np.float32)
    log_mat = np.clip(unpacked, 1e-4, 1.0)
    for r in range(min(100, log_mat.shape[0])):
        for c in range(min(99, log_mat.shape[1])):
            if unpacked[r, c] < 0.5:
                log_mat[r, c] = min(0.04, 1e-4 + 3e-3 * math.exp(-0.05 * ((r % 25) + (c % 9))))
    norm4 = mcolors.LogNorm(vmin=1e-4, vmax=1.0)
    im4 = ax4.imshow(log_mat, cmap="inferno", norm=norm4, aspect="auto", origin="upper")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_title("2D Proportional K-Map (Logarithmic)")

    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(out.resolve())


def render_4year_macro_dashboard(
    save_path: PathLike,
    *,
    seasons: Sequence[int],
    champion_scores: Sequence[float],
    opponent_scores: Sequence[float],
    price_curves: Mapping[str, Mapping[str, Sequence[float]]],
    mean_q: Sequence[float],
    mean_k: Sequence[float],
    mean_d: Sequence[float],
    margins_by_opponent: Mapping[str, float],
) -> str:
    """4-panel dashboard used by qkd_4year / scratch_build_4year_notebook."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _ensure_parent(Path(save_path))
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(seasons, champion_scores, color="#2ca02c", lw=2.2, marker="o", markersize=3)
    ax1.plot(seasons, opponent_scores, color="#d62728", lw=1.5, ls="--", alpha=0.7)
    ax1.set_title("4-Year Macro Wealth Compounding")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = fig.add_subplot(gs[0, 1])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (item, curve) in enumerate(price_curves.items()):
        ax2.plot(curve["mean_inv"], curve["mean_price"], label=item, color=colors[i % len(colors)], lw=2.0)
    ax2.set_title("Price vs Inventory Curves")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(seasons, mean_q, color="#9467bd", lw=2.0, label="Q")
    ax3.plot(seasons, mean_k, color="#8c564b", lw=2.0, label="K")
    ax3.plot(seasons, mean_d, color="#17becf", lw=2.0, label="D")
    ax3.set_title("QKD Probe Stationarity")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax4 = fig.add_subplot(gs[1, 1])
    names = list(margins_by_opponent.keys())
    vals = [margins_by_opponent[n] for n in names]
    x_pos = np.arange(len(names))
    ax4.bar(x_pos, vals, color="#3b528b", alpha=0.85)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax4.set_title("Mean Victory Margin by Opponent")
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out.resolve())


def render_bc_pretrain_loss(save_path: PathLike, losses: Sequence[float]) -> str:
    """BC pretrain loss curve from kaggriculture-self-training notebooks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _ensure_parent(Path(save_path))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(losses) + 1), list(losses), marker="o")
    ax.set(xlabel="BC epoch / day", ylabel="Loss", title="BC pretrain")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out.resolve())


def chart_output_inventory() -> List[Dict[str, Any]]:
    """Machine-readable inventory of chart/graph output sites in this repo."""
    return [
        {"id": "proportional_kmap.render_heatmap", "module": "proportional_kmap"},
        {"id": "proportional_kmap.render_logarithmic_heatmap", "module": "proportional_kmap"},
        {"id": "vector_memory_bank.save_kmap_heatmap", "module": "vector_memory_bank"},
        {"id": "build_kmap_confusion_matrix.plot_cms", "module": "scripts.build_kmap_confusion_matrix"},
        {"id": "visualize.update_experiment_plots", "module": "visualize"},
        {"id": "visualize.MetricsVisualizer.plot_loss_curves", "module": "visualize"},
        {"id": "visualize.MetricsVisualizer.plot_epsilon_decay", "module": "visualize"},
        {"id": "visualize.MetricsVisualizer.plot_buffer_size", "module": "visualize"},
        {"id": "visualize.MetricsVisualizer.plot_eval_rewards", "module": "visualize"},
        {"id": "visualize.MetricsVisualizer.plot_final_reward_distribution", "module": "visualize"},
        {"id": "visualize.MetricsVisualizer.plot_self_play_episode_metrics", "module": "visualize"},
        {"id": "visualize.MetricsVisualizer.plot_win_rate_eval", "module": "visualize"},
        {"id": "visualize.MetricsVisualizer.save_figures", "module": "visualize"},
        {"id": "visualize.MetricsVisualizer.show_figures", "module": "visualize"},
        {"id": "fol_workbench.evaluation.visualize_model", "module": "fol_workbench.evaluation"},
        {"id": "chart_dashboards.render_90day_analytics_dashboard", "module": "chart_dashboards"},
        {"id": "chart_dashboards.render_120day_gbdt_dashboard", "module": "chart_dashboards"},
        {"id": "chart_dashboards.render_4year_macro_dashboard", "module": "chart_dashboards"},
        {"id": "chart_dashboards.render_bc_pretrain_loss", "module": "chart_dashboards"},
    ]
