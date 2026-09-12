"""Single multi-headed attention matrix gated by a 720×720 mask (30-day horizon).

Compresses the multi-head QKD farm policy into one fused attention matrix:
  - Multi-head fit over 1035-d state → 10×10 spatial maps (action/tile/liq/value/spatial)
  - Fuse heads → single 10×10 decision matrix
  - Gate with season mask M ∈ R^{720×720} (30 days × 24 turns rows × 720 channels)
  - No 100 MB capacity-bank padding — lean inference payload only
"""

from __future__ import annotations

import io
import json
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge

from fitted_multihead_attention import (
    ECON_DIM,
    GRID,
    HEAD_NAMES,
    N_HEADS,
    QKD_DIM,
    STATE_DIM,
    VOXEL_DIM,
    FittedMultiHeadAttentionMatrix,
    _softmax2d,
)

SEASON_DAYS = 30
TURNS_PER_DAY = 24
TOTAL_SEASON_TURNS = SEASON_DAYS * TURNS_PER_DAY  # 720
MASK_DIM = TOTAL_SEASON_TURNS  # 720×720

MACRO_ACTIONS = [
    "PLANT_HIGH_VALUE_CROP",
    "WATER_GROWING_CROPS",
    "HARVEST_MATURE_CROPS",
    "CLEAR_WEEDS",
    "EXPAND_FARM_QUADRANT",
    "PURCHASE_WORKER_HANDS",
    "PURCHASE_LIVESTOCK",
    "FERTILIZE_ACTIVE_SOIL",
    "MARKET_SELL_PRESSURE",
    "PASS_ALIGN",
]
ID_TO_MACRO = {i: a for i, a in enumerate(MACRO_ACTIONS)}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def load_720_mask(path: Union[str, Path]) -> np.ndarray:
    """Load float32 720×720 mask from npz (key ``matrix``) or raw .npy."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix == ".npz":
        data = np.load(p)
        if "matrix" in data.files:
            mask = np.asarray(data["matrix"], dtype=np.float32)
        else:
            mask = np.asarray(data[data.files[0]], dtype=np.float32)
    else:
        mask = np.asarray(np.load(p), dtype=np.float32)
    if mask.shape != (MASK_DIM, MASK_DIM):
        raise ValueError(f"Expected mask shape {(MASK_DIM, MASK_DIM)}, got {mask.shape}")
    return np.clip(mask, 0.0, 1.0)


def mask_row_to_spatial(mask_row: np.ndarray, height: int = GRID, width: int = GRID) -> np.ndarray:
    """Project one 720-d mask row → 10×10 spatial prior (family-pooled)."""
    row = np.asarray(mask_row, dtype=np.float32).reshape(-1)
    if row.size != MASK_DIM:
        raise ValueError(f"mask row must be length {MASK_DIM}, got {row.size}")
    # 10 families × 72 channels → pool each family to one scalar, then expand to 10×10
    fam = row.reshape(10, 72).mean(axis=1)  # (10,)
    spatial = np.outer(fam, fam).astype(np.float32)
    harmonic = np.zeros((height, width), dtype=np.float32)
    for i in range(height * width):
        y, x = divmod(i, width)
        harmonic[y, x] = float(fam[y] * (0.5 + 0.5 * fam[x]))
    prior = 0.55 * _softmax2d(spatial + 1e-6) + 0.45 * _softmax2d(harmonic + 1e-6)
    return prior.astype(np.float32)


@dataclass
class SingleMultiHeadAttention720:
    """One fused multi-head attention matrix + 720×720 season mask (30-day)."""

    horizon_days: int = SEASON_DAYS
    season_turns: int = TOTAL_SEASON_TURNS
    n_heads: int = N_HEADS
    d_model: int = 64
    d_k: int = 32
    height: int = GRID
    width: int = GRID
    head_names: Tuple[str, ...] = HEAD_NAMES
    mask_gate_alpha: float = 0.35
    is_fitted: bool = False

    # Core fitted pieces (no capacity bank)
    mha: Optional[FittedMultiHeadAttentionMatrix] = None
    mask_720: Optional[np.ndarray] = None  # (720, 720)
    action_ridge: Any = None  # state → macro action logits (10)
    liq_ridge: Any = None  # state → liquidation fraction
    fused_prior: Optional[np.ndarray] = None  # (10, 10) dataset-mean fused map
    fit_metrics: Dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        X: np.ndarray,
        y_act: np.ndarray,
        y_tile: np.ndarray,
        y_liq: np.ndarray,
        y_val: np.ndarray,
        mask_720: np.ndarray,
        *,
        max_samples: int = 4000,
        verbose: bool = True,
    ) -> "SingleMultiHeadAttention720":
        """Fit multi-head maps, fuse to one matrix prior, attach 720×720 mask."""
        mask_720 = np.asarray(mask_720, dtype=np.float32)
        if mask_720.shape != (MASK_DIM, MASK_DIM):
            raise ValueError(f"mask must be {(MASK_DIM, MASK_DIM)}, got {mask_720.shape}")
        self.mask_720 = np.clip(mask_720, 0.0, 1.0)

        if verbose:
            print(
                f"[*] Fitting single MHA @ {self.horizon_days}-day / "
                f"{self.season_turns}-turn horizon with 720×720 mask..."
            )
        t0 = time.time()
        self.mha = FittedMultiHeadAttentionMatrix(
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_k=self.d_k,
            height=self.height,
            width=self.width,
        )
        self.mha.fit(
            X,
            y_act,
            y_tile,
            y_liq,
            y_val,
            max_samples=max_samples,
            verbose=verbose,
        )

        X = np.asarray(X, dtype=np.float32)
        n = min(len(X), max_samples)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=n, replace=False) if len(X) > n else np.arange(len(X))
        Xs = X[idx]
        y_act_s = np.asarray(y_act, dtype=np.int64)[idx]
        y_liq_s = np.asarray(y_liq, dtype=np.float32)[idx]

        # Macro-action head (linear) — lean, no HistGBDT trees
        act_oh = np.zeros((len(Xs), len(MACRO_ACTIONS)), dtype=np.float32)
        for i, a in enumerate(y_act_s):
            act_oh[i, int(a) % len(MACRO_ACTIONS)] = 1.0
        self.action_ridge = Ridge(alpha=1.0, random_state=0)
        self.action_ridge.fit(Xs, act_oh)
        self.liq_ridge = Ridge(alpha=1.0, random_state=1)
        self.liq_ridge.fit(Xs, y_liq_s)

        # Dataset-mean fused attention = the single matrix prior
        fused_sum = np.zeros((self.height, self.width), dtype=np.float64)
        take = min(256, len(Xs))
        for i in range(take):
            fused_sum += self.mha.fused_attention_matrix(Xs[i])
        self.fused_prior = _softmax2d((fused_sum / take).astype(np.float32))

        self.is_fitted = True
        self.fit_metrics = {
            "n_samples": int(n),
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "horizon_days": self.horizon_days,
            "season_turns": self.season_turns,
            "mask_shape": list(self.mask_720.shape),
            "mask_polarization": float(np.mean(np.abs(2.0 * self.mask_720 - 1.0))),
            "fit_seconds": float(time.time() - t0),
            "state_dim": STATE_DIM,
            "payload_kind": "single_mha_720_30day",
        }
        if verbose:
            mb = self.serialized_size_mb()
            print(
                f"[+] Single MHA+720 mask fitted in {self.fit_metrics['fit_seconds']:.2f}s "
                f"→ ~{mb:.2f} MB lean payload"
            )
        return self

    def serialized_size_mb(self) -> float:
        buf = io.BytesIO()
        pickle.dump(self, buf, protocol=pickle.HIGHEST_PROTOCOL)
        return len(buf.getvalue()) / (1024.0 * 1024.0)

    def step_index(self, obs: Dict[str, Any]) -> int:
        step = int(_get(obs, "step", 0) or 0)
        day = int(_get(obs, "day", step // TURNS_PER_DAY) or 0)
        hour = int(_get(obs, "hour", step % TURNS_PER_DAY) or 0)
        # 30-day wrap: season-local turn
        turn = (day % self.horizon_days) * TURNS_PER_DAY + (hour % TURNS_PER_DAY)
        return int(turn % self.season_turns)

    def gated_attention_matrix(self, state_vec: np.ndarray, step: int) -> np.ndarray:
        """Fuse multi-head maps and gate with mask row → single 10×10 matrix."""
        if not self.is_fitted or self.mha is None or self.mask_720 is None:
            raise RuntimeError("Model not fitted")
        fused = self.mha.fused_attention_matrix(state_vec)
        prior = self.fused_prior if self.fused_prior is not None else fused
        mask_spatial = mask_row_to_spatial(self.mask_720[int(step) % self.season_turns])
        # Log-domain mixture then renormalize
        alpha = float(self.mask_gate_alpha)
        logits = (
            np.log(fused + 1e-8)
            + 0.25 * np.log(prior + 1e-8)
            + alpha * np.log(mask_spatial + 1e-8)
        )
        return _softmax2d(logits.astype(np.float32))

    def predict_heads(self, state_vec: np.ndarray) -> Tuple[int, float, float]:
        """Return (macro_action_id, peak_tile_score∈[0,1], liquidation_frac)."""
        x = np.asarray(state_vec, dtype=np.float32).reshape(1, -1)
        act_logits = np.asarray(self.action_ridge.predict(x)[0], dtype=np.float32)
        act_id = int(np.argmax(act_logits))
        liq = float(np.clip(self.liq_ridge.predict(x)[0], 0.0, 1.0))
        return act_id, liq, float(np.max(act_logits))

    def peak_tile(self, matrix: np.ndarray) -> Tuple[int, int]:
        flat = int(np.argmax(matrix))
        y, x = divmod(flat, self.width)
        return int(x), int(y)

    def Att(
        self,
        obs: Dict[str, Any],
        configuration: Any = None,
        market_functions: Optional[Dict[str, Callable]] = None,
        opponent_functions: Optional[Dict[str, Callable]] = None,
        state_vec: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Two-stage compatible Att: stage2 when opponent_functions is not None."""
        from market_config_suite import (
            build_initial_terminal_configuration,
            build_market_functions,
            build_opponent_functions,
            rank_sell_slots,
        )
        from voxel_state_extractor import COMMODITIES_LIST, VoxelStateExtractor

        config = (
            configuration
            if configuration is not None and hasattr(configuration, "number_of_days")
            else build_initial_terminal_configuration(obs, configuration)
        )
        # Force 30-day horizon belief
        if hasattr(config, "number_of_days"):
            config.number_of_days = self.horizon_days
        if market_functions is None:
            market_functions = build_market_functions(obs, config)
        if state_vec is None:
            extractor = VoxelStateExtractor()
            opp = (
                opponent_functions
                if opponent_functions is not None
                else build_opponent_functions(obs, config)
            )
            state_vec = extractor.extract_full_state_vector(
                obs, config, market_functions, opp
            )

        step = self.step_index(obs)
        matrix = self.gated_attention_matrix(state_vec, step)
        act_id, liq, _ = self.predict_heads(state_vec)
        macro = ID_TO_MACRO.get(act_id, "PLANT_HIGH_VALUE_CROP")
        tx, ty = self.peak_tile(matrix)

        player_idx = int(_get(obs, "player", 0) or 0)
        farms = list(_get(obs, "farms", []) or [])
        my_farm = farms[player_idx] if player_idx < len(farms) else {}
        my_money = float(
            _get(my_farm, "money", _get(my_farm, "cash", 3000.0)) or 3000.0
        )
        hands = list(_get(my_farm, "hands", []) or [])

        day = step // TURNS_PER_DAY
        if day < 10:
            crop = "WHEAT" if my_money < 1500 else "CARROT"
        elif day < 20:
            crop = "TOMATO" if my_money < 3000 else "STRAWBERRY"
        else:
            crop = "MELON" if my_money >= 4000 else "STRAWBERRY"

        if macro in ("PLANT_HIGH_VALUE_CROP", "FERTILIZE_ACTIVE_SOIL"):
            farmer = ["PLANT", tx, ty, crop]
        elif macro == "WATER_GROWING_CROPS":
            farmer = ["WATER", tx, ty]
        elif macro == "HARVEST_MATURE_CROPS":
            farmer = ["HARVEST", tx, ty]
        elif macro == "CLEAR_WEEDS":
            farmer = ["DIG"] if opponent_functions is not None else ["TEND", tx, ty]
        elif macro == "EXPAND_FARM_QUADRANT" and my_money >= 5000:
            farmer = ["EXPAND"]
        elif macro == "PURCHASE_WORKER_HANDS" and my_money >= 2000 and len(hands) < 4:
            farmer = ["HIRE_HAND"]
        elif macro == "PURCHASE_LIVESTOCK" and my_money >= 3500:
            farmer = ["BUY_ANIMAL", "COW"]
        else:
            farmer = ["TEND", tx, ty]

        hands_orders = [["WATER", (tx + i + 1) % 10, ty] for i in range(len(hands))]
        market_orders: List[List[Any]] = []
        warehouse = (
            _get(my_farm, "warehouse", {})
            or _get(my_farm, "inventory", {})
            or _get(obs.get("private", {}) or {}, "shed", {})
            or {}
        )
        # Stage2: stronger liquidation when adversarially gated
        liq_use = liq if opponent_functions is None else min(1.0, liq + 0.15)
        for item in COMMODITIES_LIST:
            qty = int(_get(warehouse, item, 0) or 0)
            if qty > 0 and liq_use > 0.05:
                sell_qty = max(1, int(math.ceil(qty * max(0.2, liq_use))))
                market_orders.append(["SELL", item, sell_qty])

        raw = {"farmer": farmer, "hands": hands_orders, "market": market_orders}
        # Stash matrix for diagnostics
        self._last_matrix = matrix
        self._last_step = step
        return rank_sell_slots(obs, raw, config)

    def act(self, obs: Dict[str, Any], configuration: Any = None) -> Dict[str, Any]:
        from market_config_suite import (
            build_initial_terminal_configuration,
            build_market_functions,
            build_opponent_functions,
        )

        config = build_initial_terminal_configuration(obs, configuration)
        if hasattr(config, "number_of_days"):
            config.number_of_days = self.horizon_days
        mkt = build_market_functions(obs, config)
        opp = build_opponent_functions(obs, config)
        self.Att(obs, config, mkt)  # stage 1
        return self.Att(obs, config, mkt, opp)  # stage 2

    def save(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = path.stat().st_size / (1024.0 * 1024.0)
        meta = {
            "model_path": str(path.resolve()),
            "final_model_size_mb": size_mb,
            "within_100mb": size_mb <= 100.0,
            "within_90mb_zip_budget": size_mb <= 90.0,
            "horizon_days": self.horizon_days,
            "mask_shape": list(self.mask_720.shape) if self.mask_720 is not None else None,
            "n_heads": self.n_heads,
            "fit_metrics": self.fit_metrics,
            "status": "SAFE" if size_mb <= 100.0 else "OVERSIZED",
            "payload_kind": "single_mha_720_30day",
        }
        return meta

    @staticmethod
    def load(path: Union[str, Path]) -> "SingleMultiHeadAttention720":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, SingleMultiHeadAttention720):
            raise TypeError(f"Expected SingleMultiHeadAttention720, got {type(obj)}")
        return obj

    def render_heatmap(self, state_vec: np.ndarray, step: int = 0, save_path: str = "") -> str:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matrix = self.gated_attention_matrix(state_vec, step)
        heads = self.mha.attention_maps(state_vec) if self.mha else None
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=140)
        axes = axes.ravel()
        if heads is not None:
            for h in range(min(5, heads.shape[0])):
                im = axes[h].imshow(heads[h], cmap="viridis")
                axes[h].set_title(f"Head: {self.head_names[h]}")
                fig.colorbar(im, ax=axes[h], fraction=0.046)
        im = axes[5].imshow(matrix, cmap="magma")
        axes[5].set_title(f"Single gated matrix (step={step})")
        fig.colorbar(im, ax=axes[5], fraction=0.046)
        fig.suptitle(
            "Single Multi-Head Attention + 720×720 mask (30-day)",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        out = Path(save_path or "single_mha_720_30day_heatmap.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return str(out.resolve())


def fit_single_mha_720_30day(
    *,
    model_save_path: str = "models/qkd_single_mha_720_30day.pkl",
    mask_path: str = "artifacts/kmap_720x720/kmap_720x720_matrix.npz",
    feature_cache: str = "experiments/qkd_120day_attention_features.npz",
    heatmap_path: str = "single_mha_720_30day_heatmap.png",
    max_samples: int = 4000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit lean single-MHA model from cached trajectories + 720×720 mask."""
    root = Path(__file__).resolve().parent.parent
    mask = load_720_mask(root / mask_path if not Path(mask_path).is_absolute() else mask_path)
    cache = root / feature_cache if not Path(feature_cache).is_absolute() else Path(feature_cache)
    if not cache.exists():
        raise FileNotFoundError(
            f"Feature cache missing: {cache}. Run GBDT/MHA harvest first or point --features."
        )
    data = np.load(cache)
    X, y_act, y_tile, y_liq, y_val = (
        data["X"],
        data["y_act"],
        data["y_tile"],
        data["y_liq"],
        data["y_val"],
    )
    # Prefer samples whose Q_DAYS_REMAINING channel implies early/mid 30-day season:
    # features layout: voxels[1000] + qkd[3] + econ[32]; Q is index 1000.
    q_days = X[:, VOXEL_DIM]
    # Keep a mixture but weight toward one-season dynamics (q closer to full remaining)
    # Soft filter: take all, fit already subsamples.
    if verbose:
        print(f"[*] Features {X.shape}; mask {mask.shape}; Q_DAYS mean={float(q_days.mean()):.3f}")

    model = SingleMultiHeadAttention720(horizon_days=SEASON_DAYS, season_turns=TOTAL_SEASON_TURNS)
    model.fit(X, y_act, y_tile, y_liq, y_val, mask, max_samples=max_samples, verbose=verbose)

    save_path = root / model_save_path if not Path(model_save_path).is_absolute() else Path(model_save_path)
    result = model.save(save_path)
    sample = X[len(X) // 2]
    heat = model.render_heatmap(sample, step=TOTAL_SEASON_TURNS // 2, save_path=str(root / heatmap_path))
    result["heatmap_path"] = heat
    result["n_train_samples"] = int(len(X))
    result["feature_dim"] = int(X.shape[1])
    result["mask_path"] = str(Path(mask_path).resolve() if Path(mask_path).is_absolute() else (root / mask_path).resolve())
    return result
