"""Single multi-headed attention + 120×120 opponent confusion matrix (30-day horizon).

Correct payload (replaces the mistaken 720×720 season-turn mask):
  - Horizon: 30 days (720 turns = 30 × 24 for the season clock only)
  - Opponent confusion: C ∈ R^{120×120}
      120 classes = 10 day-phase bins × 12 opponent slots (11 league + self)
  - Multi-head farm attention fused → one 10×10 decision matrix, gated by C[row]
"""

from __future__ import annotations

import io
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix

from fitted_multihead_attention import (
    GRID,
    HEAD_NAMES,
    N_HEADS,
    STATE_DIM,
    VOXEL_DIM,
    FittedMultiHeadAttentionMatrix,
    _softmax2d,
)

SEASON_DAYS = 30
TURNS_PER_DAY = 24
TOTAL_SEASON_TURNS = SEASON_DAYS * TURNS_PER_DAY  # calendar only
N_DAY_PHASES = 10  # 30 days / 3
N_OPP_SLOTS = 12  # 11 league + self
N_OPP_CLASSES = N_DAY_PHASES * N_OPP_SLOTS  # 120
CM_DIM = N_OPP_CLASSES

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
assert len(LEAGUE_OPPONENT_NAMES) == 11

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


def day_phase(day: int) -> int:
    """Map calendar day ∈ [0, 29] → phase bin ∈ [0, 9]."""
    d = int(day) % SEASON_DAYS
    return min(N_DAY_PHASES - 1, d // (SEASON_DAYS // N_DAY_PHASES))


def opp_class_id(day: int, opp_slot: int) -> int:
    """Encode (day_phase, opponent_slot) → class in [0, 120)."""
    slot = int(opp_slot) % N_OPP_SLOTS
    return day_phase(day) * N_OPP_SLOTS + slot


def decode_opp_class(class_id: int) -> Tuple[int, int]:
    cid = int(class_id) % N_OPP_CLASSES
    return divmod(cid, N_OPP_SLOTS)  # phase, slot


def confusion_row_to_spatial(row: np.ndarray, height: int = GRID, width: int = GRID) -> np.ndarray:
    """Project one 120-d confusion row → 10×10 spatial prior."""
    r = np.asarray(row, dtype=np.float32).reshape(-1)
    if r.size != CM_DIM:
        raise ValueError(f"confusion row must be length {CM_DIM}, got {r.size}")
    # 10 phases × 12 slots → phase marginals as rows, slot-pooled cols
    block = r.reshape(N_DAY_PHASES, N_OPP_SLOTS)  # (10, 12)
    phase = block.mean(axis=1)  # (10,)
    # Use first 10 of 12 slots for columns; blend last 2 into self/anchor weight
    slots10 = block[:, :10].mean(axis=0)  # (10,)
    slots10 = slots10 + 0.05 * float(block[:, 10:].mean())
    spatial = np.outer(phase + 1e-6, slots10 + 1e-6).astype(np.float32)
    return _softmax2d(spatial)


def build_opponent_confusion_120(
    *,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    seed: int = 7,
) -> np.ndarray:
    """Build / estimate C ∈ R^{120×120}.

    If true/pred class labels are provided, use sklearn confusion_matrix.
    Otherwise synthesize a structured prior from league topology + day-phase coupling.
    """
    if y_true is not None and y_pred is not None:
        yt = np.asarray(y_true, dtype=np.int64) % N_OPP_CLASSES
        yp = np.asarray(y_pred, dtype=np.int64) % N_OPP_CLASSES
        cm = confusion_matrix(yt, yp, labels=list(range(N_OPP_CLASSES))).astype(np.float32)
        # Row-normalize to probabilities (avoid zero rows)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.where(row_sum > 0, cm / np.maximum(row_sum, 1e-8), 1.0 / N_OPP_CLASSES)
        return cm.astype(np.float32)

    rng = np.random.default_rng(seed)
    cm = np.zeros((N_OPP_CLASSES, N_OPP_CLASSES), dtype=np.float32)
    for i in range(N_OPP_CLASSES):
        phase_i, slot_i = decode_opp_class(i)
        for j in range(N_OPP_CLASSES):
            phase_j, slot_j = decode_opp_class(j)
            # Diagonal dominance
            score = 0.0
            if i == j:
                score += 3.0
            # Same opponent slot across nearby phases (temporal confusion)
            if slot_i == slot_j:
                score += 1.5 * math.exp(-0.35 * abs(phase_i - phase_j))
            # Adjacent league opponents confuse each other
            d_slot = min(abs(slot_i - slot_j), N_OPP_SLOTS - abs(slot_i - slot_j))
            score += 0.8 * math.exp(-0.6 * d_slot)
            # Same phase different slots: mild market co-occurrence
            if phase_i == phase_j and slot_i != slot_j:
                score += 0.25
            score += 0.02 * float(rng.random())
            cm[i, j] = score
        cm[i] = cm[i] / max(1e-8, float(cm[i].sum()))
    return cm


@dataclass
class SingleMultiHeadAttentionOpp120:
    """Single fused MHA gated by 120×120 opponent confusion (30-day horizon)."""

    horizon_days: int = SEASON_DAYS
    n_opp_classes: int = N_OPP_CLASSES
    n_heads: int = N_HEADS
    d_model: int = 64
    d_k: int = 32
    height: int = GRID
    width: int = GRID
    head_names: Tuple[str, ...] = HEAD_NAMES
    confusion_gate_alpha: float = 0.40
    is_fitted: bool = False

    mha: Optional[FittedMultiHeadAttentionMatrix] = None
    confusion_120: Optional[np.ndarray] = None  # (120, 120)
    opp_ridge: Any = None  # state → 120-class logits
    action_ridge: Any = None
    liq_ridge: Any = None
    fused_prior: Optional[np.ndarray] = None
    fit_metrics: Dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        X: np.ndarray,
        y_act: np.ndarray,
        y_tile: np.ndarray,
        y_liq: np.ndarray,
        y_val: np.ndarray,
        *,
        y_opp_class: Optional[np.ndarray] = None,
        confusion_120: Optional[np.ndarray] = None,
        max_samples: int = 4000,
        verbose: bool = True,
    ) -> "SingleMultiHeadAttentionOpp120":
        if verbose:
            print(
                f"[*] Fitting single MHA @ {self.horizon_days}-day horizon "
                f"with {self.n_opp_classes}×{self.n_opp_classes} opponent confusion..."
            )
        t0 = time.time()
        X = np.asarray(X, dtype=np.float32)
        n = min(len(X), max_samples)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=n, replace=False) if len(X) > n else np.arange(len(X))
        Xs = X[idx]
        y_act_s = np.asarray(y_act, dtype=np.int64)[idx]
        y_tile_s = np.asarray(y_tile, dtype=np.float32)[idx]
        y_liq_s = np.asarray(y_liq, dtype=np.float32)[idx]
        y_val_s = np.asarray(y_val, dtype=np.float32)[idx]

        # Derive pseudo opponent-class labels from QKD channels if not provided:
        # K_OPP ≈ index 1001, day from Q ≈ 1000
        if y_opp_class is None:
            q = Xs[:, VOXEL_DIM]
            k = Xs[:, VOXEL_DIM + 1]
            day_est = np.clip(((1.0 - q) * SEASON_DAYS).astype(np.int64), 0, SEASON_DAYS - 1)
            # Map K wallet mass to opponent slot 0..10; 11 = self when D dominates
            d_ch = Xs[:, VOXEL_DIM + 2]
            slot = np.clip((k * 11.0).astype(np.int64), 0, 10)
            slot = np.where(d_ch > k + 0.15, N_OPP_SLOTS - 1, slot)  # self-ish
            y_opp_class = np.array(
                [opp_class_id(int(day_est[i]), int(slot[i])) for i in range(len(Xs))],
                dtype=np.int64,
            )
        else:
            y_opp_class = np.asarray(y_opp_class, dtype=np.int64)[idx] % N_OPP_CLASSES

        self.mha = FittedMultiHeadAttentionMatrix(
            n_heads=self.n_heads, d_model=self.d_model, d_k=self.d_k
        )
        self.mha.fit(Xs, y_act_s, y_tile_s, y_liq_s, y_val_s, max_samples=n, verbose=verbose)

        # Opponent-class head
        opp_oh = np.zeros((len(Xs), N_OPP_CLASSES), dtype=np.float32)
        for i, c in enumerate(y_opp_class):
            opp_oh[i, int(c) % N_OPP_CLASSES] = 1.0
        self.opp_ridge = Ridge(alpha=1.0, random_state=2)
        self.opp_ridge.fit(Xs, opp_oh)

        # Predict on train for empirical confusion
        pred_logits = self.opp_ridge.predict(Xs)
        y_pred = np.argmax(pred_logits, axis=1).astype(np.int64)
        if confusion_120 is not None:
            cm = np.asarray(confusion_120, dtype=np.float32)
            if cm.shape != (CM_DIM, CM_DIM):
                raise ValueError(f"confusion must be {(CM_DIM, CM_DIM)}, got {cm.shape}")
            self.confusion_120 = cm
        else:
            self.confusion_120 = build_opponent_confusion_120(y_true=y_opp_class, y_pred=y_pred)

        act_oh = np.zeros((len(Xs), len(MACRO_ACTIONS)), dtype=np.float32)
        for i, a in enumerate(y_act_s):
            act_oh[i, int(a) % len(MACRO_ACTIONS)] = 1.0
        self.action_ridge = Ridge(alpha=1.0, random_state=0)
        self.action_ridge.fit(Xs, act_oh)
        self.liq_ridge = Ridge(alpha=1.0, random_state=1)
        self.liq_ridge.fit(Xs, y_liq_s)

        fused_sum = np.zeros((self.height, self.width), dtype=np.float64)
        take = min(256, len(Xs))
        for i in range(take):
            fused_sum += self.mha.fused_attention_matrix(Xs[i])
        self.fused_prior = _softmax2d((fused_sum / take).astype(np.float32))

        # Confusion purity metrics
        diag = float(np.trace(self.confusion_120))
        self.is_fitted = True
        self.fit_metrics = {
            "n_samples": int(n),
            "n_heads": self.n_heads,
            "horizon_days": self.horizon_days,
            "confusion_shape": [CM_DIM, CM_DIM],
            "confusion_diag_mass": round(diag / CM_DIM, 4),
            "opp_train_acc": float(np.mean(y_pred == y_opp_class)),
            "fit_seconds": float(time.time() - t0),
            "payload_kind": "single_mha_30day_opp120_confusion",
        }
        if verbose:
            print(
                f"[+] Fitted in {self.fit_metrics['fit_seconds']:.2f}s → "
                f"~{self.serialized_size_mb():.2f} MB | "
                f"opp_acc={self.fit_metrics['opp_train_acc']:.3f} | "
                f"C_diag_mass/row≈{self.fit_metrics['confusion_diag_mass']:.3f}"
            )
        return self

    def serialized_size_mb(self) -> float:
        buf = io.BytesIO()
        pickle.dump(self, buf, protocol=pickle.HIGHEST_PROTOCOL)
        return len(buf.getvalue()) / (1024.0 * 1024.0)

    def predict_opp_class(self, state_vec: np.ndarray) -> int:
        x = np.asarray(state_vec, dtype=np.float32).reshape(1, -1)
        logits = np.asarray(self.opp_ridge.predict(x)[0], dtype=np.float32)
        return int(np.argmax(logits))

    def gated_attention_matrix(self, state_vec: np.ndarray, opp_class: Optional[int] = None) -> np.ndarray:
        if not self.is_fitted or self.mha is None or self.confusion_120 is None:
            raise RuntimeError("Model not fitted")
        fused = self.mha.fused_attention_matrix(state_vec)
        prior = self.fused_prior if self.fused_prior is not None else fused
        cid = self.predict_opp_class(state_vec) if opp_class is None else int(opp_class) % CM_DIM
        cm_spatial = confusion_row_to_spatial(self.confusion_120[cid])
        alpha = float(self.confusion_gate_alpha)
        logits = (
            np.log(fused + 1e-8)
            + 0.25 * np.log(prior + 1e-8)
            + alpha * np.log(cm_spatial + 1e-8)
        )
        return _softmax2d(logits.astype(np.float32))

    def predict_heads(self, state_vec: np.ndarray) -> Tuple[int, float]:
        x = np.asarray(state_vec, dtype=np.float32).reshape(1, -1)
        act_id = int(np.argmax(self.action_ridge.predict(x)[0]))
        liq = float(np.clip(self.liq_ridge.predict(x)[0], 0.0, 1.0))
        return act_id, liq

    def peak_tile(self, matrix: np.ndarray) -> Tuple[int, int]:
        flat = int(np.argmax(matrix))
        y, x = divmod(flat, self.width)
        return int(x), int(y)

    def step_day(self, obs: Dict[str, Any]) -> int:
        step = int(_get(obs, "step", 0) or 0)
        day = int(_get(obs, "day", step // TURNS_PER_DAY) or 0)
        return int(day) % self.horizon_days

    def Att(
        self,
        obs: Dict[str, Any],
        configuration: Any = None,
        market_functions: Optional[Dict[str, Callable]] = None,
        opponent_functions: Optional[Dict[str, Callable]] = None,
        state_vec: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
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

        matrix = self.gated_attention_matrix(state_vec)
        act_id, liq = self.predict_heads(state_vec)
        macro = ID_TO_MACRO.get(act_id, "PLANT_HIGH_VALUE_CROP")
        tx, ty = self.peak_tile(matrix)
        cid = self.predict_opp_class(state_vec)
        phase, slot = decode_opp_class(cid)

        player_idx = int(_get(obs, "player", 0) or 0)
        farms = list(_get(obs, "farms", []) or [])
        my_farm = farms[player_idx] if player_idx < len(farms) else {}
        my_money = float(_get(my_farm, "money", _get(my_farm, "cash", 3000.0)) or 3000.0)
        hands = list(_get(my_farm, "hands", []) or [])
        day = self.step_day(obs)

        if day < 10:
            crop = "WHEAT" if my_money < 1500 else "CARROT"
        elif day < 20:
            crop = "TOMATO" if my_money < 3000 else "STRAWBERRY"
        else:
            crop = "MELON" if my_money >= 4000 else "STRAWBERRY"

        # Confuse-aware aggression: high off-diagonal mass on predicted row → liquidate
        row = self.confusion_120[cid]
        confuse = float(1.0 - row[cid])  # mass off true class

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
        liq_use = liq
        if opponent_functions is not None:
            liq_use = min(1.0, liq + 0.10 + 0.25 * confuse)
        for item in COMMODITIES_LIST:
            qty = int(_get(warehouse, item, 0) or 0)
            if qty > 0 and liq_use > 0.05:
                sell_qty = max(1, int(math.ceil(qty * max(0.2, liq_use))))
                market_orders.append(["SELL", item, sell_qty])

        self._last_matrix = matrix
        self._last_opp_class = cid
        self._last_phase_slot = (phase, slot)
        raw = {"farmer": farmer, "hands": hands_orders, "market": market_orders}
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
        self.Att(obs, config, mkt)
        return self.Att(obs, config, mkt, opp)

    def save(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = path.stat().st_size / (1024.0 * 1024.0)
        return {
            "model_path": str(path.resolve()),
            "final_model_size_mb": size_mb,
            "within_100mb": size_mb <= 100.0,
            "within_90mb_zip_budget": size_mb <= 90.0,
            "horizon_days": self.horizon_days,
            "confusion_shape": list(self.confusion_120.shape) if self.confusion_120 is not None else None,
            "n_heads": self.n_heads,
            "fit_metrics": self.fit_metrics,
            "status": "SAFE" if size_mb <= 100.0 else "OVERSIZED",
            "payload_kind": "single_mha_30day_opp120_confusion",
        }

    @staticmethod
    def load(path: Union[str, Path]) -> "SingleMultiHeadAttentionOpp120":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, SingleMultiHeadAttentionOpp120):
            raise TypeError(f"Expected SingleMultiHeadAttentionOpp120, got {type(obj)}")
        return obj

    def render_confusion_heatmap(self, save_path: str = "opp120_confusion_heatmap.png") -> str:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        assert self.confusion_120 is not None
        fig, ax = plt.subplots(figsize=(10, 9), dpi=140)
        im = ax.imshow(self.confusion_120, cmap="magma", vmin=0, vmax=float(self.confusion_120.max()))
        ax.set_title("120×120 Opponent Confusion Matrix (10 phases × 12 slots)")
        ax.set_xlabel("Predicted opponent-class")
        ax.set_ylabel("True / query opponent-class")
        # Phase boundaries every 12
        for p in range(1, N_DAY_PHASES):
            ax.axhline(p * N_OPP_SLOTS - 0.5, color="cyan", lw=0.6, alpha=0.7)
            ax.axvline(p * N_OPP_SLOTS - 0.5, color="cyan", lw=0.6, alpha=0.7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return str(out.resolve())


def fit_single_mha_30day_opp120(
    *,
    model_save_path: str = "models/qkd_single_mha_30day_opp120.pkl",
    feature_cache: str = "experiments/qkd_120day_attention_features.npz",
    confusion_png: str = "opp120_confusion_heatmap.png",
    max_samples: int = 4000,
    verbose: bool = True,
) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    cache = Path(feature_cache) if Path(feature_cache).is_absolute() else root / feature_cache
    data = np.load(cache)
    X, y_act, y_tile, y_liq, y_val = data["X"], data["y_act"], data["y_tile"], data["y_liq"], data["y_val"]
    model = SingleMultiHeadAttentionOpp120(horizon_days=SEASON_DAYS)
    model.fit(X, y_act, y_tile, y_liq, y_val, max_samples=max_samples, verbose=verbose)
    save_path = Path(model_save_path) if Path(model_save_path).is_absolute() else root / model_save_path
    result = model.save(save_path)
    png = model.render_confusion_heatmap(str(root / confusion_png if not Path(confusion_png).is_absolute() else confusion_png))
    result["confusion_png"] = png
    result["n_train_samples"] = int(len(X))
    # Also dump matrix npz for clones
    npz_path = root / "artifacts" / "opp120_confusion" / "opponent_confusion_120x120.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, confusion=model.confusion_120)
    result["confusion_npz"] = str(npz_path.resolve())
    return result
