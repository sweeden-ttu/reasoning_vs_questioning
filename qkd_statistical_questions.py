"""Statistical Questioning Engine over the Q-K-D Vector Space.

Generates and evaluates strategic questions across Questions (Q),
2D K-Map Observations (K), and Decisions (D).
Ranks questions by statistical variance and standard deviation:
  - Highest Standard Deviation (sigma): Maximum discriminative power / high-entropy forks.
  - Low Standard Deviation (sigma): Invariant baseline background states.

Hardcoded Strategic Question Scope:
  1. Number of days remaining (Q-Channel: Q_DAYS_REMAINING)
  2. Balance of opponent's wallet (K-Channel: K_OPP_WALLET_BALANCE)
  3. Balance of subagents' wallets (D-Channel: D_SUBAGENTS_WALLET_BALANCE)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from vector_memory_bank import DEFAULT_EMBEDDING_DIM, StateVectorEncoder
except ImportError:
    from reasoning_vs_questioning.vector_memory_bank import (
        DEFAULT_EMBEDDING_DIM,
        StateVectorEncoder,
    )

logger = logging.getLogger("qkd_questions")


@dataclass
class QKDStatisticalQuestion:
    """Individual strategic question probe over Q-K-D vector observations."""
    qid: str
    channel: str  # "Q" (Days Remaining / State), "K" (Opponent Wallet), "D" (Subagents' Wallets)
    text: str
    description: str
    evaluator: Callable[[Dict[str, Any], np.ndarray, np.ndarray], float]
    # Statistical properties computed across game trajectory samples
    mean: float = 0.0
    std: float = 0.0
    variance: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    sample_count: int = 0


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


# ── Strategic Question Taxonomy Evaluators ─────────────────────────────────────

def _eval_q_days_remaining(obs: Any, s_self: Optional[np.ndarray] = None, s_opp: Optional[np.ndarray] = None) -> float:
    """Q: What is the normalized count of days remaining in the 30-day season?"""
    obs_dict = _safe_dict(obs)
    day = _safe_float(obs_dict.get("day", 0.0), 0.0)
    return float(max(0.0, (30.0 - day) / 30.0))


def _eval_k_opp_wallet_balance(obs: Any, s_self: Optional[np.ndarray] = None, s_opp: Optional[np.ndarray] = None) -> float:
    """K: What is the opponent's current wallet balance?"""
    obs_dict = _safe_dict(obs)
    player = _safe_int(obs_dict.get("player", 0), 0)
    player = min(max(player, 0), 1)
    opp = 1 - player
    farms = _safe_list(obs_dict.get("farms", []))
    opp_farm = _safe_dict(farms[opp]) if len(farms) > opp else {}
    opp_money = _safe_float(opp_farm.get("money", 0.0), 0.0)
    return float(np.log1p(max(0.0, opp_money)) / 12.0)


def _eval_d_subagents_wallet_balance(obs: Any, s_self: Optional[np.ndarray] = None, s_opp: Optional[np.ndarray] = None) -> float:
    """D: What is the wallet balance across my subagents and active workforce?"""
    obs_dict = _safe_dict(obs)
    player = _safe_int(obs_dict.get("player", 0), 0)
    player = min(max(player, 0), 1)
    farms = _safe_list(obs_dict.get("farms", []))
    my_farm = _safe_dict(farms[player]) if len(farms) > player else {}
    money = _safe_float(my_farm.get("money", 0.0), 0.0)
    hands_raw = my_farm.get("hands", [])
    hands = _safe_list(hands_raw) if not isinstance(hands_raw, str) else []
    subagent_multiplier = 1.0 + (len(hands) * 0.1)
    return float((np.log1p(max(0.0, money)) / 12.0) * subagent_multiplier)


# ── Canonical Question Registry ──────────────────────────────────────────────

CANONICAL_QKD_QUESTIONS: List[QKDStatisticalQuestion] = [
    QKDStatisticalQuestion(
        qid="Q_DAYS_REMAINING",
        channel="Q",
        text="How many days are remaining in the season?",
        description="Calculates normalized days remaining in the 30-day season (30 - day).",
        evaluator=_eval_q_days_remaining,
    ),
    QKDStatisticalQuestion(
        qid="K_OPP_WALLET_BALANCE",
        channel="K",
        text="What is the balance of the opponent's wallet?",
        description="Tracks the opponent's liquid bank money balance from 2D K-map observations.",
        evaluator=_eval_k_opp_wallet_balance,
    ),
    QKDStatisticalQuestion(
        qid="D_SUBAGENTS_WALLET_BALANCE",
        channel="D",
        text="What is the balance of my subagents' wallets?",
        description="Tracks collective capital balance and active workforce capacity across subagents.",
        evaluator=_eval_d_subagents_wallet_balance,
    ),
]


# ── Observation-to-Question Mapping Engine ──────────────────────────────────────

@dataclass
class QKDObservationMap:
    """Complete tripartite mapping of a game observation to the QKD question space."""
    day: int
    hour: int
    player: int
    days_remaining: float          # Raw days remaining (30 - day)
    opp_wallet_balance: float      # Raw opponent money
    subagents_wallet_balance: float # Raw self money * workforce capacity
    q_norm: float                  # Q-Channel: normalized days remaining [0, 1]
    k_norm: float                  # K-Channel: normalized opponent wallet [0, 1]
    d_norm: float                  # D-Channel: normalized subagents wallet [0, 1+]
    vector: np.ndarray             # Tripartite question vector [q_norm, k_norm, d_norm]
    probes: Dict[str, float]       # Keyed probe dictionary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "hour": self.hour,
            "player": self.player,
            "days_remaining": self.days_remaining,
            "opp_wallet_balance": self.opp_wallet_balance,
            "subagents_wallet_balance": self.subagents_wallet_balance,
            "q_norm": self.q_norm,
            "k_norm": self.k_norm,
            "d_norm": self.d_norm,
            "vector": self.vector.tolist(),
            "probes": dict(self.probes),
        }


def map_observation_to_questions(obs: Any) -> Dict[str, float]:
    """Map a raw observation dictionary directly to canonical question probe responses."""
    obs_dict = _safe_dict(obs)
    day = _safe_float(obs_dict.get("day", 0.0), 0.0)
    player = _safe_int(obs_dict.get("player", 0), 0)
    player = min(max(player, 0), 1)
    opp = 1 - player
    farms = _safe_list(obs_dict.get("farms", []))
    my_farm = _safe_dict(farms[player]) if len(farms) > player else {}
    opp_farm = _safe_dict(farms[opp]) if len(farms) > opp else {}

    # Q: Days Remaining
    q_days = max(0.0, (30.0 - day) / 30.0)

    # K: Opponent Wallet Balance
    opp_money = _safe_float(opp_farm.get("money", 0.0), 0.0)
    k_opp = float(np.log1p(max(0.0, opp_money)) / 12.0)

    # D: Subagents' Wallets Balance
    my_money = _safe_float(my_farm.get("money", 0.0), 0.0)
    hands_raw = my_farm.get("hands", [])
    hands = _safe_list(hands_raw) if not isinstance(hands_raw, str) else []
    subagent_multiplier = 1.0 + (len(hands) * 0.1)
    d_subagents = float((np.log1p(max(0.0, my_money)) / 12.0) * subagent_multiplier)

    return {
        "Q_DAYS_REMAINING": float(q_days),
        "K_OPP_WALLET_BALANCE": float(k_opp),
        "D_SUBAGENTS_WALLET_BALANCE": float(d_subagents),
    }


def map_observation_to_qkd(obs: Any) -> QKDObservationMap:
    """Map a raw observation dictionary to a full structured QKDObservationMap."""
    obs_dict = _safe_dict(obs)
    day = _safe_int(obs_dict.get("day", 0), 0)
    hour = _safe_int(obs_dict.get("hour", 0), 0)
    player = _safe_int(obs_dict.get("player", 0), 0)
    player = min(max(player, 0), 1)
    opp = 1 - player
    farms = _safe_list(obs_dict.get("farms", []))
    my_farm = _safe_dict(farms[player]) if len(farms) > player else {}
    opp_farm = _safe_dict(farms[opp]) if len(farms) > opp else {}

    raw_days_rem = max(0.0, 30.0 - float(day))
    raw_opp_money = _safe_float(opp_farm.get("money", 0.0), 0.0)
    raw_my_money = _safe_float(my_farm.get("money", 0.0), 0.0)
    hands_raw = my_farm.get("hands", [])
    hands = _safe_list(hands_raw) if not isinstance(hands_raw, str) else []
    subagent_mult = 1.0 + (len(hands) * 0.1)
    raw_subagents_money = raw_my_money * subagent_mult

    q_norm = max(0.0, (30.0 - float(day)) / 30.0)
    k_norm = float(np.log1p(max(0.0, raw_opp_money)) / 12.0)
    d_norm = float((np.log1p(max(0.0, raw_my_money)) / 12.0) * subagent_mult)

    probes = {
        "Q_DAYS_REMAINING": q_norm,
        "K_OPP_WALLET_BALANCE": k_norm,
        "D_SUBAGENTS_WALLET_BALANCE": d_norm,
    }
    vec = np.array([q_norm, k_norm, d_norm], dtype=np.float32)

    return QKDObservationMap(
        day=day,
        hour=hour,
        player=player,
        days_remaining=raw_days_rem,
        opp_wallet_balance=raw_opp_money,
        subagents_wallet_balance=raw_subagents_money,
        q_norm=q_norm,
        k_norm=k_norm,
        d_norm=d_norm,
        vector=vec,
        probes=probes,
    )


def map_observations_trajectory(observations: Optional[Sequence[Any]]) -> np.ndarray:
    """Map a trajectory of T observation dictionaries to a (T, 3) question activation matrix."""
    if not observations:
        return np.empty((0, 3), dtype=np.float32)
    matrix = np.zeros((len(observations), 3), dtype=np.float32)
    for t, obs in enumerate(observations):
        qkd = map_observation_to_qkd(obs)
        matrix[t, :] = qkd.vector
    return matrix


class QKDStatisticalQuestionBank:
    """Statistical Question Evaluator & Variance Ranker for Q-K-D Vectors."""

    def __init__(
        self,
        questions: Optional[Sequence[QKDStatisticalQuestion]] = None,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        try:
            from kmap_2d_mask import build_2d_kmap_mask
        except ImportError:
            try:
                from agents.rag_observer_reasoning_agent import build_2d_kmap_mask
            except ImportError:
                build_2d_kmap_mask = None  # type: ignore

        self.encoder = StateVectorEncoder(dim=embedding_dim, half_dim=128)
        if build_2d_kmap_mask is not None:
            self.kmap_2d_mask = build_2d_kmap_mask(128, 128)
        else:
            import numpy as _np
            self.kmap_2d_mask = _np.ones((128, 128), dtype=_np.float32)
        self.questions: List[QKDStatisticalQuestion] = (
            list(questions) if questions is not None else list(CANONICAL_QKD_QUESTIONS)
        )

    def evaluate_observation(self, obs: Any) -> Dict[str, float]:
        """Evaluate all questions on a single game state observation."""
        return map_observation_to_questions(obs)

    def map_observation(self, obs: Any) -> QKDObservationMap:
        """Map a single observation to structured QKDObservationMap."""
        return map_observation_to_qkd(obs)

    def map_trajectory(self, observations: Optional[Sequence[Any]]) -> np.ndarray:
        """Map sequence of observations to (T, 3) question activation matrix."""
        return map_observations_trajectory(observations)

    def fit_sample_distribution(
        self,
        observations: Optional[Sequence[Any]],
    ) -> List[QKDStatisticalQuestion]:
        """Compute mean, std, variance, and rank questions by Standard Deviation (descending)."""
        if not observations:
            return self.questions

        n = len(observations)
        raw_matrix = np.zeros((len(self.questions), n), dtype=np.float32)

        for j, obs in enumerate(observations):
            obs_dict = _safe_dict(obs)
            s_self, s_opp = self.encoder.encode_split(obs_dict)
            for i, q in enumerate(self.questions):
                raw_matrix[i, j] = float(q.evaluator(obs_dict, s_self, s_opp))

        # Compute summary statistics
        for i, q in enumerate(self.questions):
            vals = raw_matrix[i, :]
            q.mean = float(np.mean(vals))
            q.std = float(np.std(vals))
            q.variance = float(np.var(vals))
            q.min_val = float(np.min(vals))
            q.max_val = float(np.max(vals))
            q.sample_count = n

        # Sort in descending order of standard deviation (most significant first)
        self.questions.sort(key=lambda q: q.std, reverse=True)
        return self.questions

    def get_ranked_questions(self) -> List[QKDStatisticalQuestion]:
        """Return questions sorted by standard deviation descending."""
        return sorted(self.questions, key=lambda q: q.std, reverse=True)

    def get_top_k_significant_questions(self, k: int = 3) -> List[QKDStatisticalQuestion]:
        """Get the Top-K most significant (highest standard deviation) questions."""
        ranked = self.get_ranked_questions()
        return ranked[:k]

    def summary_table(self) -> List[Dict[str, Any]]:
        """Produce a formatted statistical summary dictionary of all questions."""
        ranked = self.get_ranked_questions()
        table = []
        for rank, q in enumerate(ranked, start=1):
            table.append({
                "rank": rank,
                "qid": q.qid,
                "channel": q.channel,
                "std_deviation": round(q.std, 4),
                "variance": round(q.variance, 4),
                "mean": round(q.mean, 4),
                "range": [round(q.min_val, 3), round(q.max_val, 3)],
                "question": q.text,
                "description": q.description,
            })
        return table

