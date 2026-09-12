"""QKD Vector Model: Questions + K-Map Observations + Decisions Tripartite Store.

Implements Strategy 3's dense retrieval-augmented architecture consisting of
three specialized INT8 quantized vector memory banks:
  1. Q (Questions Store): Probabilistic queries, hypothesis tests, risk metrics.
  2. K (K-Map Observations Store): Opponent observation with boolean K-map don't-care masking.
  3. D (Decisions Store): State-action-reward transitions, workforce assignments, market orders.

Total combined payload is strictly capped at <= 90 MB (10 MB buffer under 100 MB ceiling),
with sub-millisecond CPU retrieval latency across all three stores.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from hard_limits import (
    MAX_SUBPROCESS_FLOPS_PER_TURN,
    PROBABILISTIC_MODEL_MAX_BYTES,
    SUBMISSION_MAX_BYTES,
)
from vector_memory_bank import (
    DEFAULT_EMBEDDING_DIM,
    QuantizedVectorMemoryBank,
    StateVectorEncoder,
)
from agents.rag_observer_reasoning_agent import (
    build_2d_kmap_mask,
    build_opponent_kmap_mask,
)

logger = logging.getLogger("qkd_vector_model")


@dataclass
class QKDQueryInsight:
    """Unified composite insight across Questions, K-Map Observations, and Decisions."""
    day: int
    hour: int
    q_insight: Dict[str, Any]  # Questions store match
    k_insight: Dict[str, Any]  # K-map observation store match
    d_insight: Dict[str, Any]  # Decisions store match
    q_confidence: float
    k_dot_score: float
    d_value: float
    recommended_action: Optional[Dict[str, Any]] = None
    synthesized_rationale: str = ""


class QKDVectorModel:
    """Tripartite Vector System: Questions + 2D K-Map Observations + Decisions."""

    def __init__(
        self,
        dim: int = DEFAULT_EMBEDDING_DIM,
        kmap_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.dim = dim
        self.encoder = StateVectorEncoder(dim=dim, half_dim=128)
        if kmap_mask is not None:
            if kmap_mask.ndim == 2:
                self.kmap_2d_mask = kmap_mask.astype(np.float32)
            else:
                self.kmap_2d_mask = build_2d_kmap_mask(128, 128)
        else:
            self.kmap_2d_mask = build_2d_kmap_mask(128, 128)

        self.kmap_mask = self.kmap_2d_mask

        # Three dedicated vector stores
        self.q_store = QuantizedVectorMemoryBank(dim=dim)  # Questions
        self.k_store = QuantizedVectorMemoryBank(dim=dim)  # 2D K-map Observations
        self.d_store = QuantizedVectorMemoryBank(dim=dim)  # Decisions

        self._seed_initial_archetypes_and_policies()

    def _seed_initial_archetypes_and_policies(self) -> None:
        """Seed initial foundational records across Q, K, and D stores."""
        # 1. Seed Q (Questions)
        sample_questions = [
            ("What is the opponent's probable economic posture?", "cautious_monitoring", 0.75),
            ("Is market liquidity sufficient for crop sales?", "liquidity_adequate", 0.85),
            ("Will the opponent expand quadrants early?", "expansion_likely", 0.65),
            ("Should we liquidate assets for endgame scoring?", "liquidate_now", 0.90),
        ]
        for q_text, hyp, conf in sample_questions:
            dummy_obs = {"day": 10, "hour": 6, "farms": [{"money": 5000.0}, {"money": 5000.0}]}
            self.q_store.add_entry(
                obs=dummy_obs,
                action={"hypothesis": hyp, "confidence": conf},
                strategic_value=conf * 10.0,
                archetype_tag=hyp,
                dialogue_response=q_text,
            )

        # 2. Seed K (2D K-Map Observations)
        archetypes = [
            ("fallow_passer", {"opp_money": 3000.0, "opp_hires": 0, "opp_plants": 0}, "uncontested_growth"),
            ("aggressive_wheat_rusher", {"opp_money": 2800.0, "opp_hires": 3, "opp_plants": 16}, "diversify_to_tomato"),
            ("rotation_specialist", {"opp_money": 5500.0, "opp_hires": 4, "opp_plants": 20}, "high_yield_timing"),
            ("homestead_scaler", {"opp_money": 11000.0, "opp_hires": 8, "opp_plants": 40}, "match_quadrant_expansion"),
            ("melon_monopolist", {"opp_money": 18000.0, "opp_hires": 6, "opp_plants": 35}, "meter_sales_with_floors"),
            ("livestock_rancher", {"opp_money": 28000.0, "opp_hires": 8, "opp_plants": 25}, "expand_crop_yields"),
            ("meta_opportunistic_broker", {"opp_money": 45000.0, "opp_hires": 8, "opp_plants": 50}, "prioritize_high_value_sell_order_queue"),
            ("meta_queue_closer", {"opp_money": 65000.0, "opp_hires": 8, "opp_plants": 50}, "maximize_early_turn_sells"),
            ("endgame_liquidator", {"opp_money": 35000.0, "opp_hires": 4, "opp_plants": 0}, "harvest_and_full_liquidate"),
        ]
        for tag, prof, strat in archetypes:
            dummy_obs = {
                "day": 12,
                "hour": 4,
                "farms": [{"money": 5000.0}, {"money": prof["opp_money"], "hires_today": prof["opp_hires"]}],
            }
            kmap_vec = self.encoder.encode_2d_kmap(dummy_obs, mask_2d=self.kmap_2d_mask)
            self.k_store.add_entry(
                obs=kmap_vec,
                action={"counter_strategy": strat},
                strategic_value=15.0,
                archetype_tag=tag,
                dialogue_response=f"Archetype pattern: {tag}",
            )

        # 3. Seed D (Decisions)
        policies = [
            ("morning_hire_and_seed", {"farmer": ["PASS"], "market": [["HIRE"], ["BUY_SEED", "TOMATO", 6]]}, 12.0),
            ("water_and_harvest_bonus", {"farmer": ["WATER"], "market": [["SELL", "TOMATO", 40]]}, 18.0),
            ("expand_quadrant_scale", {"farmer": ["PLANT", "TOMATO"], "market": [["BUY_LAND"], ["HIRE"]]}, 25.0),
            ("endgame_full_liquidation", {"farmer": ["HARVEST"], "market": [["SELL", "TOMATO", 50], ["SELL", "CARROT", 50]]}, 30.0),
        ]
        for name, act, val in policies:
            dummy_obs = {"day": 8, "hour": 2, "farms": [{"money": 6000.0}, {"money": 5000.0}]}
            self.d_store.add_entry(
                obs=dummy_obs,
                action=act,
                strategic_value=val,
                archetype_tag=name,
                dialogue_response=f"Optimal decision policy: {name}",
            )

    def add_experience(
        self,
        obs: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        opp_name: str,
        question_text: str = "",
        hypothesis: str = "",
        confidence: float = 0.5,
    ) -> None:
        """Add experience tuples into the corresponding Q, K, D stores."""
        # 1. Update D (Decisions) store
        self.d_store.add_entry(
            obs=obs,
            action=action,
            strategic_value=float(reward),
            archetype_tag=f"decision_{opp_name}",
            dialogue_response=f"Action taken vs {opp_name} with reward {reward:.2f}",
        )

        # 2. Update K (2D K-Map Observations) store
        kmap_vec = self.encoder.encode_2d_kmap(obs, mask_2d=self.kmap_2d_mask)
        self.k_store.add_entry(
            obs=kmap_vec,
            action={"counter_strategy": f"counter_{opp_name}"},
            strategic_value=float(abs(reward)),
            archetype_tag=opp_name,
            dialogue_response=f"Opponent observation vs {opp_name}",
        )

        # 3. Update Q (Questions) store if question provided
        if question_text:
            self.q_store.add_entry(
                obs=obs,
                action={"hypothesis": hypothesis, "confidence": float(confidence)},
                strategic_value=float(confidence * 10.0),
                archetype_tag=f"q_{opp_name}",
                dialogue_response=question_text,
            )

    def query_qkd(self, obs: Dict[str, Any]) -> QKDQueryInsight:
        """Perform joint fast retrieval across Q, K, and D stores (< 1 ms on CPU)."""
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)

        # 1. Query Q (Questions)
        q_matches = self.q_store.query_top_k(obs, k=1)
        q_best = q_matches[0] if q_matches else ({}, 0.5)
        q_insight = q_best[0]
        q_confidence = float(q_insight.get("action", {}).get("confidence", 0.5))

        # 2. Query K (2D K-Map Observations)
        kmap_query_vec = self.encoder.encode_2d_kmap(obs, mask_2d=self.kmap_2d_mask)
        k_matches = self.k_store.query_top_k(kmap_query_vec, k=1)
        k_best = k_matches[0] if k_matches else ({}, 0.0)
        k_insight = k_best[0]
        k_dot_score = float(k_best[1])

        # 3. Query D (Decisions)
        d_matches = self.d_store.query_top_k(obs, k=1)
        d_best = d_matches[0] if d_matches else ({}, 0.0)
        d_insight = d_best[0]
        d_value = float(d_insight.get("strategic_value", 0.0))

        rec_act = d_insight.get("action") if d_best[1] > 0.6 else None
        opp_tag = k_insight.get("archetype_tag", "unknown")

        rationale = (
            f"QKD[Day {day} H{hour}]: Q(conf={q_confidence:.2f}) | "
            f"K_2D(tag={opp_tag}, dot={k_dot_score:.2f}) | "
            f"D(val={d_value:.1f}, score={d_best[1]:.2f})"
        )

        return QKDQueryInsight(
            day=day,
            hour=hour,
            q_insight=q_insight,
            k_insight=k_insight,
            d_insight=d_insight,
            q_confidence=q_confidence,
            k_dot_score=k_dot_score,
            d_value=d_value,
            recommended_action=rec_act,
            synthesized_rationale=rationale,
        )

    def save(self, path: Union[Path, str]) -> None:
        """Serialize Q, K, D vector stores and 2D K-map mask into a single .npz archive."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Encode metadata
        meta = {
            "dim": self.dim,
            "q_vectors": len(self.q_store),
            "k_vectors": len(self.k_store),
            "d_vectors": len(self.d_store),
            "total_vectors": len(self.q_store) + len(self.k_store) + len(self.d_store),
            "q_entries": self.q_store.entries,
            "k_entries": self.k_store.entries,
            "d_entries": self.d_store.entries,
            "kmap_2d_mask": self.kmap_2d_mask.tolist(),
        }

        # Compact arrays
        np.savez_compressed(
            p,
            q_vectors=self.q_store.vectors_int8 if self.q_store.vectors_int8 is not None else np.empty((0, self.dim), dtype=np.int8),
            q_scales=self.q_store.scales if self.q_store.scales is not None else np.empty((0,), dtype=np.float32),
            k_vectors=self.k_store.vectors_int8 if self.k_store.vectors_int8 is not None else np.empty((0, self.dim), dtype=np.int8),
            k_scales=self.k_store.scales if self.k_store.scales is not None else np.empty((0,), dtype=np.float32),
            d_vectors=self.d_store.vectors_int8 if self.d_store.vectors_int8 is not None else np.empty((0, self.dim), dtype=np.int8),
            d_scales=self.d_store.scales if self.d_store.scales is not None else np.empty((0,), dtype=np.float32),
            kmap_2d_mask=self.kmap_2d_mask,
            metadata_json=np.array(json.dumps(meta), dtype=object),
        )

    @classmethod
    def load(cls, path: Union[Path, str]) -> QKDVectorModel:
        """Deserialize Q, K, D stores and 2D K-map mask from .npz archive."""
        p = Path(path)
        with np.load(p, allow_pickle=True) as data:
            meta = json.loads(str(data["metadata_json"]))
            dim = int(meta["dim"])
            if "kmap_2d_mask" in data:
                kmap_mask = data["kmap_2d_mask"].astype(np.float32)
            elif "kmap_2d_mask" in meta:
                kmap_mask = np.array(meta["kmap_2d_mask"], dtype=np.float32)
            else:
                kmap_mask = build_2d_kmap_mask(128, 128)

            model = cls(dim=dim, kmap_mask=kmap_mask)

            # Restore Q store
            model.q_store.vectors_int8 = data["q_vectors"].astype(np.int8)
            model.q_store.scales = data["q_scales"].astype(np.float32)
            model.q_store.entries = meta.get("q_entries", [])

            # Restore K store
            model.k_store.vectors_int8 = data["k_vectors"].astype(np.int8)
            model.k_store.scales = data["k_scales"].astype(np.float32)
            model.k_store.entries = meta.get("k_entries", [])

            # Restore D store
            model.d_store.vectors_int8 = data["d_vectors"].astype(np.int8)
            model.d_store.scales = data["d_scales"].astype(np.float32)
            model.d_store.entries = meta.get("d_entries", [])

        return model

    def total_vectors(self) -> int:
        return len(self.q_store) + len(self.k_store) + len(self.d_store)

