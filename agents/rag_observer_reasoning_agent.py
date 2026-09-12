"""RAGObserverReasoningAgent: Retrieval-Augmented Reasoning Agent with K-Map Don't-Care Filtering.

Subclass of ReasoningAgent that actively observes the opponent using a Dense
Vector Memory Bank (Strategy 3) and Karnaugh-Map (K-Map) 'don't-care' masking.

Mathematical formulation:
  - Observation Vector: x in R^D (D = 256)
  - K-Map Don't-Care Mask M in {0, 1}^D:
      M_i = 1 if feature i is an observable, decisive opponent state feature
      M_i = 0 if feature i is a "don't-care" (e.g. private unobservables, micro-noise)
  - Masked Key Vector K:
      K = M (elementwise-product) k_candidate
  - Q Query Attention Score:
      Q = dot(q_observer, K) = sum_{i in Active} q_i * k_i
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from market_config_suite import (
    InitialTerminalConfiguration,
    build_initial_terminal_configuration,
    build_market_functions,
    build_opponent_functions,
)
from hard_limits import clamp_planning_bank
from memory_protocol import (
    DeterminedFact,
    MemoryBank,
    MemoryProtocol,
    ProbableSummary,
    clamp_memory_slots,
)
from vector_memory_bank import (
    DEFAULT_EMBEDDING_DIM,
    QuantizedVectorMemoryBank,
    StateVectorEncoder,
)
from .reasoning_agent import (
    CHARITY_DONATION,
    PASS_ACTION,
    ReasoningAgent,
)

logger = logging.getLogger(__name__)


def build_2d_kmap_mask(self_dim: int = 128, opp_dim: int = 128) -> np.ndarray:
    """Construct 2-Dimensional boolean K-map mask M in {0, 1}^{128 x 128}.

    Axis 0 (Rows): My Current State (S_self in R^128)
    Axis 1 (Columns): Opponent Observation (S_opp in R^128)

    Decisive cross-interaction blocks (M_{i, j} = 1):
      1. Macro Calendar Coupling:
         Rows 0..7 (My time/harmonics) x Cols 0..7 (Opp time/harmonics)
      2. Economic Competitive Race:
         Rows 8..15 (My money/hires/quads) x Cols 8..15 (Opp money/delta/hires/quads)
      3. Market Arbitrage & Liquidation:
         Rows 26..35 (My shed harvest) x Cols 16..35 (Opp/market prices & inventory)
      4. Crop Purchasing & Diversification:
         Rows 16..25 (My seeds) x Cols 16..25 (Opp/market prices) & Cols 36..45 (Opp crops)
      5. Land Expansion & Board Scale:
         Rows 36..45 (My farm stage) x Cols 8..15 (Opp quads/hires) & Cols 36..45 (Opp stage)
      6. Endgame Scoring Transition:
         Rows 0..7 (Calendar) & Rows 26..35 (Shed) x Cols 8..15 (Opp money & delta)

    All other cells are 0 (Don't-Care cross interactions).
    """
    mask = np.zeros((self_dim, opp_dim), dtype=np.float32)

    # 1. Macro calendar coupling
    mask[0:8, 0:8] = 1.0

    # 2. Economic competitive race
    mask[8:16, 8:16] = 1.0

    # 3. Market arbitrage & sales timing
    mask[26:36, 16:36] = 1.0

    # 4. Crop purchasing & diversification
    mask[16:26, 16:26] = 1.0
    mask[16:26, 36:46] = 1.0

    # 5. Land expansion & farm scaling
    mask[36:46, 8:16] = 1.0
    mask[36:46, 36:46] = 1.0

    # 6. Endgame score liquidation
    mask[0:8, 8:12] = 1.0
    mask[26:36, 8:12] = 1.0

    return mask


def build_opponent_kmap_mask(dim: int = DEFAULT_EMBEDDING_DIM) -> np.ndarray:
    """Backward-compatible wrapper returning 2D K-map mask of shape (128, 128)."""
    return build_2d_kmap_mask(self_dim=128, opp_dim=128)


# Built-in Library of Opponent Strategic Archetypes for RAG Retrieval
DEFAULT_OPPONENT_ARCHETYPES = [
    {
        "tag": "aggressive_wheat_rusher",
        "description": "Opponent rapidly hires hands and dumps wheat early to crash market.",
        "obs_profile": {"day": 3, "hour": 6, "opp_money": 2800.0, "opp_hires": 3, "opp_plants": 16, "crop": "WHEAT"},
        "counter_strategy": "diversify_to_corn",
        "recommended_market": [["BUY_SEED", "CORN", 4]],
    },
    {
        "tag": "passive_hoarder",
        "description": "Opponent accumulates large shed inventory without selling, planning end-game dump.",
        "obs_profile": {"day": 20, "hour": 14, "opp_money": 12000.0, "opp_hires": 1, "opp_plants": 25, "crop": "WHEAT"},
        "counter_strategy": "preemptive_market_liquidation",
        "recommended_market": [["SELL", "WHEAT", 40]],
    },
    {
        "tag": "land_expander",
        "description": "Opponent prioritizes unlocking all 4 quadrants early.",
        "obs_profile": {"day": 10, "hour": 8, "opp_money": 5000.0, "opp_hires": 2, "opp_plants": 30, "crop": "WHEAT"},
        "counter_strategy": "outscale_with_hires",
        "recommended_market": [["HIRE"]],
    },
    {
        "tag": "fellowship_collaborator",
        "description": "Opponent responds positively to initial charity donation and refrains from market griefing.",
        "obs_profile": {"day": 5, "hour": 12, "opp_money": 4500.0, "opp_hires": 2, "opp_plants": 20, "crop": "WHEAT"},
        "counter_strategy": "cooperative_expansion",
        "recommended_market": [["BUY_SEED", "WHEAT", 4]],
    },
    {
        "tag": "fallow_passer",
        "description": "Opponent never plants or acts (reward floor baseline). Maximize compound growth safely.",
        "obs_profile": {"day": 2, "hour": 0, "opp_money": 3000.0, "opp_hires": 0, "opp_plants": 0, "crop": "NONE"},
        "counter_strategy": "uncontested_crop_compound",
        "recommended_market": [["HIRE"], ["BUY_SEED", "CARROT", 6]],
    },
    {
        "tag": "rotation_specialist",
        "description": "Opponent runs 3-crop rotation (wheat/carrot/tomato) with 4 hands.",
        "obs_profile": {"day": 6, "hour": 4, "opp_money": 5500.0, "opp_hires": 4, "opp_plants": 20, "crop": "TOMATO"},
        "counter_strategy": "high_yield_timing_and_workers",
        "recommended_market": [["HIRE"], ["BUY_SEED", "TOMATO", 6]],
    },
    {
        "tag": "homestead_scaler",
        "description": "Opponent expands land to NE quadrant and scales up to 8 hands.",
        "obs_profile": {"day": 12, "hour": 2, "opp_money": 11000.0, "opp_hires": 8, "opp_plants": 40, "crop": "CARROT"},
        "counter_strategy": "match_quadrant_expansion_and_scale",
        "recommended_market": [["BUY_LAND"], ["HIRE"], ["BUY_SEED", "TOMATO", 8]],
    },
    {
        "tag": "melon_monopolist",
        "description": "Opponent fertilizes melons to 6-unit cap and meters sales with floor price 120.",
        "obs_profile": {"day": 14, "hour": 6, "opp_money": 18000.0, "opp_hires": 6, "opp_plants": 35, "crop": "MELON"},
        "counter_strategy": "capture_early_market_and_diversify",
        "recommended_market": [["BUY_SEED", "TOMATO", 6], ["SELL", "CARROT", 30]],
    },
    {
        "tag": "livestock_rancher",
        "description": "Opponent builds pastures/coops, buys cows/sheep, and depends on continuous wheat feed.",
        "obs_profile": {"day": 16, "hour": 4, "opp_money": 28000.0, "opp_hires": 8, "opp_plants": 25, "crop": "WHEAT"},
        "counter_strategy": "expand_crop_yields_and_meter_wheat",
        "recommended_market": [["BUY_LAND"], ["HIRE"], ["BUY_SEED", "TOMATO", 8]],
    },
    {
        "tag": "meta_opportunistic_broker",
        "description": "Opponent runs meta field plan with opportunistic cash-timed wheat transactions.",
        "obs_profile": {"day": 18, "hour": 8, "opp_money": 45000.0, "opp_hires": 8, "opp_plants": 50, "crop": "STRAWBERRY"},
        "counter_strategy": "prioritize_high_value_sell_order_queue",
        "recommended_market": [["SELL", "TOMATO", 40], ["SELL", "CARROT", 40], ["HIRE"]],
    },
    {
        "tag": "meta_queue_closer",
        "description": "Opponent reorders sell queue in-place so buy orders stay fully funded.",
        "obs_profile": {"day": 22, "hour": 6, "opp_money": 65000.0, "opp_hires": 8, "opp_plants": 50, "crop": "STRAWBERRY"},
        "counter_strategy": "maximize_early_turn_sells_and_compound",
        "recommended_market": [["SELL", "STRAWBERRY", 40], ["SELL", "TOMATO", 40], ["HIRE"]],
    },
    {
        "tag": "endgame_score_maximizer",
        "description": "Day 28-29 opponent liquidating all assets into pure bank coins.",
        "obs_profile": {"day": 29, "hour": 4, "opp_money": 35000.0, "opp_hires": 4, "opp_plants": 0, "crop": "WHEAT"},
        "counter_strategy": "harvest_and_full_liquidate",
        "recommended_market": [["SELL", "WHEAT", 50], ["SELL", "CARROT", 50], ["SELL", "TOMATO", 50]],
    },
]


class RAGObserverReasoningAgent(ReasoningAgent):
    """Subclass of ReasoningAgent with 2-Dimensional K-Map Opponent Observation.

    Uses a 2D K-map matrix M in {0, 1}^{128 x 128} with My State (S_self) on
    Axis 0 and Opponent Observation (S_opp) on Axis 1 to eliminate don't-care cross-terms.
    """

    name = "rag_observer_reasoning"

    def __init__(
        self,
        memory_slots: int = 10,
        *,
        force_self_talk: bool = False,
        aggressive_when_ahead: bool = True,
        edge_question_bias: bool = False,
        vector_bank: Optional[QuantizedVectorMemoryBank] = None,
        kmap_mask: Optional[np.ndarray] = None,
        rag_confidence_threshold: float = 0.65,
    ) -> None:
        super().__init__(
            memory_slots=memory_slots,
            force_self_talk=force_self_talk,
            aggressive_when_ahead=aggressive_when_ahead,
            edge_question_bias=edge_question_bias,
        )
        self.encoder = StateVectorEncoder(dim=DEFAULT_EMBEDDING_DIM, half_dim=128)
        if kmap_mask is not None:
            if kmap_mask.ndim == 2:
                self.kmap_2d_mask = kmap_mask.astype(np.float32)
            else:
                self.kmap_2d_mask = build_2d_kmap_mask(128, 128)
        else:
            self.kmap_2d_mask = build_2d_kmap_mask(128, 128)

        # Legacy 1D alias for backward compatibility
        self.kmap_mask = self.kmap_2d_mask

        self.rag_confidence_threshold = rag_confidence_threshold
        self.rag_observations_log: List[Dict[str, Any]] = []

        # Initialize or seed vector bank
        if vector_bank is not None:
            self.vector_bank = vector_bank
        else:
            self.vector_bank = QuantizedVectorMemoryBank(dim=DEFAULT_EMBEDDING_DIM)
            self._seed_default_opponent_archetypes()

    def _seed_default_opponent_archetypes(self) -> None:
        """Seed vector bank with canonical opponent behavioral archetypes."""
        for item in DEFAULT_OPPONENT_ARCHETYPES:
            prof = item["obs_profile"]
            dummy_obs = {
                "day": prof["day"],
                "hour": prof["hour"],
                "player": 0,
                "farms": [
                    {"money": 3000.0, "hires_today": 0, "unlocked_quadrants": [0], "tiles": []},
                    {
                        "money": prof["opp_money"],
                        "hires_today": prof["opp_hires"],
                        "unlocked_quadrants": [0, 1] if prof["opp_money"] > 5000 else [0],
                        "tiles": [[{"kind": "PLANT", "stage": 3} for _ in range(5)] for _ in range(4)],
                    },
                ],
                "private": {"seeds": {"WHEAT": 2}, "shed": {}},
                "market": {"prices": {"WHEAT": 15.0, "CORN": 25.0}, "inventory": {}},
            }
            act = {"farmer": ["PASS"], "market": item["recommended_market"]}
            # Encode via 2D K-map
            kmap_vec = self.encoder.encode_2d_kmap(dummy_obs, mask_2d=self.kmap_2d_mask)
            self.vector_bank.add_entry(
                obs=kmap_vec,
                action=act,
                strategic_value=10.0,
                archetype_tag=item["tag"],
                dialogue_response=item["description"],
            )

    def compute_2d_kmap_interaction(self, obs: Dict[str, Any]) -> np.ndarray:
        """Compute 2D K-map bilinear interaction vector (256-D)."""
        return self.encoder.encode_2d_kmap(obs, mask_2d=self.kmap_2d_mask)

    def observe_opponent_rag(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Perform RAG retrieval over opponent state with 2D K-map don't-care filtering."""
        q_2d_vec = self.encoder.encode_2d_kmap(obs, mask_2d=self.kmap_2d_mask)
        top_k = self.vector_bank.query_top_k(q_2d_vec, k=3)

        best_tag = "standard_play"
        best_q_score = 0.0
        best_entry: Optional[Dict[str, Any]] = None

        if top_k:
            for entry, raw_score in top_k:
                tag = entry.get("archetype_tag", "")
                if raw_score > best_q_score:
                    best_q_score = raw_score
                    best_tag = tag
                    best_entry = entry

        is_confident = best_q_score >= self.rag_confidence_threshold
        result = {
            "day": int(obs.get("day", 0) or 0),
            "hour": int(obs.get("hour", 0) or 0),
            "matched_archetype": best_tag if is_confident else "unknown_posture",
            "q_dot_k_score": best_q_score,
            "confident": is_confident,
            "recommended_action": best_entry["action"] if (best_entry and is_confident) else None,
            "description": best_entry.get("dialogue_response", "") if best_entry else "",
        }
        self.rag_observations_log.append(result)

        # Record into deterministic memory protocol
        if is_confident:
            slot_idx = self._next_slot()
            summary_text = (
                f"Opponent 2D-KMap Archetype: {best_tag} (Score={best_q_score:.3f}). "
                f"Action cue: {result['description']}"
            )
            fact = DeterminedFact(
                text=summary_text,
                key=f"rag_opp_{result['day']}_{result['hour']}",
                value=result,
            )
            if slot_idx < len(self.bank.slots):
                self.bank.slots[slot_idx].content = fact
                self.bank.slots[slot_idx].facts_stored += 1

        return result

    def Att(
        self,
        obs: Dict[str, Any],
        initial_terminal_configuration: Union[InitialTerminalConfiguration, Dict[str, Any]],
        market_functions: Dict[str, Callable],
        opponent_functions: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, Any]:
        """Attention/Action Decision Function with two-stage invocation pattern.
        
        Initial Call:
            Att(observation, initial_terminal_configuration, market_functions)
            Computes baseline prime-hour farming actions.

        Second Call:
            Att(observation, initial_terminal_configuration, market_functions, opponent_functions)
            Applies RAG opponent K-Map archetype detection and counter-actions.
        """
        hour = int(obs.get("hour", 0) or 0)
        if not self.may_act(hour):
            return dict(PASS_ACTION)

        base_action = super(ReasoningAgent, self)._farm_action(obs) if hasattr(super(ReasoningAgent, self), "_farm_action") else super()._farm_action(obs)
        if opponent_functions is None:
            return base_action

        # Stage 2: Integrate RAG opponent insights
        rag_insight = self.observe_opponent_rag(obs)
        if rag_insight["confident"] and rag_insight["recommended_action"]:
            rec_act = rag_insight["recommended_action"]
            rec_market = rec_act.get("market", [])
            if rec_market:
                current_market = base_action.get("market", [])
                base_action["market"] = rec_market + current_market

        return base_action

    def metrics(self) -> Dict[str, Any]:
        m = super().metrics()
        active_cells = int(np.sum(self.kmap_2d_mask > 0))
        total_cells = int(self.kmap_2d_mask.size)
        m.update({
            "rag_observations_count": len(self.rag_observations_log),
            "rag_confident_detections": sum(1 for r in self.rag_observations_log if r.get("confident")),
            "vector_bank_size": len(self.vector_bank),
            "kmap_2d_shape": list(self.kmap_2d_mask.shape),
            "kmap_2d_active_cells": active_cells,
            "kmap_2d_density": round(active_cells / max(1, total_cells), 4),
        })
        return m
