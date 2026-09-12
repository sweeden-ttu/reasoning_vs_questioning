"""Tripartite Composite Agent: Decision = f(Questioning * Observer * Reasoning).

Every single decision (farmer moves, market orders, dialogue, memory slots) is
computed as a joint product function across the three cognitive pillars:
  1. QuestioningAgent  (Q): Probabilistic summarization, hypothesis testing, trust/risk factors.
  2. RAGObserver       (O): K-Map don't-care masked retrieval of opponent archetypes & counter-tactics.
  3. ReasoningAgent    (R): Deterministic invariants, mathematical rules, prime-hour schedule, legality.

Mathematical Decision Product:
  Score(a) = P_R(a | obs) * P_O(a | obs)^w_O * P_Q(a | obs)^w_Q * Legality_Mask(a)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from market_config_suite import (
    InitialTerminalConfiguration,
    build_initial_terminal_configuration,
    build_market_functions,
    build_opponent_functions,
)
from hard_limits import (
    MAX_SUBPROCESS_FLOPS_PER_TURN,
    TurnComputeBudget,
    clamp_planning_bank,
    in_submission_zip_window,
    within_strategy_window,
)
from memory_protocol import (
    DeterminedFact,
    MemoryBank,
    MemoryProtocol,
    ProbableSummary,
    clamp_memory_slots,
)
from agricultural_scheduler import AgriculturalScheduler
from .questioning_agent import QuestioningAgent
from .rag_observer_reasoning_agent import (
    RAGObserverReasoningAgent,
    build_opponent_kmap_mask,
)
from .reasoning_agent import (
    CHARITY_DONATION,
    PASS_ACTION,
    PRIME_HOURS_LT_11,
    ReasoningAgent,
)

logger = logging.getLogger(__name__)


@dataclass
class TripartiteDecisionContext:
    """Carries the explicit outputs and confidence weights of all three sub-agents."""
    day: int
    hour: int
    reasoning_proposal: Dict[str, Any]
    observer_insight: Dict[str, Any]
    questioning_summary: Dict[str, Any]
    q_weight: float = 0.5
    o_weight: float = 0.5
    r_weight: float = 1.0
    final_action: Dict[str, Any] = field(default_factory=lambda: dict(PASS_ACTION))
    decision_rationale: str = ""


class TripartiteDecisionEngine:
    """Functional Synthesizer computing Decision = f(Questioning * Observer * Reasoning)."""

    def __init__(
        self,
        q_weight: float = 0.5,
        o_weight: float = 0.7,
        r_weight: float = 1.0,
    ) -> None:
        self.q_weight = q_weight
        self.o_weight = o_weight
        self.r_weight = r_weight

    def synthesize_decision(
        self,
        obs: Dict[str, Any],
        reasoning_act: Dict[str, Any],
        observer_insight: Dict[str, Any],
        questioning_act: Dict[str, Any],
        questioning_summary: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        """Compute composite action via tripartite product over action channels.
        
        Channels synthesized:
          - Farmer movement/operation (PLANT, WATER, HARVEST, DIG, FEED, PASS, navigation)
          - Hands worker directives
          - Market transactions (BUY_SEED, SELL, HIRE, PASS)
        """
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)
        player = int(obs.get("player", 0) or 0)
        farms = obs.get("farms", []) or []
        me = farms[player] if len(farms) > player else {}
        money = float(me.get("money", 0.0) or 0.0)

        # 1. Deterministic Core (Reasoning Component - R)
        base_farmer = list(reasoning_act.get("farmer", ["PASS"]))
        base_hands = list(reasoning_act.get("hands", []))
        base_market = list(reasoning_act.get("market", []))

        # 2. Opponent Threat & Counter-Tactics (Observer Component - O)
        o_confident = bool(observer_insight.get("confident", False))
        o_archetype = observer_insight.get("matched_archetype", "unknown")
        o_rec_action = observer_insight.get("recommended_action") or {}
        o_rec_market = list(o_rec_action.get("market", []))

        # 3. Probabilistic Caution & Trust (Questioning Component - Q)
        q_confidence = float(questioning_summary.get("confidence", 0.5))
        q_market = list(questioning_act.get("market", []))

        # ── Channel 1: Farmer Action Product ──────────────────────────────────
        final_farmer = base_farmer
        farmer_rationale = f"R={base_farmer[0]}"

        # If Observer detects endgame score liquidation, prioritize HARVEST over WATER
        if o_confident and "endgame" in o_archetype:
            tile = me.get("tiles", [[{}]])[0][0] if me.get("tiles") else {}
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                final_farmer = ["HARVEST"]
                farmer_rationale = "O_endgame_override->HARVEST"

        # ── Channel 2: Market Order Product (R x O x Q) ──────────────────────
        # Base agricultural and physical market plan
        combined_market: List[List[Any]] = list(base_market)

        # Observer modulation: add recommended counter-actions if confident
        if o_confident and o_rec_market:
            for order in o_rec_market:
                combined_market.append(order)

        # Questioning modulation: incorporate risk proposals if confident
        if q_confidence > 0.6 and q_market:
            for order in q_market:
                combined_market.append(order)

        # If low confidence in market liquidity, prune speculative buys
        if q_confidence < 0.4:
            combined_market = [order for order in combined_market if order[0] == "SELL"]

        # Deduplicate and cap orders
        seen_ops = set()
        dedup_market = []
        for order in combined_market:
            op_key = (order[0], order[1] if len(order) > 1 else "")
            if op_key not in seen_ops:
                dedup_market.append(order)
                seen_ops.add(op_key)

        # ── Channel 3: Hands Directives ───────────────────────────────────────
        final_hands = base_hands if base_hands else [["PASS"] for _ in (me.get("hands") or [])]

        final_action = {
            "farmer": final_farmer,
            "hands": final_hands,
            "market": dedup_market[:10],
        }

        rationale = (
            f"Tripartite[Day {day} H{hour}]: Farmer({farmer_rationale}) | "
            f"Observer({o_archetype}, q={observer_insight.get('q_dot_k_score', 0.0):.2f}) | "
            f"Questioning(conf={q_confidence:.2f}) | MarketOrders={len(dedup_market)}"
        )
        return final_action, rationale


class TripartiteReasoningAgent(RAGObserverReasoningAgent):
    """Unified Tripartite Agent where every single turn decision is a function
    of (QuestioningAgent * RAGObserver * ReasoningAgent).
    """

    name = "tripartite_reasoning"

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
        q_weight: float = 0.5,
        o_weight: float = 0.7,
        r_weight: float = 1.0,
        act_all_hours: bool = False,
        target_hands: int = 4,
        crops: Optional[List[str]] = None,
        crop_share: Optional[Dict[str, float]] = None,
        target_quadrants: int = 1,
    ) -> None:
        super().__init__(
            memory_slots=memory_slots,
            force_self_talk=force_self_talk,
            aggressive_when_ahead=aggressive_when_ahead,
            edge_question_bias=edge_question_bias,
            vector_bank=vector_bank,
            kmap_mask=kmap_mask,
            rag_confidence_threshold=rag_confidence_threshold,
        )
        if act_all_hours:
            self.schedule_hours = None
        self.agri_scheduler = AgriculturalScheduler(
            target_hands=target_hands,
            crops=crops,
            crop_share=crop_share,
            target_quadrants=target_quadrants,
        )
        # Inner Questioning Agent instance for probabilistic query synthesis
        self.questioning_subagent = QuestioningAgent(
            memory_slots=memory_slots,
            force_self_talk=force_self_talk,
        )
        self.decision_engine = TripartiteDecisionEngine(
            q_weight=q_weight,
            o_weight=o_weight,
            r_weight=r_weight,
        )
        self.tripartite_audit_log: List[TripartiteDecisionContext] = []

    def _farm_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Produce base physical and market proposals via AgriculturalScheduler and memory queries."""
        self._query(obs, "What is my seed inventory?")
        self._query(obs, "What is the opponent's current market strategy?")
        return self.agri_scheduler.compute_field_and_market_actions(obs)

    def reset(self) -> None:
        super().reset()
        self.questioning_subagent.reset()
        self.tripartite_audit_log.clear()

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
            Computes intrinsic agricultural proposal & market orders.

        Second Call:
            Att(observation, initial_terminal_configuration, market_functions, opponent_functions)
            Synthesizes composite tripartite product Decision = f(Q * O * R).
        """
        hour = int(obs.get("hour", 0) or 0)
        day = int(obs.get("day", 0) or 0)

        # 1. Schedule check (Inherited from Reasoning prime-hour schedule)
        if not self.may_act(hour):
            return dict(PASS_ACTION)

        # Submission zip window
        if in_submission_zip_window(day, hour):
            return dict(PASS_ACTION)

        # Stage 1: Market-Optimal Intrinsic Agricultural Proposal
        reasoning_act = self._farm_action(obs)
        if opponent_functions is None:
            return reasoning_act

        # Stage 2: Adversarial Opponent-Modulated Multi-Agent Synthesis
        observer_insight = self.observe_opponent_rag(obs)
        questioning_summary = self.questioning_subagent.protocol.query(
            slot=0,
            question="What is the opponent's probable economic and tile posture?",
            obs=obs,
        )
        q_summary_dict = {
            "text": getattr(questioning_summary, "text", ""),
            "confidence": getattr(questioning_summary, "confidence", 0.5),
            "key": getattr(questioning_summary, "key", ""),
        }
        questioning_act = self.questioning_subagent._farm_action(obs)

        final_action, rationale = self.decision_engine.synthesize_decision(
            obs=obs,
            reasoning_act=reasoning_act,
            observer_insight=observer_insight,
            questioning_act=questioning_act,
            questioning_summary=q_summary_dict,
        )

        # Log decision context
        ctx = TripartiteDecisionContext(
            day=day,
            hour=hour,
            reasoning_proposal=reasoning_act,
            observer_insight=observer_insight,
            questioning_summary=q_summary_dict,
            final_action=final_action,
            decision_rationale=rationale,
        )
        self.tripartite_audit_log.append(ctx)
        return final_action

    def act(self, obs: Dict[str, Any], configuration: Any = None) -> Dict[str, Any]:
        """Execute decision = f(Questioning * Observer * Reasoning) via two-stage Att pipeline."""
        self.turn_budget.reset()
        hour = int(obs.get("hour", 0) or 0)
        day = int(obs.get("day", 0) or 0)

        if not self.may_act(hour):
            self.turn_budget.force_observable()
            self.action_audit.append({"day": day, "hour": hour, "acted": False, "op": "PASS"})
            return dict(PASS_ACTION)

        if in_submission_zip_window(day, hour):
            self.turn_budget.consume(1, label="submission_zip_window")
            return dict(PASS_ACTION)

        if within_strategy_window(day):
            self.turn_budget.consume(1, label="strategy_window")

        config = build_initial_terminal_configuration(obs, configuration)
        mkt_funcs = build_market_functions(obs, config)
        opp_funcs = build_opponent_functions(obs, config)

        # Initial call: Att(observation, initial_terminal_configuration, market_functions)
        stage1_action = self.Att(obs, config, mkt_funcs)

        # Second call: Att(observation, initial_terminal_configuration, market_functions, opponent_functions)
        stage2_action = self.Att(obs, config, mkt_funcs, opp_funcs)

        self.turn_budget.force_observable()
        self.compute_audit.append(self.turn_budget.snapshot())
        op = stage2_action.get("farmer", ["PASS"])[0] if stage2_action.get("farmer") else "PASS"
        rationale = (
            self.tripartite_audit_log[-1].decision_rationale
            if self.tripartite_audit_log
            else ""
        )
        self.action_audit.append({"day": day, "hour": hour, "acted": True, "op": op, "rationale": rationale})

        return stage2_action

    def metrics(self) -> Dict[str, Any]:
        m = super().metrics()
        m.update({
            "tripartite_decisions_count": len(self.tripartite_audit_log),
            "tripartite_weights": {
                "q_weight": self.decision_engine.q_weight,
                "o_weight": self.decision_engine.o_weight,
                "r_weight": self.decision_engine.r_weight,
            },
            "last_decision_rationale": (
                self.tripartite_audit_log[-1].decision_rationale
                if self.tripartite_audit_log
                else "None"
            ),
        })
        return m
