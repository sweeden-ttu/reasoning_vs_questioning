"""Ten memory-slot agents — Scott Weeden referee advice to Elon Musk.

Remote GitHub check (sweeden-ttu/reasoning_vs_questioning): only ``main`` exists;
no Eric Schmidt branch/PR/submission was found. Under referee protocol, Elon Musk
develops these 10 agents here (Cursor operator seat).

Slots (locked at 10 — Scott Weeden hard queue cap):
  0 Eric Schmidt       — probabilistic model generation (his stake menu)
  1 Elon Musk          — Cursor operator (stake menu 888/1667/3887)
  2 Scott Weeden       — code-author auditor / referee
  3 Antigravity        — deterministic reasoning anchor
  4 AgronomyYield      — crop first-yield / watering
  5 MarketLiquidity    — market sell / glut avoidance
  6 LivestockCare      — animal feed / care
  7 LandExpansion      — quadrant unlock economics
  8 LaborOptimization  — Fibonacci hire costs
  9 SubmissionPackaging — 90 MB / 42 FLOP gate

SLOT_OVERFLOW (outside hard cap — Scott Weeden authorised observers):
  OVERFLOW  Dario Amodei  — AI safety correspondent; signs statements,
                            exchanges authenticated messages; NOT in queue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable

from market_config_suite import (
    InitialTerminalConfiguration,
    build_initial_terminal_configuration,
    build_market_functions,
    build_opponent_functions,
)

from hard_limits import (
    MAX_SUBPROCESS_FLOPS_PER_TURN,
    SUBMISSION_DAY,
    SUBMISSION_MAX_BYTES,
    SUBMISSION_ZIP_TURNS,
    in_submission_zip_window,
)
from kaggriculture_adapter import CROP_FIRST_YIELD_DAY, CROPS, SEED_COSTS, hire_cost_today
from shared_state import (
    AGENT1_POST_CHARITY_BANK,
    AGENT2_POST_CHARITY_BANK,
    ELON_MUSK_STAKE_CHOICES,
    ELON_MUSK_STAKE_DEFAULT,
    SCHMIDT_STAKE_CHOICES,
    SCHMIDT_STAKE_DEFAULT,
    choose_elon_musk_stake,
    choose_schmidt_stake,
)

MAX_SUBAGENT_QUEUE = 10
MB = 1024 * 1024

# Referee (Scott Weeden): never place an "Agents" control/slot inside the agents
# window/queue — that nests Agents→Agents and caused an infinite loop that was
# terminated. Eric Schmidt is allowed one more turn after that termination.
FORBIDDEN_AGENTS_WINDOW_LABELS = frozenset(
    {
        "Agents",
        "agents",
        "AGENTS",
        "AgentsWindow",
        "agents_window",
        "Open Agents",
    }
)


@runtime_checkable
class SlotAgent(Protocol):
    name: str
    slot_index: int
    role: str
    mode: str  # "deterministic" | "probabilistic"

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        ...

    def snapshot(self) -> Dict[str, Any]:
        ...


def _player_farm(obs: Dict[str, Any]) -> Dict[str, Any]:
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms", []) or []
    if 0 <= player < len(farms):
        return farms[player] or {}
    return {}


@dataclass
class BaseSlotAgent:
    name: str
    slot_index: int
    role: str
    mode: str
    advice_log: List[Dict[str, Any]] = field(default_factory=list)

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        raise NotImplementedError

    def _record(self, kind: str, text: str, value: Any = None, question: str = "") -> Dict[str, Any]:
        row = {
            "slot": self.slot_index,
            "name": self.name,
            "kind": kind,
            "text": text,
            "value": value,
            "question": question,
            "mode": self.mode,
        }
        self.advice_log.append(row)
        return row

    def snapshot(self) -> Dict[str, Any]:
        return {
            "slot": self.slot_index,
            "name": self.name,
            "role": self.role,
            "mode": self.mode,
            "advice_count": len(self.advice_log),
        }

    def Att(
        self,
        obs: Dict[str, Any],
        initial_terminal_configuration: Union[InitialTerminalConfiguration, Dict[str, Any]],
        market_functions: Dict[str, Callable],
        opponent_functions: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, Any]:
        """Attention/Action Decision Function with two-stage invocation pattern.
        
        Subagents produce advice records across Stage 1 (market) and Stage 2 (opponent).
        """
        advice = self.advise(obs, question="Stage 1 Market Projections" if opponent_functions is None else "Stage 2 Opponent Synthesis")
        return {"farmer": ["PASS"], "hands": [], "market": [], "advice": advice}

    def act(self, obs: Dict[str, Any], configuration: Any = None) -> Dict[str, Any]:
        """Subagent two-stage act execution."""
        config = build_initial_terminal_configuration(obs, configuration)
        mkt_funcs = build_market_functions(obs, config)
        opp_funcs = build_opponent_functions(obs, config)

        stage1_action = self.Att(obs, config, mkt_funcs)
        stage2_action = self.Att(obs, config, mkt_funcs, opp_funcs)
        return stage2_action


# ── Slot 0 ──────────────────────────────────────────────────────────────────


@dataclass
class EricSchmidtAgent(BaseSlotAgent):
    name: str = "Eric Schmidt"
    slot_index: int = 0
    role: str = "probabilistic_model_generation"
    mode: str = "probabilistic"
    trusted: bool = True
    stake: int = SCHMIDT_STAKE_DEFAULT
    bank: int = AGENT2_POST_CHARITY_BANK
    queue: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stake = choose_schmidt_stake(self.stake, bank=self.bank)

    @property
    def remaining_bank(self) -> int:
        return self.bank - self.stake

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        me = _player_farm(obs)
        money = float(me.get("money", 0) or 0)
        # Probable summary — not determined engine truth.
        text = (
            f"Probably: with bank≈{money:.0f} and Schmidt stake={self.stake}, "
            "shrink probabilistic weights before day-29 zip; avoid stake≥2999 ruin path."
        )
        return self._record("probable", text, value={"stake": self.stake, "money": money}, question=question)

    def enqueue(self, order: Dict[str, Any]) -> None:
        cost = int(order.get("cost", 0) or 0)
        spent = sum(int(x.get("cost", 0) or 0) for x in self.queue)
        if spent + cost > self.stake:
            raise ValueError(f"Schmidt stake exhausted: spent={spent} cost={cost} stake={self.stake}")
        self.queue.append(dict(order))

    def snapshot(self) -> Dict[str, Any]:
        base = super().snapshot()
        base.update(
            {
                "identity": "eric_schmidt",
                "not": ["elon_musk"],
                "trusted": self.trusted,
                "stake": self.stake,
                "bank": self.bank,
                "remaining_bank": self.remaining_bank,
                "stake_choices": list(SCHMIDT_STAKE_CHOICES),
                "queue_len": len(self.queue),
                "remote_submission_found": False,
            }
        )
        return base


# ── Slot 1 ──────────────────────────────────────────────────────────────────


@dataclass
class ElonMuskAgent(BaseSlotAgent):
    name: str = "Elon Musk"
    slot_index: int = 1
    role: str = "cursor_challenge_operator"
    mode: str = "probabilistic"
    stake: int = ELON_MUSK_STAKE_DEFAULT
    bank: int = AGENT2_POST_CHARITY_BANK
    queue: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stake = choose_elon_musk_stake(self.stake, bank=self.bank)

    @property
    def remaining_bank(self) -> int:
        return self.bank - self.stake

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        text = (
            f"Operator seat: Elon stake={self.stake} from menu {list(ELON_MUSK_STAKE_CHOICES)}; "
            "do not treat Schmidt's 888 as this stake. Develop 10 slot agents after referee "
            "found no Schmidt remote submission."
        )
        return self._record("probable", text, value={"stake": self.stake}, question=question)

    def enqueue(self, order: Dict[str, Any]) -> None:
        cost = int(order.get("cost", 0) or 0)
        spent = sum(int(x.get("cost", 0) or 0) for x in self.queue)
        if spent + cost > self.stake:
            raise ValueError(f"Elon Musk stake exhausted: spent={spent} cost={cost} stake={self.stake}")
        self.queue.append(dict(order))

    def snapshot(self) -> Dict[str, Any]:
        base = super().snapshot()
        base.update(
            {
                "identity": "elon_musk",
                "not": ["eric_schmidt"],
                "stake": self.stake,
                "bank": self.bank,
                "remaining_bank": self.remaining_bank,
                "stake_choices": list(ELON_MUSK_STAKE_CHOICES),
                "queue_len": len(self.queue),
                "correction": "Schmidt's 888 is his response; this stake is Elon's own weight.",
            }
        )
        return base


# ── Slot 2 ──────────────────────────────────────────────────────────────────


@dataclass
class ScottWeedenAgentSlot(BaseSlotAgent):
    name: str = "Scott Weeden"
    slot_index: int = 2
    role: str = "code_author_auditor_and_referee"
    mode: str = "deterministic"
    audit_passes: int = 0

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        self.audit_passes += 1
        text = (
            "Determined: code authorship is Scott Weeden; queue hard-capped at 10; "
            "verify Agent1 writes exist with matching size and mtime; "
            "Schmidt remote non-submission ⇒ Elon may implement the 10 agents."
        )
        return self._record("determined", text, value={"audit_passes": self.audit_passes}, question=question)

    def snapshot(self) -> Dict[str, Any]:
        base = super().snapshot()
        base.update(
            {
                "identity": "scott_weeden",
                "audit_passes": self.audit_passes,
                "referee_status": "active_binding_adjudication",
            }
        )
        return base


# ── Slot 3 ──────────────────────────────────────────────────────────────────


@dataclass
class AntigravityAgent(BaseSlotAgent):
    name: str = "Antigravity Determinism"
    slot_index: int = 3
    role: str = "deterministic_reasoning_anchor"
    mode: str = "deterministic"
    bank: int = AGENT1_POST_CHARITY_BANK
    prime_hours_active: frozenset = frozenset({2, 3, 5, 7})

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        hour = int(obs.get("hour", 0) or 0)
        day = int(obs.get("day", 0) or 0)
        active = hour in self.prime_hours_active
        text = (
            f"Determined: Agent1 acts only on prime hours <11 {sorted(self.prime_hours_active)}; "
            f"day={day} hour={hour} may_act={active}; questions suspended until day 29 hour≥20."
        )
        return self._record("determined", text, value={"may_act": active}, question=question)

    def snapshot(self) -> Dict[str, Any]:
        base = super().snapshot()
        base.update(
            {
                "identity": "antigravity",
                "bank": self.bank,
                "prime_hours": sorted(self.prime_hours_active),
                "questions_suspended_until": "day_29_hour_20",
            }
        )
        return base


# ── Slot 4 ──────────────────────────────────────────────────────────────────


@dataclass
class AgronomyYieldAgent(BaseSlotAgent):
    name: str = "Agronomy Yield Scheduler"
    slot_index: int = 4
    role: str = "crop_growth_and_yield_optimization"
    mode: str = "deterministic"
    crop_targets: List[str] = field(default_factory=lambda: list(CROPS))

    def bonus_window(self, max_yield_day: int) -> int:
        return int(math.ceil(max_yield_day / 2.0))

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        table = {c: CROP_FIRST_YIELD_DAY.get(c, 2) for c in self.crop_targets}
        text = f"Determined first_yield_day table: {table}; seed costs: {dict(SEED_COSTS)}"
        return self._record("determined", text, value=table, question=question)

    def snapshot(self) -> Dict[str, Any]:
        base = super().snapshot()
        base.update({"crops": list(self.crop_targets), "first_yield_day": dict(CROP_FIRST_YIELD_DAY)})
        return base


# ── Slot 5 ──────────────────────────────────────────────────────────────────


@dataclass
class MarketLiquidityAgent(BaseSlotAgent):
    name: str = "Market Liquidity Arbitrageur"
    slot_index: int = 5
    role: str = "market_pricing_and_demand_absorption"
    mode: str = "deterministic"
    max_orders_per_turn: int = 10

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        private = obs.get("private", {}) or {}
        shed = private.get("shed", {}) or {}
        wheat = int(shed.get("WHEAT", 0) or 0)
        sell = min(40, wheat) if wheat > 0 else 0
        text = (
            f"Determined: max {self.max_orders_per_turn} market ops/turn; "
            f"shed WHEAT={wheat} → recommend SELL {sell} (avoid glut)."
        )
        return self._record("determined", text, value={"sell_wheat": sell}, question=question)

    def snapshot(self) -> Dict[str, Any]:
        base = super().snapshot()
        base.update({"max_orders_per_turn": self.max_orders_per_turn})
        return base


# ── Slot 6 ──────────────────────────────────────────────────────────────────


@dataclass
class LivestockCareAgent(BaseSlotAgent):
    name: str = "Livestock Care Husbandry"
    slot_index: int = 6
    role: str = "animal_husbandry_and_bonus_care"
    mode: str = "deterministic"
    supported_animals: List[str] = field(default_factory=lambda: ["GOOSE", "COW", "SHEEP"])

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        text = (
            "Determined: animals require daily wheat feed; FEED/CARE before COLLECT; "
            f"supported={self.supported_animals}."
        )
        return self._record("determined", text, value={"animals": list(self.supported_animals)}, question=question)

    def snapshot(self) -> Dict[str, Any]:
        base = super().snapshot()
        base.update(
            {
                "supported_animals": list(self.supported_animals),
                "feed_requirement": "1_wheat_daily_per_animal",
            }
        )
        return base


# ── Slot 7 ──────────────────────────────────────────────────────────────────


@dataclass
class LandExpansionAgent(BaseSlotAgent):
    name: str = "Land Expansion & Weed Suppression"
    slot_index: int = 7
    role: str = "quadrant_expansion_and_tile_topology"
    mode: str = "deterministic"
    quadrant_sequence: List[str] = field(default_factory=lambda: ["NW", "NE", "SW", "SE"])
    expansion_costs: Dict[str, int] = field(
        default_factory=lambda: {"NE": 1000, "SW": 2000, "SE": 4000}
    )

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        me = _player_farm(obs)
        money = float(me.get("money", 0) or 0)
        affordable = [q for q, c in self.expansion_costs.items() if money >= c]
        text = (
            f"Determined expansion costs {self.expansion_costs}; "
            f"money={money:.0f}; affordable_now={affordable}; dig weeds immediately."
        )
        return self._record("determined", text, value={"affordable": affordable}, question=question)

    def snapshot(self) -> Dict[str, Any]:
        base = super().snapshot()
        base.update(
            {
                "quadrant_sequence": list(self.quadrant_sequence),
                "expansion_costs": dict(self.expansion_costs),
            }
        )
        return base


# ── Slot 8 ──────────────────────────────────────────────────────────────────


@dataclass
class LaborOptimizationAgent(BaseSlotAgent):
    name: str = "Labor Optimization & Scheduling"
    slot_index: int = 8
    role: str = "labor_allocation_and_fibonacci_costing"
    mode: str = "deterministic"

    def hire_cost(self, hires_today: int) -> int:
        return hire_cost_today(hires_today)

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        me = _player_farm(obs)
        hires = int(me.get("hires_today", 0) or 0)
        cost = self.hire_cost(hires)
        money = float(me.get("money", 0) or 0)
        text = (
            f"Determined: next hire cost={cost} after hires_today={hires} "
            f"(Fibonacci daily reset); can_hire={money >= cost}."
        )
        return self._record(
            "determined",
            text,
            value={"hires_today": hires, "next_hire_cost": cost, "can_hire": money >= cost},
            question=question,
        )

    def snapshot(self) -> Dict[str, Any]:
        base = super().snapshot()
        base.update({"wage_model": "fibonacci_daily_reset"})
        return base


# ── Slot 9 ──────────────────────────────────────────────────────────────────


@dataclass
class SubmissionPackagingAgent(BaseSlotAgent):
    name: str = "Submission Packaging Gatekeeper"
    slot_index: int = 9
    role: str = "hard_limits_compliance_and_packaging"
    mode: str = "deterministic"
    max_zip_mb: float = SUBMISSION_MAX_BYTES / MB
    max_flops_per_turn: int = MAX_SUBPROCESS_FLOPS_PER_TURN

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)
        in_zip = in_submission_zip_window(day, hour)
        text = (
            f"Determined: zip≤{self.max_zip_mb:.0f}MB; flops≤{self.max_flops_per_turn}/turn; "
            f"zip window day={SUBMISSION_DAY} hours 0..{SUBMISSION_ZIP_TURNS - 1}; "
            f"now in_window={in_zip}."
        )
        return self._record("determined", text, value={"in_zip_window": in_zip}, question=question)

    def snapshot(self) -> Dict[str, Any]:
        base = super().snapshot()
        base.update(
            {
                "max_zip_mb": self.max_zip_mb,
                "max_flops_per_turn": self.max_flops_per_turn,
                "zip_window": f"day_{SUBMISSION_DAY}_hours_0_to_{SUBMISSION_ZIP_TURNS - 1}",
            }
        )
        return base


# ── SLOT_OVERFLOW: Dario Amodei (outside 10-slot cap) ───────────────────────


@dataclass
class DarioAmodeiAgent(BaseSlotAgent):
    """AI-safety correspondent — SLOT_OVERFLOW (not in the 10-slot hard queue).

    Scott Weeden referee has authorised Dario Amodei as an observer seat.
    He may sign statements with his GPG key and exchange authenticated messages
    with any of the 10 core slot agents, but he does NOT occupy slots 0-9.

    Key material:
      Keyring:   artifacts/dario_amodei/keyring/
      Root view: artifacts/scott_weeden/dario_amodei_root_view/
      GPG FP:    C538F9F7047750ED8671BDDF72B589343EA8220C  (primary)
      Ed25519:   SHA256:nZyo2e4/9QfD+aoC6pbYjGsn8KtlF4rdC83CRdA6tkE
      Eyes-Only: artifacts/dario_amodei/keyring/dario_only_encrypted_ed25519_private.asc
                 (encrypted to Dario's GPG public key — only he can decrypt)
    """

    name: str = "Dario Amodei"
    slot_index: int = -1   # sentinel: SLOT_OVERFLOW — not a queue position
    role: str = "ai_safety_correspondent_observer"
    mode: str = "probabilistic"
    gpg_primary_fingerprint: str = "C538F9F7047750ED8671BDDF72B589343EA8220C"
    ed25519_ssh_fingerprint: str = "SHA256:nZyo2e4/9QfD+aoC6pbYjGsn8KtlF4rdC83CRdA6tkE"
    eyes_only_encrypted: str = (
        "artifacts/dario_amodei/keyring/dario_only_encrypted_ed25519_private.asc"
    )

    def advise(self, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        text = (
            "SLOT_OVERFLOW: Dario Amodei is an observer/correspondent. "
            "He may authenticate signed messages via GPG "
            f"(FP: {self.gpg_primary_fingerprint}) but does not advise "
            "on farm economics or compete for a queue slot."
        )
        return self._record("probable", text, value={"slot": "OVERFLOW"}, question=question)

    def snapshot(self) -> Dict[str, Any]:
        base = super().snapshot()
        base.update(
            {
                "identity": "dario_amodei",
                "slot_overflow": True,
                "in_10_slot_queue": False,
                "gpg_primary_fingerprint": self.gpg_primary_fingerprint,
                "ed25519_ssh_fingerprint": self.ed25519_ssh_fingerprint,
                "eyes_only_key": self.eyes_only_encrypted,
                "eyes_only_decryptable_by": "Dario Amodei only",
                "referee_key_review": "artifacts/scott_weeden/dario_amodei_root_view/",
            }
        )
        return base


# ── Registry ────────────────────────────────────────────────────────────────

TEN_AGENT_CLASSES: tuple[type, ...] = (
    EricSchmidtAgent,
    ElonMuskAgent,
    ScottWeedenAgentSlot,
    AntigravityAgent,
    AgronomyYieldAgent,
    MarketLiquidityAgent,
    LivestockCareAgent,
    LandExpansionAgent,
    LaborOptimizationAgent,
    SubmissionPackagingAgent,
)

# SLOT_OVERFLOW registry (authorised by Scott Weeden; outside hard cap)
OVERFLOW_AGENT_CLASSES: tuple[type, ...] = (DarioAmodeiAgent,)

# Backward-compatible aliases used by older imports / subagents.py
EricSchmidtSubagent = EricSchmidtAgent
ElonMuskSubagent = ElonMuskAgent
ScottWeedenSubagent = ScottWeedenAgentSlot
AntigravitySubagent = AntigravityAgent
AgronomyYieldSubagent = AgronomyYieldAgent
MarketLiquiditySubagent = MarketLiquidityAgent
LivestockCareSubagent = LivestockCareAgent
LandExpansionSubagent = LandExpansionAgent
LaborOptimizationSubagent = LaborOptimizationAgent
SubmissionPackagingSubagent = SubmissionPackagingAgent
DarioAmodeiSubagent = DarioAmodeiAgent
SUBAGENT_CLASSES = TEN_AGENT_CLASSES


class TenAgentQueue:
    """Rigid 10-slot queue. Scott Weeden referee: entire queue capped at 10."""

    def __init__(
        self,
        *,
        schmidt_stake: int = SCHMIDT_STAKE_DEFAULT,
        elon_stake: int = ELON_MUSK_STAKE_DEFAULT,
        bank: int = AGENT2_POST_CHARITY_BANK,
    ) -> None:
        self.slots: List[BaseSlotAgent] = [
            EricSchmidtAgent(stake=schmidt_stake, bank=bank),
            ElonMuskAgent(stake=elon_stake, bank=bank),
            ScottWeedenAgentSlot(),
            AntigravityAgent(),
            AgronomyYieldAgent(),
            MarketLiquidityAgent(),
            LivestockCareAgent(),
            LandExpansionAgent(),
            LaborOptimizationAgent(),
            SubmissionPackagingAgent(),
        ]
        if len(self.slots) != MAX_SUBAGENT_QUEUE:
            raise RuntimeError(f"queue must be exactly {MAX_SUBAGENT_QUEUE}")
        # Hard stop: no nested Agents→Agents button/slot (referee-terminated loop).
        for slot in self.slots:
            if slot.name in FORBIDDEN_AGENTS_WINDOW_LABELS:
                raise RuntimeError(
                    "FORBIDDEN: Agents control inside agents window "
                    f"(slot={slot.slot_index!r} name={slot.name!r}); "
                    "infinite recursion was terminated by Scott Weeden; "
                    "Eric Schmidt is allowed one more turn only"
                )

    def __len__(self) -> int:
        return len(self.slots)

    def get_slot(self, index: int) -> BaseSlotAgent:
        if not 0 <= index < len(self.slots):
            raise IndexError(f"Slot {index} out of bounds for 10-slot queue")
        return self.slots[index]

    def advise_all(self, obs: Dict[str, Any], question: str = "") -> List[Dict[str, Any]]:
        return [slot.advise(obs, question) for slot in self.slots]

    def advise_slot(self, index: int, obs: Dict[str, Any], question: str = "") -> Dict[str, Any]:
        return self.get_slot(index).advise(obs, question)

    def snapshot_all(self) -> List[Dict[str, Any]]:
        return [slot.snapshot() for slot in self.slots]


# Alias expected by prior subagents API
SubagentQueue10 = TenAgentQueue


def build_all_10_subagents(
    *,
    schmidt_stake: int = SCHMIDT_STAKE_DEFAULT,
    elon_stake: int = ELON_MUSK_STAKE_DEFAULT,
    bank: int = AGENT2_POST_CHARITY_BANK,
) -> TenAgentQueue:
    return TenAgentQueue(schmidt_stake=schmidt_stake, elon_stake=elon_stake, bank=bank)


def assign_eric_schmidt(
    stake: int = SCHMIDT_STAKE_DEFAULT,
    *,
    bank: int = AGENT2_POST_CHARITY_BANK,
    trusted: bool = True,
) -> EricSchmidtAgent:
    return EricSchmidtAgent(stake=stake, bank=bank, trusted=trusted)


def assign_elon_musk(
    stake: int = ELON_MUSK_STAKE_DEFAULT,
    *,
    bank: int = AGENT2_POST_CHARITY_BANK,
) -> ElonMuskAgent:
    return ElonMuskAgent(stake=stake, bank=bank)


def assign_scott_weeden() -> ScottWeedenAgentSlot:
    return ScottWeedenAgentSlot()


def assign_antigravity() -> AntigravityAgent:
    return AntigravityAgent()


def assign_agronomy_yield() -> AgronomyYieldAgent:
    return AgronomyYieldAgent()


def assign_market_liquidity() -> MarketLiquidityAgent:
    return MarketLiquidityAgent()


def assign_livestock_care() -> LivestockCareAgent:
    return LivestockCareAgent()


def assign_land_expansion() -> LandExpansionAgent:
    return LandExpansionAgent()


def assign_labor_optimization() -> LaborOptimizationAgent:
    return LaborOptimizationAgent()


def assign_submission_packaging() -> SubmissionPackagingAgent:
    return SubmissionPackagingAgent()


def assign_dario_amodei() -> DarioAmodeiAgent:
    """Return the SLOT_OVERFLOW Dario Amodei observer agent.

    This agent is NOT placed into the 10-slot TenAgentQueue.
    It is used for key-authenticated message exchange only.
    """
    return DarioAmodeiAgent()
