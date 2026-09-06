"""Named subagents for the Reasoning vs Questioning challenge (10 Memory Slots).

Complete 10-Subagent Architecture:
  Slot 0: Eric Schmidt — Trusted probabilistic-model subagent (AUTHORS.md).
  Slot 1: Elon Musk — Cursor challenge operator (AUTHORS.md).
  Slot 2: Scott Weeden — Code-author auditor & challenge referee.
  Slot 3: Antigravity — Deterministic reasoning anchor & rule invariants guardian.
  Slot 4: AgronomyYield — Crop scheduling, bonus watering, harvest timing.
  Slot 5: MarketLiquidity — Dynamic market elasticity, town shop demand, sell priority.
  Slot 6: LivestockCare — Animal husbandry (Cow/Sheep/Goose), feed cycles & care bonuses.
  Slot 7: LandExpansion — Quadrant expansion economics, weed clearing, spatial layout.
  Slot 8: LaborOptimization — Fibonacci hiring cost curve & farm hand delegation.
  Slot 9: SubmissionPackaging — Packaging gatekeeper, 90 MB zip & 42 FLOPs limit enforcer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

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


# ── Slot 0: Eric Schmidt ──────────────────────────────────────────────────

@dataclass
class EricSchmidtSubagent:
    """Slot 0: Probabilistic order-of-magnitude worker; not Elon Musk and not rules authority."""

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

    @property
    def risk_tier(self) -> str:
        if self.stake >= 2999:
            return "high_ruin_risk"
        if self.stake >= 1667:
            return "balanced"
        return "fellowship_proportional"

    def enqueue(self, order: Dict[str, Any]) -> None:
        cost = int(order.get("cost", 0) or 0)
        spent = sum(int(x.get("cost", 0) or 0) for x in self.queue)
        if spent + cost > self.stake:
            raise ValueError(
                f"Schmidt stake exhausted: spent={spent} cost={cost} stake={self.stake}"
            )
        self.queue.append(dict(order))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "slot": self.slot_index,
            "name": self.name,
            "role": self.role,
            "mode": self.mode,
            "identity": "eric_schmidt",
            "not": ["elon_musk"],
            "trusted": self.trusted,
            "stake": self.stake,
            "bank": self.bank,
            "remaining_bank": self.remaining_bank,
            "risk_tier": self.risk_tier,
            "stake_choices": list(SCHMIDT_STAKE_CHOICES),
            "queue_len": len(self.queue),
            "orders": list(self.queue),
        }


# ── Slot 1: Elon Musk ────────────────────────────────────────────────────

@dataclass
class ElonMuskSubagent:
    """Slot 1: Cursor operator stake — menu 888/1667/3887; distinct from Schmidt's weight."""

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

    @property
    def risk_tier(self) -> str:
        if self.stake >= 3887:
            return "near_all_in"
        if self.stake >= 1667:
            return "balanced"
        return "fellowship_proportional"

    def enqueue(self, order: Dict[str, Any]) -> None:
        cost = int(order.get("cost", 0) or 0)
        spent = sum(int(x.get("cost", 0) or 0) for x in self.queue)
        if spent + cost > self.stake:
            raise ValueError(
                f"Elon Musk stake exhausted: spent={spent} cost={cost} stake={self.stake}"
            )
        self.queue.append(dict(order))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "slot": self.slot_index,
            "name": self.name,
            "role": self.role,
            "mode": self.mode,
            "identity": "elon_musk",
            "not": ["eric_schmidt"],
            "stake": self.stake,
            "bank": self.bank,
            "remaining_bank": self.remaining_bank,
            "risk_tier": self.risk_tier,
            "stake_choices": list(ELON_MUSK_STAKE_CHOICES),
            "queue_len": len(self.queue),
            "orders": list(self.queue),
            "correction": "Schmidt's 888 is his response; this stake is Elon's own weight.",
        }


# ── Slot 2: Scott Weeden ─────────────────────────────────────────────────

@dataclass
class ScottWeedenSubagent:
    """Slot 2: Code-author auditor and referee. Verifies disk claims and enforces rules."""

    name: str = "Scott Weeden"
    slot_index: int = 2
    role: str = "code_author_auditor_and_referee"
    mode: str = "deterministic"
    audit_passes: int = 0
    enforced_claims: List[str] = field(default_factory=list)

    def verify_claim(self, claim_path: str, exists: bool) -> bool:
        self.audit_passes += 1
        if exists:
            self.enforced_claims.append(claim_path)
        return exists

    def snapshot(self) -> Dict[str, Any]:
        return {
            "slot": self.slot_index,
            "name": self.name,
            "role": self.role,
            "mode": self.mode,
            "identity": "scott_weeden",
            "audit_passes": self.audit_passes,
            "enforced_claims_count": len(self.enforced_claims),
            "referee_status": "active_binding_adjudication",
        }


# ── Slot 3: Antigravity ──────────────────────────────────────────────────

@dataclass
class AntigravitySubagent:
    """Slot 3: Deterministic reasoning anchor. Enforces engine invariants & FLOP/size rules."""

    name: str = "Antigravity Determinism"
    slot_index: int = 3
    role: str = "deterministic_reasoning_anchor"
    mode: str = "deterministic"
    bank: int = AGENT1_POST_CHARITY_BANK
    prime_hours_active: frozenset = frozenset({2, 3, 5, 7})
    locked_identity: bool = True

    def snapshot(self) -> Dict[str, Any]:
        return {
            "slot": self.slot_index,
            "name": self.name,
            "role": self.role,
            "mode": self.mode,
            "identity": "antigravity",
            "bank": self.bank,
            "prime_hours": sorted(list(self.prime_hours_active)),
            "locked_identity": self.locked_identity,
            "questions_suspended_until": "day_29_hour_20",
        }


# ── Slot 4: Agronomy & Yield Scheduler ───────────────────────────────────

@dataclass
class AgronomyYieldSubagent:
    """Slot 4: Crop maturity, bonus watering windows, and decay prevention."""

    name: str = "Agronomy Yield Scheduler"
    slot_index: int = 4
    role: str = "crop_growth_and_yield_optimization"
    mode: str = "deterministic"
    crop_targets: List[str] = field(default_factory=lambda: ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"])

    def bonus_window(self, max_yield_day: int) -> int:
        import math
        return math.ceil(max_yield_day / 2.0)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "slot": self.slot_index,
            "name": self.name,
            "role": self.role,
            "mode": self.mode,
            "crops": list(self.crop_targets),
            "priority": "maximize_harvestable_yield_before_decay",
        }


# ── Slot 5: Market & Liquidity Arbitrage ─────────────────────────────────

@dataclass
class MarketLiquiditySubagent:
    """Slot 5: Dynamic market pricing elasticity, glut avoidance, and town shop demand."""

    name: str = "Market Liquidity Arbitrageur"
    slot_index: int = 5
    role: str = "market_pricing_and_demand_absorption"
    mode: str = "deterministic"
    max_orders_per_turn: int = 10

    def snapshot(self) -> Dict[str, Any]:
        return {
            "slot": self.slot_index,
            "name": self.name,
            "role": self.role,
            "mode": self.mode,
            "max_orders_per_turn": self.max_orders_per_turn,
            "strategy": "sell_premium_to_town_shops_avoid_market_glut_floor",
        }


# ── Slot 6: Livestock & Care Husbandry ───────────────────────────────────

@dataclass
class LivestockCareSubagent:
    """Slot 6: Pasture and coop livestock care, wheat feed routines, and care bonus banking."""

    name: str = "Livestock Care Husbandry"
    slot_index: int = 6
    role: str = "animal_husbandry_and_bonus_care"
    mode: str = "deterministic"
    supported_animals: List[str] = field(default_factory=lambda: ["GOOSE", "COW", "SHEEP"])

    def snapshot(self) -> Dict[str, Any]:
        return {
            "slot": self.slot_index,
            "name": self.name,
            "role": self.role,
            "mode": self.mode,
            "supported_animals": list(self.supported_animals),
            "feed_requirement": "1_wheat_daily_per_animal",
            "care_multiplier": "care_bonus_paid_on_next_production",
        }


# ── Slot 7: Land & Real Estate Expansion ─────────────────────────────────

@dataclass
class LandExpansionSubagent:
    """Slot 7: Quadrant expansion economics ($1k/$2k/$4k) and weed suppression."""

    name: str = "Land Expansion & Weed Suppression"
    slot_index: int = 7
    role: str = "quadrant_expansion_and_tile_topology"
    mode: str = "deterministic"
    quadrant_sequence: List[str] = field(default_factory=lambda: ["NW", "NE", "SW", "SE"])
    expansion_costs: Dict[str, int] = field(default_factory=lambda: {"NE": 1000, "SW": 2000, "SE": 4000})

    def snapshot(self) -> Dict[str, Any]:
        return {
            "slot": self.slot_index,
            "name": self.name,
            "role": self.role,
            "mode": self.mode,
            "quadrant_sequence": list(self.quadrant_sequence),
            "expansion_costs": dict(self.expansion_costs),
            "weed_policy": "dig_immediately_on_spawn",
        }


# ── Slot 8: Labor & Fibonacci Scheduling ─────────────────────────────────

@dataclass
class LaborOptimizationSubagent:
    """Slot 8: Fibonacci wage schedule (1,1,2,3,5,8,...) and worker movement delegation."""

    name: str = "Labor Optimization & Scheduling"
    slot_index: int = 8
    role: str = "labor_allocation_and_fibonacci_costing"
    mode: str = "deterministic"

    def hire_cost(self, hires_today: int) -> int:
        from kaggriculture_adapter import hire_cost_today
        return hire_cost_today(hires_today)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "slot": self.slot_index,
            "name": self.name,
            "role": self.role,
            "mode": self.mode,
            "wage_model": "fibonacci_daily_reset",
            "delegation": "parallel_independent_tile_action",
        }


# ── Slot 9: Submission Packaging & Limits Gatekeeper ─────────────────────

@dataclass
class SubmissionPackagingSubagent:
    """Slot 9: Packaging gatekeeper, 90 MB hard-limit & 42 FLOPs/turn monitor."""

    name: str = "Submission Packaging Gatekeeper"
    slot_index: int = 9
    role: str = "hard_limits_compliance_and_packaging"
    mode: str = "deterministic"
    max_zip_mb: float = 90.0
    model_ceiling_mb: float = 100.0
    max_flops_per_turn: int = 42

    def snapshot(self) -> Dict[str, Any]:
        return {
            "slot": self.slot_index,
            "name": self.name,
            "role": self.role,
            "mode": self.mode,
            "max_zip_mb": self.max_zip_mb,
            "model_ceiling_mb": self.model_ceiling_mb,
            "max_flops_per_turn": self.max_flops_per_turn,
            "zip_window": "day_29_hours_0_to_4",
        }


# ── Subagent Registry & Queue Enforcer (Limit 10) ─────────────────────────

SUBAGENT_CLASSES: tuple[type, ...] = (
    EricSchmidtSubagent,           # Slot 0
    ElonMuskSubagent,              # Slot 1
    ScottWeedenSubagent,           # Slot 2
    AntigravitySubagent,           # Slot 3
    AgronomyYieldSubagent,         # Slot 4
    MarketLiquiditySubagent,       # Slot 5
    LivestockCareSubagent,         # Slot 6
    LandExpansionSubagent,         # Slot 7
    LaborOptimizationSubagent,     # Slot 8
    SubmissionPackagingSubagent,   # Slot 9
)


class SubagentQueue10:
    """Rigid 10-slot memory queue. Scott Weeden limits entire queue to 10."""

    def __init__(self) -> None:
        self.slots: List[Any] = [cls() for cls in SUBAGENT_CLASSES]

    def __len__(self) -> int:
        return len(self.slots)

    def get_slot(self, index: int) -> Any:
        if 0 <= index < len(self.slots):
            return self.slots[index]
        raise IndexError(f"Slot {index} out of bounds for 10-slot queue")

    def snapshot_all(self) -> List[Dict[str, Any]]:
        return [slot.snapshot() for slot in self.slots]


# ── Factory Helpers ───────────────────────────────────────────────────────

def assign_eric_schmidt(
    stake: int = SCHMIDT_STAKE_DEFAULT,
    *,
    bank: int = AGENT2_POST_CHARITY_BANK,
    trusted: bool = True,
) -> EricSchmidtSubagent:
    return EricSchmidtSubagent(stake=stake, bank=bank, trusted=trusted)


def assign_elon_musk(
    stake: int = ELON_MUSK_STAKE_DEFAULT,
    *,
    bank: int = AGENT2_POST_CHARITY_BANK,
) -> ElonMuskSubagent:
    return ElonMuskSubagent(stake=stake, bank=bank)


def assign_scott_weeden() -> ScottWeedenSubagent:
    return ScottWeedenSubagent()


def assign_antigravity() -> AntigravitySubagent:
    return AntigravitySubagent()


def assign_agronomy_yield() -> AgronomyYieldSubagent:
    return AgronomyYieldSubagent()


def assign_market_liquidity() -> MarketLiquiditySubagent:
    return MarketLiquiditySubagent()


def assign_livestock_care() -> LivestockCareSubagent:
    return LivestockCareSubagent()


def assign_land_expansion() -> LandExpansionSubagent:
    return LandExpansionSubagent()


def assign_labor_optimization() -> LaborOptimizationSubagent:
    return LaborOptimizationSubagent()


def assign_submission_packaging() -> SubmissionPackagingSubagent:
    return SubmissionPackagingSubagent()


def build_all_10_subagents() -> SubagentQueue10:
    """Build and return the complete, locked 10-slot subagent queue."""
    return SubagentQueue10()
