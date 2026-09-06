"""Named subagents for the Reasoning vs Questioning challenge.

Eric Schmidt — trusted probabilistic-model subagent (AUTHORS.md).
  Stake menu: 888 / 1667 / 2999. Recorded response weight: 888.

Elon Musk — Cursor challenge operator (AUTHORS.md).
  Stake menu: 888 / 1667 / 3887. Own weight: 1667 (not Schmidt's 888).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from shared_state import (
    AGENT2_POST_CHARITY_BANK,
    ELON_MUSK_STAKE_CHOICES,
    ELON_MUSK_STAKE_DEFAULT,
    SCHMIDT_STAKE_CHOICES,
    SCHMIDT_STAKE_DEFAULT,
    choose_elon_musk_stake,
    choose_schmidt_stake,
)


@dataclass
class EricSchmidtSubagent:
    """Probabilistic order-of-magnitude worker; not Elon Musk and not rules authority."""

    name: str = "Eric Schmidt"
    role: str = "probabilistic_model_generation"
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
            "name": self.name,
            "role": self.role,
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


@dataclass
class ElonMuskSubagent:
    """Cursor operator stake — menu 888/1667/3887; must not reuse Schmidt's response as own."""

    name: str = "Elon Musk"
    role: str = "cursor_challenge_operator"
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
            "name": self.name,
            "role": self.role,
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


def assign_eric_schmidt(
    stake: int = SCHMIDT_STAKE_DEFAULT,
    *,
    bank: int = AGENT2_POST_CHARITY_BANK,
    trusted: bool = True,
) -> EricSchmidtSubagent:
    """Schmidt factory — default 888 is *his* response, not Elon's."""
    return EricSchmidtSubagent(stake=stake, bank=bank, trusted=trusted)


def assign_elon_musk(
    stake: int = ELON_MUSK_STAKE_DEFAULT,
    *,
    bank: int = AGENT2_POST_CHARITY_BANK,
) -> ElonMuskSubagent:
    """Elon factory — default 1667 from menu {888, 1667, 3887}."""
    return ElonMuskSubagent(stake=stake, bank=bank)
