"""Named subagents for the Reasoning vs Questioning challenge.

Eric Schmidt — trusted probabilistic-model subagent (AUTHORS.md).
Stake magnitudes: 888 (default), 1667, or 2999 against post-charity bank 3888.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shared_state import (
    AGENT2_POST_CHARITY_BANK,
    SCHMIDT_STAKE_CHOICES,
    SCHMIDT_STAKE_DEFAULT,
    choose_schmidt_stake,
)


@dataclass
class EricSchmidtSubagent:
    """Probabilistic order-of-magnitude worker; not a deterministic rules authority."""

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
        """Accept a probabilistic order only within remaining stake budget."""
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
            "trusted": self.trusted,
            "stake": self.stake,
            "bank": self.bank,
            "remaining_bank": self.remaining_bank,
            "risk_tier": self.risk_tier,
            "stake_choices": list(SCHMIDT_STAKE_CHOICES),
            "queue_len": len(self.queue),
            "orders": list(self.queue),
        }


def assign_eric_schmidt(
    stake: int = SCHMIDT_STAKE_DEFAULT,
    *,
    bank: int = AGENT2_POST_CHARITY_BANK,
    trusted: bool = True,
) -> EricSchmidtSubagent:
    """Factory: Cursor default is stake=888 (not 2999 identity gamble)."""
    return EricSchmidtSubagent(stake=stake, bank=bank, trusted=trusted)
