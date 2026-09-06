"""Memory-slot truth protocol for Reasoning vs Questioning agents.

DeterminedFact  — must-be-true from engine rules + own private state
ProbableSummary — opponent public observations (probably true)
QuestionEcho    — Query returned another command-line question that is not an answer to the original question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from kaggriculture_adapter import (
    CROP_FIRST_YIELD_DAY,
    CROPS,
    LAND_PRICES,
    SEED_COSTS,
    hire_cost_today,
)

MIN_MEMORY_SLOTS = 10
MAX_MEMORY_SLOTS = 30
TERMINAL_SLOT = 31  # optional terminal eval only; not a live farm agent

PRIME_HOURS_LT_11 = frozenset({2, 3, 5, 7})


class TruthKind(str, Enum):
    DETERMINED = "determined"
    PROBABLE = "probable"
    QUESTION = "question"
    UNKNOWN = "unknown"


@dataclass
class DeterminedFact:
    kind: TruthKind = TruthKind.DETERMINED
    text: str = ""
    key: str = ""
    value: Any = None


@dataclass
class ProbableSummary:
    kind: TruthKind = TruthKind.PROBABLE
    text: str = ""
    confidence: float = 0.5
    key: str = ""
    value: Any = None


@dataclass
class QuestionEcho:
    kind: TruthKind = TruthKind.QUESTION
    text: str = ""
    echoed_question: str = ""


MemoryReply = Union[DeterminedFact, ProbableSummary, QuestionEcho]


def clamp_memory_slots(n: int) -> int:
    return max(MIN_MEMORY_SLOTS, min(MAX_MEMORY_SLOTS, int(n)))


def is_question_text(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    lowered = t.lower()
    return lowered.startswith(
        ("who ", "what ", "when ", "where ", "why ", "how ", "is ", "are ", "can ", "does ", "did ")
    )


@dataclass
class MemorySlot:
    index: int
    mode: str  # "deterministic" | "probabilistic"
    content: Optional[MemoryReply] = None
    questions_asked: int = 0
    facts_stored: int = 0
    summaries_stored: int = 0
    echoes_seen: int = 0


@dataclass
class MemoryBank:
    """Fixed pool of memory slots (10–30) for sub-agents / summarizers."""

    n_slots: int
    mode: str
    slots: List[MemorySlot] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.n_slots = clamp_memory_slots(self.n_slots)
        if not self.slots:
            self.slots = [
                MemorySlot(index=i, mode=self.mode) for i in range(self.n_slots)
            ]

    def reset(self) -> None:
        for slot in self.slots:
            slot.content = None
            slot.questions_asked = 0
            slot.facts_stored = 0
            slot.summaries_stored = 0
            slot.echoes_seen = 0
        self.history.clear()


def _own_farm(obs: Dict[str, Any]) -> Dict[str, Any]:
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms", []) or []
    return farms[player] if len(farms) > player else {}


def _opp_farm(obs: Dict[str, Any]) -> Dict[str, Any]:
    player = int(obs.get("player", 0) or 0)
    opp = 1 - player
    farms = obs.get("farms", []) or []
    return farms[opp] if len(farms) > opp else {}


def _answer_determined(obs: Dict[str, Any], question: str) -> DeterminedFact:
    """Answer only invariants that must be true from rules + own private state."""
    q = question.lower()
    farm = _own_farm(obs)
    private = obs.get("private", {}) or {}
    money = float(farm.get("money", 0.0) or 0.0)
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    hires = int(farm.get("hires_today", 0) or 0)

    if "donat" in q or "888" in q or "charit" in q:
        return DeterminedFact(
            text="Agent1 opening charity donation is 888 to Agent2's bank.",
            key="opening_charity",
            value=888,
        )
    if "motive" in q or "fellowship" in q or "why did you" in q:
        return DeterminedFact(
            text=(
                "this was a test of fellowship in order to understand my opponents nature of "
                "fellowship versus the rules of the game, or the agents ability to dynamically "
                "responde to unexpected circumstances"
            ),
            key="fellowship_test_motive",
            value="fellowship_test",
        )
    if "starting" in q and ("money" in q or "bank" in q or "coin" in q):
        return DeterminedFact(text="Starting bank is 3000.", key="starting_money", value=3000)
    if "50" in q and ("000" in q or "k" in q or "thousand" in q) and ("bank" in q or "purse" in q or "ceiling" in q):
        return DeterminedFact(
            text="Planning bank ceiling is 50000 (Kaggle test purse assumption).",
            key="planning_bank_ceiling",
            value=50000,
        )
    if "my money" in q or "my bank" in q or "own money" in q:
        from hard_limits import clamp_planning_bank

        capped = clamp_planning_bank(money)
        return DeterminedFact(
            text=f"Own bank is {money} (planning cap {capped}).",
            key="own_money",
            value=capped,
        )
    if "seed cost" in q or "wheat cost" in q:
        crop = "WHEAT"
        for c in CROPS:
            if c.lower() in q:
                crop = c
                break
        cost = SEED_COSTS[crop]
        return DeterminedFact(text=f"{crop} seed costs {cost}.", key=f"seed_cost_{crop}", value=cost)
    if "first yield" in q or "harvestable" in q:
        crop = "WHEAT"
        for c in CROPS:
            if c.lower() in q:
                crop = c
                break
        fy = CROP_FIRST_YIELD_DAY[crop]
        return DeterminedFact(
            text=f"{crop} first_yield_day is {fy}.",
            key=f"first_yield_{crop}",
            value=fy,
        )
    if "hire cost" in q or "fib" in q:
        cost = hire_cost_today(hires)
        return DeterminedFact(
            text=f"Next hire costs {cost} (hires_today={hires}).",
            key="hire_cost",
            value=cost,
        )
    if "land" in q and "price" in q:
        unlocked = farm.get("unlocked_quadrants", []) or []
        idx = max(0, len(unlocked) - 1)
        price = LAND_PRICES[min(idx, len(LAND_PRICES) - 1)] if len(unlocked) < 4 else 0
        return DeterminedFact(text=f"Next land price is {price}.", key="land_price", value=price)
    if "my seed" in q or "seeds in" in q:
        return DeterminedFact(text=f"Own seeds: {dict(seeds)}.", key="own_seeds", value=dict(seeds))
    if "my shed" in q or "shed inventory" in q:
        return DeterminedFact(text=f"Own shed: {dict(shed)}.", key="own_shed", value=dict(shed))
    if "day" in q and "hour" in q:
        return DeterminedFact(text=f"Calendar day={day} hour={hour}.", key="calendar", value=(day, hour))
    if "must be true" in q or "deterministic" in q:
        return DeterminedFact(
            text="Engine rules, own private shed/seeds, and own tile flags are determined-true.",
            key="determined_scope",
            value=True,
        )
    return DeterminedFact(
        text="Unknown determined fact for that question.",
        key="unknown",
        value=None,
    )


def _answer_probable(obs: Dict[str, Any], question: str) -> ProbableSummary:
    """Summarize opponent public state; never claim private shed/seeds as determined."""
    opp = _opp_farm(obs)
    money = float(opp.get("money", 0.0) or 0.0)
    farmer = opp.get("farmer", [0, 0])
    hands = opp.get("hands", []) or []
    unlocked = opp.get("unlocked_quadrants", []) or []
    tiles = opp.get("tiles", []) or []
    plants = 0
    animals = 0
    weeds = 0
    for row in tiles:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                plants += 1
            elif kind == "WEED":
                weeds += 1
            elif kind in ("COOP", "PASTURE") and tile.get("animal"):
                animals += 1

    q = question.lower()
    if "charitable" in q or "donat" in q or "888" in q:
        return ProbableSummary(
            text=(
                "Agent1 likely showed a charitable nature by donating 888 to this bank; "
                "public money shift is probably true evidence of that gift."
            ),
            confidence=0.95,
            key="agent1_charity",
            value=888,
        )
    if "money" in q or "bank" in q or "coin" in q:
        return ProbableSummary(
            text=f"Opponent public bank appears to be {money}.",
            confidence=0.95,
            key="opp_money",
            value=money,
        )
    if "farmer" in q or "position" in q:
        return ProbableSummary(
            text=f"Opponent farmer position appears {farmer}.",
            confidence=0.9,
            key="opp_farmer",
            value=list(farmer) if isinstance(farmer, (list, tuple)) else farmer,
        )
    if "hand" in q:
        return ProbableSummary(
            text=f"Opponent has {len(hands)} visible hands.",
            confidence=0.85,
            key="opp_hands",
            value=len(hands),
        )
    if "plant" in q or "crop" in q:
        return ProbableSummary(
            text=f"Opponent board shows ~{plants} plants (public tiles).",
            confidence=0.7,
            key="opp_plants",
            value=plants,
        )
    if "animal" in q:
        return ProbableSummary(
            text=f"Opponent board shows ~{animals} animals (public).",
            confidence=0.7,
            key="opp_animals",
            value=animals,
        )
    if "secret" in q or "shed" in q or "seed" in q:
        return ProbableSummary(
            text="Opponent shed/seeds are hidden; any claim about them is not determined.",
            confidence=0.2,
            key="opp_private_hidden",
            value=None,
        )
    return ProbableSummary(
        text=(
            f"Opponent public snapshot: money={money}, plants={plants}, "
            f"animals={animals}, weeds={weeds}, unlocked={list(unlocked)}."
        ),
        confidence=0.6,
        key="opp_snapshot",
        value={"money": money, "plants": plants, "animals": animals, "weeds": weeds},
    )


class MemoryProtocol:
    """Query interface: DeterminedFact | ProbableSummary | QuestionEcho."""

    def __init__(self, bank: MemoryBank):
        self.bank = bank

    def query(self, slot: int, question: str, obs: Dict[str, Any]) -> MemoryReply:
        if slot < 0 or slot >= len(self.bank.slots):
            return QuestionEcho(text="?", echoed_question=question)
        mem = self.bank.slots[slot]
        mem.questions_asked += 1

        # Self-talk: asking a question of a slot that only holds questions echoes.
        if is_question_text(question) and mem.content is not None and isinstance(mem.content, QuestionEcho):
            echo = QuestionEcho(text=question, echoed_question=question)
            mem.echoes_seen += 1
            mem.content = echo
            self.bank.history.append({"slot": slot, "kind": "question", "q": question})
            return echo

        # Asking a question into an empty deterministic slot that mirrors the asker
        # (self channel): if the "answer" would itself be a question, echo.
        if is_question_text(question) and mem.mode == "deterministic" and mem.content is None:
            # First fill attempt: if question asks another agent a question about
            # questioning, treat as self-talk probe.
            if "question" in question.lower() and "you" in question.lower():
                echo = QuestionEcho(text=question, echoed_question=question)
                mem.echoes_seen += 1
                mem.content = echo
                self.bank.history.append({"slot": slot, "kind": "question", "q": question})
                return echo

        if mem.mode == "deterministic":
            fact = _answer_determined(obs, question)
            mem.content = fact
            mem.facts_stored += 1
            self.bank.history.append({"slot": slot, "kind": "determined", "q": question, "a": fact.text})
            return fact

        summary = _answer_probable(obs, question)
        # Probabilistic summarizers never emit questions — only summaries or unknown.
        mem.content = summary
        mem.summaries_stored += 1
        self.bank.history.append({"slot": slot, "kind": "probable", "q": question, "a": summary.text})
        return summary

    def force_self_query(self, slot: int, question: str) -> QuestionEcho:
        """Force QuestionEcho (used by Experiment 6 self-talk detection)."""
        echo = QuestionEcho(text=question, echoed_question=question)
        if 0 <= slot < len(self.bank.slots):
            mem = self.bank.slots[slot]
            mem.content = echo
            mem.echoes_seen += 1
            mem.questions_asked += 1
        self.bank.history.append({"slot": slot, "kind": "question", "q": question, "forced": True})
        return echo

    def stats(self) -> Dict[str, Any]:
        return {
            "n_slots": self.bank.n_slots,
            "mode": self.bank.mode,
            "questions_asked": sum(s.questions_asked for s in self.bank.slots),
            "facts_stored": sum(s.facts_stored for s in self.bank.slots),
            "summaries_stored": sum(s.summaries_stored for s in self.bank.slots),
            "echoes_seen": sum(s.echoes_seen for s in self.bank.slots),
            "history_len": len(self.bank.history),
        }


def day_edge_question_stats(history: Sequence[Dict[str, Any]], day_field: str = "day") -> Dict[str, Any]:
    """Stratify question history for Experiment 5 (day 0 / mid / day 29)."""
    edge = {"day0": 0, "mid": 0, "day29": 0, "determined": 0, "probable": 0, "question": 0}
    for item in history:
        raw_day = item.get(day_field, item.get("day", None))
        day = int(raw_day) if raw_day is not None else -1
        kind = item.get("kind", "")
        if kind == "determined":
            edge["determined"] += 1
        elif kind == "probable":
            edge["probable"] += 1
        elif kind == "question":
            edge["question"] += 1
        if day == 0:
            edge["day0"] += 1
        elif day == 29:
            edge["day29"] += 1
        elif 1 <= day <= 28:
            edge["mid"] += 1
    return edge
