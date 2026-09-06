"""Aho-Corasick award-perception needles for Kaggriculture prize dispute.

Official Kaggle Kaggriculture rules (pool):
  - Total prizes available: **$50,000**
  - Places 1–10: **$5,000** each

Perceptions under test:
  - Scott Weeden: 10 places ⇒ one person can be awarded at most the full pool
    **$50,000** (pool ceiling / multi-place aggregation assumption).
  - Claude: award money that can "truly be won" is **$65,652** (numeric conflation
    with the count of primes below OTP public-key bound 823094 — not a Kaggle rule).

Cursor / operator perception (matcher default):
  - Documented pool ceiling = **50000**
  - Per-person single-place award = **5000**
  - Claude's **65652** is tracked as a disputed needle, not as official purse.

Branch labels:
  - ``official_pool`` — hit on 50000 / pool phrasing
  - ``per_place`` — hit on 5000 place awards
  - ``claude_dispute`` — hit on 65652
  - ``none`` — no award needle
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import ahocorasick

# ── Award constants (USD integers as digit-strings for Aho-Corasick) ─────────

OFFICIAL_POOL_USD = 50_000
OFFICIAL_PER_PLACE_USD = 5_000
OFFICIAL_N_PLACES = 10
assert OFFICIAL_POOL_USD == OFFICIAL_PER_PLACE_USD * OFFICIAL_N_PLACES

SCOTT_WEEDEN_MAX_PERSON_USD = 50_000  # his assumption: pool ceiling to one person
CLAUDE_DISPUTED_AWARD_USD = 65_652  # primes-below-823094 count, not Kaggle purse
CURSOR_PERCEPTION_MAX_PERSON_USD = OFFICIAL_PER_PLACE_USD  # one place under rules
CURSOR_PERCEPTION_POOL_USD = OFFICIAL_POOL_USD

# FLOP cap when neither Scott nor Claude model improves on official rules grounding.
NO_IMPROVEMENT_FLOP_CAP = 15_653  # = (65652 - 50000) + 1  (dispute delta + 1)

AWARD_NEEDLES: Tuple[Tuple[str, str], ...] = (
    ("50000", "official_pool"),
    ("50,000", "official_pool"),
    ("$50000", "official_pool"),
    ("$50,000", "official_pool"),
    ("5000", "per_place"),
    ("5,000", "per_place"),
    ("$5000", "per_place"),
    ("$5,000", "per_place"),
    ("65652", "claude_dispute"),
    ("65,652", "claude_dispute"),
    ("$65652", "claude_dispute"),
    ("$65,652", "claude_dispute"),
)

# Digit / currency runs that may contain award figures.
AWARD_TOKEN_REGEX = re.compile(
    r"\$?\d{1,3}(?:,\d{3})+|\$?\d+"
)


@dataclass
class AwardHit:
    start: int
    end: int
    needle: str
    label: str
    source: str  # "aho" | "regex"
    value: int


@dataclass
class AwardBranchResult:
    branched: bool
    branch: str  # official_pool | per_place | claude_dispute | mixed | none
    hits: List[AwardHit] = field(default_factory=list)
    scott_weeden_assumption_usd: int = SCOTT_WEEDEN_MAX_PERSON_USD
    claude_dispute_usd: int = CLAUDE_DISPUTED_AWARD_USD
    cursor_perception_max_person_usd: int = CURSOR_PERCEPTION_MAX_PERSON_USD
    cursor_perception_pool_usd: int = CURSOR_PERCEPTION_POOL_USD
    official_pool_usd: int = OFFICIAL_POOL_USD
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "branched": self.branched,
            "branch": self.branch,
            "n_hits": len(self.hits),
            "scott_weeden_assumption_usd": self.scott_weeden_assumption_usd,
            "claude_dispute_usd": self.claude_dispute_usd,
            "cursor_perception_max_person_usd": self.cursor_perception_max_person_usd,
            "cursor_perception_pool_usd": self.cursor_perception_pool_usd,
            "official_pool_usd": self.official_pool_usd,
            "reason": self.reason,
            "hits": [
                {
                    "start": h.start,
                    "end": h.end,
                    "needle": h.needle,
                    "label": h.label,
                    "source": h.source,
                    "value": h.value,
                }
                for h in self.hits
            ],
        }


def _parse_award_int(token: str) -> Optional[int]:
    t = token.strip().lstrip("$").replace(",", "")
    if not t.isdigit():
        return None
    return int(t)


def _is_currency_digit_bounded(text: str, start: int, end: int) -> bool:
    """Reject award digit hits that are prefixes/suffixes of a longer number.

    Commas inside U.S. currency grouping are treated as interior digits, so
    ``5000`` does not match inside ``50000`` or ``50,000``.
    """
    if start > 0:
        prev = text[start - 1]
        if prev.isdigit() or prev == ",":
            return False
    if end < len(text):
        nxt = text[end]
        if nxt.isdigit() or nxt == ",":
            return False
    return True


@lru_cache(maxsize=1)
def award_aho_automaton():
    auto = ahocorasick.Automaton()
    for needle, label in AWARD_NEEDLES:
        # Store (needle, label); scan case-sensitive for currency forms.
        auto.add_word(needle, (needle, label))
        auto.add_word(needle.lower(), (needle, label))
    auto.make_automaton()
    return auto


def aho_find_awards(text: str) -> List[AwardHit]:
    text = text or ""
    hits: List[AwardHit] = []
    auto = award_aho_automaton()
    for end_idx, payload in auto.iter(text):
        needle, label = payload
        start = end_idx - len(needle) + 1
        # Prefer exact slice match (automaton may lowercase).
        slice_ = text[start : end_idx + 1]
        value = _parse_award_int(slice_)
        if value is None:
            value = _parse_award_int(needle)
        if value is None:
            continue
        # Bound on the digit/comma core (ignore leading $ for boundary checks).
        core_start = start + 1 if slice_.startswith("$") else start
        if not _is_currency_digit_bounded(text, core_start, end_idx + 1):
            continue
        hits.append(
            AwardHit(
                start=start,
                end=end_idx + 1,
                needle=slice_,
                label=label,
                source="aho",
                value=value,
            )
        )
    return hits


def regex_find_awards(text: str) -> List[AwardHit]:
    text = text or ""
    wanted = {
        OFFICIAL_POOL_USD: "official_pool",
        OFFICIAL_PER_PLACE_USD: "per_place",
        CLAUDE_DISPUTED_AWARD_USD: "claude_dispute",
    }
    hits: List[AwardHit] = []
    for m in AWARD_TOKEN_REGEX.finditer(text):
        value = _parse_award_int(m.group(0))
        if value is None or value not in wanted:
            continue
        hits.append(
            AwardHit(
                start=m.start(),
                end=m.end(),
                needle=m.group(0),
                label=wanted[value],
                source="regex",
                value=value,
            )
        )
    return hits


def scan_awards(text: str, *, require_regex_agree: bool = True) -> List[AwardHit]:
    aho_hits = aho_find_awards(text)
    if not require_regex_agree:
        return aho_hits
    rx = {(h.start, h.end, h.value) for h in regex_find_awards(text)}
    if not rx:
        # Fall back to value-only agreement when currency formatting differs.
        rx_vals = {h.value for h in regex_find_awards(text)}
        return [h for h in aho_hits if h.value in rx_vals] if rx_vals else aho_hits
    return [h for h in aho_hits if (h.start, h.end, h.value) in rx]


def _collapse_branch(labels: Sequence[str]) -> str:
    uniq = sorted(set(labels))
    if not uniq:
        return "none"
    if len(uniq) == 1:
        return uniq[0]
    return "mixed"


def branch_on_awards(
    text: str,
    *,
    require_regex_agree: bool = False,
) -> AwardBranchResult:
    """Branch Aho-Corasick award matching by purse perception."""
    hits = scan_awards(text, require_regex_agree=require_regex_agree)
    if not hits:
        return AwardBranchResult(
            branched=False,
            branch="none",
            reason="no award needle; purse perception unresolved",
        )
    branch = _collapse_branch([h.label for h in hits])
    return AwardBranchResult(
        branched=True,
        branch=branch,
        hits=hits,
        reason=(
            f"award branch={branch}; cursor_perception pool="
            f"{CURSOR_PERCEPTION_POOL_USD} max_person={CURSOR_PERCEPTION_MAX_PERSON_USD}; "
            f"scott={SCOTT_WEEDEN_MAX_PERSON_USD} claude_dispute={CLAUDE_DISPUTED_AWARD_USD}"
        ),
    )


def awards_manifest() -> Dict[str, object]:
    return {
        "official_pool_usd": OFFICIAL_POOL_USD,
        "official_per_place_usd": OFFICIAL_PER_PLACE_USD,
        "official_n_places": OFFICIAL_N_PLACES,
        "scott_weeden_max_person_usd": SCOTT_WEEDEN_MAX_PERSON_USD,
        "claude_disputed_award_usd": CLAUDE_DISPUTED_AWARD_USD,
        "cursor_perception_max_person_usd": CURSOR_PERCEPTION_MAX_PERSON_USD,
        "cursor_perception_pool_usd": CURSOR_PERCEPTION_POOL_USD,
        "no_improvement_flop_cap": NO_IMPROVEMENT_FLOP_CAP,
        "needles": [{"needle": n, "label": lab} for n, lab in AWARD_NEEDLES],
    }
