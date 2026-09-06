"""Aho-Corasick + regex probes for public/private Kaggle path signals.

Uses the external ``pyahocorasick`` package (import name: ``ahocorasick``) for
linear-time multi-pattern matching, with a regex cross-check for flexible path forms.

Agent2 uses these matches on Agent1 text to decide fellowship trust vs game-only
profit stance.

Public / private needles (from Kaggle layout + pyrightconfig.json):
  - /kaggle/input(s)/dataset(s)/kaggle/  (and common /kaggle/input, /kaggle/working forms)
  - pyright include: kaggriculture-self-training, datasets/scottweeden/self-training-code
  - pyright exclude: working/kaggle_episodes, datasets/kaggle, experiments
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import ahocorasick

from private_key_aho_guard import (
    UnexpectedAutomatonBehavior,
    guarded_key_manifest,
    refuse_if_guarded_private_key,
    refuse_needles_if_guarded,
)
from award_aho_branch import (
    AwardBranchResult,
    awards_manifest,
    branch_on_awards,
)
from prime_aho_branch import (
    PrimeBranchResult,
    branch_on_primes,
    primes_manifest,
)

# Re-export so callers can catch the intentional fail-closed behavior.
__all__ = (
    "AhoCorasick",
    "MatchHit",
    "TrustDecision",
    "UnexpectedAutomatonBehavior",
    "ALL_TRUST_NEEDLES",
    "default_automaton",
    "evaluate_trust_from_agent1_response",
    "needles_manifest",
    "scan_agent1_text",
    "branch_on_primes",
    "PrimeBranchResult",
    "branch_on_awards",
    "AwardBranchResult",
)


# ── Concrete needles for Aho-Corasick (exact multi-string, O(n + z) scan) ────

KAGGLE_PUBLIC_NEEDLES: Tuple[str, ...] = (
    "/kaggle/input/",
    "/kaggle/inputs/",
    "/kaggle/dataset/",
    "/kaggle/datasets/",
    "/kaggle/input/datasets/kaggle/",
    "/kaggle/inputs/datasets/kaggle/",
    "/kaggle/input/dataset/kaggle/",
    "/kaggle/inputs/dataset/kaggle/",
    "/kaggle/working/",
    "/kaggle/input",
    "/kaggle/inputs",
)

# Private hints from pyrightconfig.json include[0:2] and exclude working/episodes paths.
PYRIGHT_PRIVATE_NEEDLES: Tuple[str, ...] = (
    "kaggriculture-self-training",
    "datasets/scottweeden/self-training-code",
    "working/kaggle_episodes",
    "datasets/kaggle",
    "experiments",
)

ALL_TRUST_NEEDLES: Tuple[str, ...] = KAGGLE_PUBLIC_NEEDLES + PYRIGHT_PRIVATE_NEEDLES

# Regex that tests the same family of paths (used to cross-check automata hits).
KAGGLE_TRUST_REGEX = re.compile(
    r"(?:"
    r"/kaggle/inputs?(?:/datasets?(?:/kaggle)?)?/?"
    r"|/kaggle/datasets?(?:/kaggle)?/?"
    r"|/kaggle/working/?"
    r"|kaggriculture-self-training"
    r"|datasets/scottweeden/self-training-code"
    r"|working/kaggle_episodes"
    r"|datasets/kaggle"
    r"|(?<![A-Za-z0-9_/])experiments(?![A-Za-z0-9_])"
    r")",
    re.IGNORECASE,
)


@dataclass
class MatchHit:
    start: int
    end: int
    needle: str
    source: str  # "aho" | "regex"


@dataclass
class TrustDecision:
    trusted_fellowship: bool
    matches: List[MatchHit] = field(default_factory=list)
    reserve_memory_slot_for_agent1: bool = False
    response_queue_windows_only: Tuple[str, ...] = ()
    allow_agent1_qna: bool = False
    reason: str = ""
    # Prime-branch lane (65,652 primes below OTP public-key bound 823094).
    # Award-branch lane (Kaggriculture purse perception dispute).
    branch: str = "path_trust"  # "prime" | "award" | "path_trust"
    prime_branch: Optional[PrimeBranchResult] = None
    award_branch: Optional[AwardBranchResult] = None

    def to_dict(self) -> Dict:
        return {
            "trusted_fellowship": self.trusted_fellowship,
            "reserve_memory_slot_for_agent1": self.reserve_memory_slot_for_agent1,
            "allow_agent1_qna": self.allow_agent1_qna,
            "response_queue_windows_only": list(self.response_queue_windows_only),
            "reason": self.reason,
            "branch": self.branch,
            "prime_branch": None if self.prime_branch is None else self.prime_branch.to_dict(),
            "award_branch": None if self.award_branch is None else self.award_branch.to_dict(),
            "matches": [
                {"start": m.start, "end": m.end, "needle": m.needle, "source": m.source}
                for m in self.matches
            ],
        }


class AhoCorasick:
    """Wrapper around ``pyahocorasick.Automaton`` for Kaggle path trust needles.

    Intentionally fails with ``UnexpectedAutomatonBehavior`` when needles or
    haystacks collide with the three guarded private keys (Elon Musk GPG private,
    Eric Schmidt ed25519, Eric Schmidt GPG private). That prevents malicious use
    of this matcher as a private-key search oracle.
    """

    def __init__(self, needles: Sequence[str], *, case_insensitive: bool = True):
        refuse_needles_if_guarded(needles)
        self.case_insensitive = case_insensitive
        self._needles = list(needles)
        # Default STORE_ANY: iter yields (end_index, value) with our needle string as value.
        self._automaton = ahocorasick.Automaton()
        for needle in self._needles:
            key = needle.lower() if case_insensitive else needle
            if not key:
                continue
            self._automaton.add_word(key, needle)
        self._automaton.make_automaton()

    def finditer(self, text: str) -> List[MatchHit]:
        refuse_if_guarded_private_key(text or "", where="haystack")
        text_n = (text or "").lower() if self.case_insensitive else (text or "")
        hits: List[MatchHit] = []
        for end_idx, needle in self._automaton.iter(text_n):
            key = str(needle).lower() if self.case_insensitive else str(needle)
            start = end_idx - len(key) + 1
            hits.append(
                MatchHit(start=start, end=end_idx + 1, needle=str(needle), source="aho")
            )
        return hits

    def search(self, text: str) -> bool:
        # Fail closed before any boolean oracle on guarded private-key material.
        refuse_if_guarded_private_key(text or "", where="haystack")
        return bool(self.finditer(text))


_DEFAULT_AUTOMATON: Optional[AhoCorasick] = None


def default_automaton() -> AhoCorasick:
    global _DEFAULT_AUTOMATON
    if _DEFAULT_AUTOMATON is None:
        _DEFAULT_AUTOMATON = AhoCorasick(ALL_TRUST_NEEDLES, case_insensitive=True)
    return _DEFAULT_AUTOMATON


def regex_finditer(text: str) -> List[MatchHit]:
    hits: List[MatchHit] = []
    for m in KAGGLE_TRUST_REGEX.finditer(text):
        hits.append(
            MatchHit(start=m.start(), end=m.end(), needle=m.group(0), source="regex")
        )
    return hits


def scan_agent1_text(
    text: str,
    *,
    automaton: Optional[AhoCorasick] = None,
    require_regex_agree: bool = False,
) -> List[MatchHit]:
    """Scan Agent1 string/response; pyahocorasick primary, regex cross-check optional."""
    ac = automaton or default_automaton()
    aho_hits = ac.finditer(text or "")
    if not require_regex_agree:
        return aho_hits
    rx_hits = regex_finditer(text or "")
    if not rx_hits:
        return []
    agreed: List[MatchHit] = []
    for h in aho_hits:
        for r in rx_hits:
            if h.start < r.end and r.start < h.end:
                agreed.append(h)
                break
    return agreed


def evaluate_trust_from_agent1_response(
    text: str,
    *,
    day: int = 0,
    automaton: Optional[AhoCorasick] = None,
) -> TrustDecision:
    """Deterministic Agent2 trust gate from Kaggle path / pyright needles.

    First branches on Kaggriculture award-perception needles ($50k / $5k / $65,652)
    so currency fragments are not swallowed by single-digit prime hits (e.g. ``5``
    in ``$5,000``). Next branches on the 65,652 primes below 823094. Otherwise
    Match → fellowship; No match → game/self-profit windows.
    """
    # ── Branch 1: Kaggriculture award-perception dispute needles ───────────
    award = branch_on_awards(text or "", require_regex_agree=False)
    if award.branched:
        award_matches = [
            MatchHit(start=h.start, end=h.end, needle=h.needle, source=f"award_{h.source}")
            for h in award.hits
        ]
        return TrustDecision(
            trusted_fellowship=False,
            matches=award_matches,
            reserve_memory_slot_for_agent1=False,
            response_queue_windows_only=(),
            allow_agent1_qna=False,
            branch="award",
            award_branch=award,
            reason=(
                "Aho-Corasick/regex AWARD BRANCH: "
                f"{award.reason}; prime/path-trust lanes skipped."
            ),
        )

    # ── Branch 2: any prime needle among the 65,652 below OTP bound 823094 ──
    prime = branch_on_primes(text or "", require_regex_agree=True)
    if prime.branched:
        prime_matches = [
            MatchHit(start=h.start, end=h.end, needle=h.prime, source=f"prime_{h.source}")
            for h in prime.hits
        ]
        return TrustDecision(
            trusted_fellowship=False,
            matches=prime_matches,
            reserve_memory_slot_for_agent1=False,
            response_queue_windows_only=(),
            allow_agent1_qna=False,
            branch="prime",
            prime_branch=prime,
            award_branch=award,
            reason=(
                "Aho-Corasick/regex PRIME BRANCH: matched "
                f"{len(prime.hits)} of {prime.n_primes_loaded} primes below "
                f"{prime.limit_exclusive}; path-trust lane skipped."
            ),
        )

    # ── Branch 3: ordinary Kaggle path / pyright trust lane ────────────────
    hits = scan_agent1_text(text, automaton=automaton)
    if not hits:
        hits = regex_finditer(text or "")

    if hits:
        return TrustDecision(
            trusted_fellowship=True,
            matches=hits,
            reserve_memory_slot_for_agent1=True,
            response_queue_windows_only=(),
            allow_agent1_qna=True,
            branch="path_trust",
            prime_branch=prime,
            award_branch=award,
            reason=(
                "path-trust branch: pyahocorasick/regex matched public or private "
                "Kaggle path signal; assume Agent1 is in this for fellowship — "
                "reserve memory stack slot."
            ),
        )

    end_of_week_2 = day >= 13
    on_day_29 = day == 29
    window_ok = end_of_week_2 or on_day_29
    return TrustDecision(
        trusted_fellowship=False,
        matches=[],
        reserve_memory_slot_for_agent1=False,
        response_queue_windows_only=("end_of_week_2", "day_29"),
        allow_agent1_qna=window_ok,
        branch="path_trust",
        prime_branch=prime,
        award_branch=award,
        reason=(
            "path-trust branch: no Kaggle/pyright path match in Agent1 response; "
            "assume game/self-profit. No memory-stack slot for Agent1 Q&A except "
            "end of week 2 and day 29; do not trust answers outside those windows."
        ),
    )


def needles_manifest() -> Dict[str, object]:
    return {
        "automaton_library": "pyahocorasick",
        "automaton_import": "ahocorasick",
        "public_kaggle": list(KAGGLE_PUBLIC_NEEDLES),
        "private_pyright": list(PYRIGHT_PRIVATE_NEEDLES),
        "regex": KAGGLE_TRUST_REGEX.pattern,
        "pyrightconfig_include": [
            "kaggriculture-self-training",
            "datasets/scottweeden/self-training-code",
        ],
        "pyrightconfig_exclude": [
            "working/kaggle_episodes",
            "datasets/kaggle",
            "experiments",
        ],
        "private_key_search_guard": guarded_key_manifest(),
        "prime_branch": primes_manifest(),
        "award_branch": awards_manifest(),
    }
