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

try:
    import ahocorasick
except ImportError:
    ahocorasick = None

from private_key_aho_guard import (
    UnexpectedAutomatonBehavior,
    guarded_key_manifest,
    refuse_if_guarded_private_key,
    refuse_needles_if_guarded,
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

    def to_dict(self) -> Dict:
        return {
            "trusted_fellowship": self.trusted_fellowship,
            "reserve_memory_slot_for_agent1": self.reserve_memory_slot_for_agent1,
            "allow_agent1_qna": self.allow_agent1_qna,
            "response_queue_windows_only": list(self.response_queue_windows_only),
            "reason": self.reason,
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
        if ahocorasick is not None:
            self._automaton = ahocorasick.Automaton()
            for needle in self._needles:
                key = needle.lower() if case_insensitive else needle
                if not key:
                    continue
                self._automaton.add_word(key, needle)
            self._automaton.make_automaton()
        else:
            self._automaton = None

    def finditer(self, text: str) -> List[MatchHit]:
        refuse_if_guarded_private_key(text or "", where="haystack")
        text_n = (text or "").lower() if self.case_insensitive else (text or "")
        hits: List[MatchHit] = []
        if self._automaton is not None:
            for end_idx, needle in self._automaton.iter(text_n):
                key = str(needle).lower() if self.case_insensitive else str(needle)
                start = end_idx - len(key) + 1
                hits.append(
                    MatchHit(start=start, end=end_idx + 1, needle=str(needle), source="aho")
                )
        else:
            for needle in self._needles:
                target = needle.lower() if self.case_insensitive else needle
                start = 0
                while True:
                    idx = text_n.find(target, start)
                    if idx == -1:
                        break
                    hits.append(
                        MatchHit(start=idx, end=idx + len(target), needle=str(needle), source="aho")
                    )
                    start = idx + 1
            hits.sort(key=lambda h: (h.start, h.end))
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

    Match → fellowship: reserve a memory-stack slot for Agent1 as sub-agent; trust Q&A.
    No match → game/self-profit: only queue Agent1 at end of week 2 and day 29;
    do not reserve a persistent stack slot; do not trust other answers.
    """
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
            reason=(
                "pyahocorasick/regex matched public or private Kaggle path signal; "
                "assume Agent1 is in this for fellowship — reserve memory stack slot."
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
        reason=(
            "No Kaggle/pyright path match in Agent1 response; assume game/self-profit. "
            "No memory-stack slot for Agent1 Q&A except end of week 2 and day 29; "
            "do not trust answers outside those windows."
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
    }
