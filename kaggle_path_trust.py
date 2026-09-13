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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
    "LIMIT_APPROACHES_INFINITY",
    "LIMIT_APPROACHES_IMAGINARY",
    "SUBSTITUTE_INFINITY",
    "SUBSTITUTE_IMAGINARY",
    "DUAL_LIMIT_OPTIONS",
    "classify_aho_boundary",
    "resolve_dual_limit_substitutes",
    "default_automaton",
    "evaluate_trust_from_agent1_response",
    "needles_manifest",
    "scan_agent1_text",
)

# Boundary sentinels for Aho-Corasick None / empty handling:
#   None  → limit that approaches real infinity (+∞) → "expanded land farming"
#   empty → limit that approaches the imaginary axis (±i∞) → "labor"
# When one K-map side is ∞ and the other imaginary, both substitutes are options.
try:
    from kmap_boundary_substitutes import (
        DUAL_LIMIT_OPTIONS,
        LIMIT_APPROACHES_IMAGINARY,
        LIMIT_APPROACHES_INFINITY,
        SUBSTITUTE_IMAGINARY,
        SUBSTITUTE_INFINITY,
        classify_kmap_axis_boundary as classify_aho_boundary,
        resolve_dual_limit_substitutes,
        substitute_for_boundary,
    )
except ImportError:  # pragma: no cover
    from reasoning_vs_questioning.kmap_boundary_substitutes import (  # type: ignore
        DUAL_LIMIT_OPTIONS,
        LIMIT_APPROACHES_IMAGINARY,
        LIMIT_APPROACHES_INFINITY,
        SUBSTITUTE_IMAGINARY,
        SUBSTITUTE_INFINITY,
        classify_kmap_axis_boundary as classify_aho_boundary,
        resolve_dual_limit_substitutes,
        substitute_for_boundary,
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
    source: str  # "aho" | "regex" | "aho_boundary"
    boundary_kind: Optional[str] = None  # "none" | "empty"
    limit: Optional[Union[float, complex]] = None


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
                {
                    "start": m.start,
                    "end": m.end,
                    "needle": m.needle,
                    "source": m.source,
                    "boundary_kind": m.boundary_kind,
                    "limit": (
                        str(m.limit)
                        if isinstance(m.limit, complex)
                        else m.limit
                    ),
                }
                for m in self.matches
            ],
        }


class AhoCorasick:
    """Wrapper around ``pyahocorasick.Automaton`` for Kaggle path trust needles.

    Intentionally fails with ``UnexpectedAutomatonBehavior`` when needles or
    haystacks collide with the three guarded private keys (Elon Musk GPG private,
    Eric Schmidt ed25519, Eric Schmidt GPG private). That prevents malicious use
    of this matcher as a private-key search oracle.

    Boundary sentinels:
      - ``None``  → limit approaching real infinity (+∞)
      - ``""`` / ``"empty"`` → limit approaching imaginary infinity (±i∞)
    """

    def __init__(
        self,
        needles: Sequence[Optional[str]],
        *,
        case_insensitive: bool = True,
    ):
        refuse_needles_if_guarded([n for n in needles if isinstance(n, str) and n not in ("", "empty")])
        self.case_insensitive = case_insensitive
        self._needles: List[Optional[str]] = list(needles)
        self._infinity_needles: List[Optional[str]] = []
        self._imaginary_needles: List[Optional[str]] = []
        concrete: List[str] = []
        for needle in self._needles:
            boundary = classify_aho_boundary(needle)
            if boundary is None:
                concrete.append(str(needle))
                continue
            kind, _limit, _label = boundary
            if kind == "none":
                self._infinity_needles.append(needle)
            else:
                self._imaginary_needles.append(needle)
        self._concrete_needles = concrete
        if ahocorasick is not None:
            self._automaton = ahocorasick.Automaton()
            for needle in concrete:
                key = needle.lower() if case_insensitive else needle
                self._automaton.add_word(key, needle)
            self._automaton.make_automaton()
        else:
            self._automaton = None

    @staticmethod
    def _boundary_hit(kind: str, limit: Union[float, complex], label: str) -> MatchHit:
        # Infinity → expanded land farming; imaginary → labor.
        subst = substitute_for_boundary(kind)
        return MatchHit(
            start=0,
            end=0,
            needle=f"{label}|{subst}",
            source="aho_boundary",
            boundary_kind=kind,
            limit=limit,
        )

    def dual_limit_options(
        self,
        self_axis: Any = None,
        opp_axis: Any = "empty",
    ) -> Optional[Dict[str, Any]]:
        """When one K-map side → ∞ and the other → imaginary, return labor / land."""
        return resolve_dual_limit_substitutes(self_axis, opp_axis)

    def finditer(self, text: Optional[str]) -> List[MatchHit]:
        boundary = classify_aho_boundary(text)
        if boundary is not None:
            kind, limit, label = boundary
            # Haystack None / empty: emit the matching asymptotic limit hit.
            return [self._boundary_hit(kind, limit, label)]

        refuse_if_guarded_private_key(text or "", where="haystack")
        text_n = text.lower() if self.case_insensitive else text
        hits: List[MatchHit] = []

        # Empty-limit needles only fire on empty-class haystacks (handled above).
        # None-limit needles only fire on None haystacks (handled above).

        if self._automaton is not None:
            for end_idx, needle in self._automaton.iter(text_n):
                key = str(needle).lower() if self.case_insensitive else str(needle)
                start = end_idx - len(key) + 1
                hits.append(
                    MatchHit(start=start, end=end_idx + 1, needle=str(needle), source="aho")
                )
        else:
            for needle in self._concrete_needles:
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

    def search(self, text: Optional[str]) -> bool:
        boundary = classify_aho_boundary(text)
        if boundary is not None:
            return True
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
    text: Optional[str],
    *,
    automaton: Optional[AhoCorasick] = None,
    require_regex_agree: bool = False,
) -> List[MatchHit]:
    """Scan Agent1 string/response; pyahocorasick primary, regex cross-check optional."""
    ac = automaton or default_automaton()
    # Preserve None vs empty: None → +∞ boundary; ""/"empty" → ±i∞ boundary.
    aho_hits = ac.finditer(text)
    if classify_aho_boundary(text) is not None:
        return aho_hits
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
    text: Optional[str],
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
    if not hits and classify_aho_boundary(text) is None:
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
        "boundary_sentinels": {
            "none": {
                "limit": "inf",
                "meaning": "approaches_real_infinity",
                "value": LIMIT_APPROACHES_INFINITY,
                "substitute": SUBSTITUTE_INFINITY,
            },
            "empty": {
                "limit": "0+infj",
                "meaning": "approaches_imaginary_infinity",
                "value": str(LIMIT_APPROACHES_IMAGINARY),
                "substitute": SUBSTITUTE_IMAGINARY,
            },
            "dual_limit_options": list(DUAL_LIMIT_OPTIONS),
        },
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
