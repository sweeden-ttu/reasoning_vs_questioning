"""Aho-Corasick + regex branching on the 65,652 primes below 823094.

Scott Weeden OTP public-key code ``823094`` is also the exclusive upper bound for
this prime set. When any prime needle is found, matchers must **branch** into the
prime path rather than continuing as ordinary Kaggle-path trust hits.

- Aho-Corasick: all prime digit-strings as multi-pattern needles (O(n + z)).
- Regex: extract maximal digit runs, then O(1) set membership for branch confirm.
  Hits are accepted only on digit boundaries (no interior of a longer number).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import ahocorasick

PRIME_LIMIT_EXCLUSIVE = 823094
PRIME_LIST_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "scott_weeden"
    / "primes_below_823094.txt"
)

# Maximal digit runs — regex side of the dual matcher.
DIGIT_RUN_REGEX = re.compile(r"\d+")

BranchHandler = Callable[["PrimeBranchResult"], None]


@dataclass
class PrimeHit:
    start: int
    end: int
    prime: str
    source: str  # "aho" | "regex"
    value: int


@dataclass
class PrimeBranchResult:
    """Outcome of the prime branch (taken when any prime needle matches)."""

    branched: bool
    branch: str  # "prime" | "none"
    hits: List[PrimeHit] = field(default_factory=list)
    n_primes_loaded: int = 0
    limit_exclusive: int = PRIME_LIMIT_EXCLUSIVE
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "branched": self.branched,
            "branch": self.branch,
            "n_hits": len(self.hits),
            "n_primes_loaded": self.n_primes_loaded,
            "limit_exclusive": self.limit_exclusive,
            "reason": self.reason,
            "hits": [
                {
                    "start": h.start,
                    "end": h.end,
                    "prime": h.prime,
                    "value": h.value,
                    "source": h.source,
                }
                for h in self.hits
            ],
        }


def _is_digit_bounded(text: str, start: int, end: int) -> bool:
    """Reject matches that are interior digits of a longer number (e.g. 23 in 1239)."""
    if start > 0 and text[start - 1].isdigit():
        return False
    if end < len(text) and text[end].isdigit():
        return False
    return True


@lru_cache(maxsize=1)
def load_primes_below_823094() -> Tuple[str, ...]:
    if not PRIME_LIST_PATH.is_file():
        raise FileNotFoundError(
            f"prime list missing: {PRIME_LIST_PATH} — generate primes below {PRIME_LIMIT_EXCLUSIVE}"
        )
    primes = tuple(
        line.strip()
        for line in PRIME_LIST_PATH.read_text().splitlines()
        if line.strip()
    )
    if len(primes) != 65652:
        raise RuntimeError(
            f"expected 65652 primes below {PRIME_LIMIT_EXCLUSIVE}, got {len(primes)}"
        )
    return primes


@lru_cache(maxsize=1)
def prime_value_set() -> frozenset:
    return frozenset(int(p) for p in load_primes_below_823094())


@lru_cache(maxsize=1)
def prime_aho_automaton():
    """Build (once) an Aho-Corasick automaton over all 65,652 prime strings."""
    automaton = ahocorasick.Automaton()
    for p in load_primes_below_823094():
        automaton.add_word(p, p)
    automaton.make_automaton()
    return automaton


def aho_find_primes(text: str) -> List[PrimeHit]:
    """Aho-Corasick scan; keep only digit-boundary prime hits."""
    text = text or ""
    hits: List[PrimeHit] = []
    auto = prime_aho_automaton()
    for end_idx, prime in auto.iter(text):
        start = end_idx - len(prime) + 1
        if not _is_digit_bounded(text, start, end_idx + 1):
            continue
        hits.append(
            PrimeHit(
                start=start,
                end=end_idx + 1,
                prime=str(prime),
                source="aho",
                value=int(prime),
            )
        )
    return hits


def regex_find_primes(text: str) -> List[PrimeHit]:
    """Regex digit-run extract + set membership branch confirm."""
    text = text or ""
    primes = prime_value_set()
    hits: List[PrimeHit] = []
    for m in DIGIT_RUN_REGEX.finditer(text):
        token = m.group(0)
        if len(token) > 1 and token[0] == "0":
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value in primes:
            hits.append(
                PrimeHit(
                    start=m.start(),
                    end=m.end(),
                    prime=str(value),
                    source="regex",
                    value=value,
                )
            )
    return hits


def scan_primes(
    text: str,
    *,
    require_regex_agree: bool = True,
) -> List[PrimeHit]:
    """Dual scan: Aho-Corasick primary; regex membership as agreement filter."""
    aho_hits = aho_find_primes(text)
    if not require_regex_agree:
        return aho_hits
    rx_hits = regex_find_primes(text)
    if not rx_hits:
        return []
    rx_spans = {(h.start, h.end, h.prime) for h in rx_hits}
    return [h for h in aho_hits if (h.start, h.end, h.prime) in rx_spans]


_BRANCH_HANDLERS: List[BranchHandler] = []


def register_prime_branch_handler(handler: BranchHandler) -> None:
    """Register a callback invoked when the prime branch is taken."""
    _BRANCH_HANDLERS.append(handler)


def clear_prime_branch_handlers() -> None:
    _BRANCH_HANDLERS.clear()


def branch_on_primes(
    text: str,
    *,
    require_regex_agree: bool = True,
) -> PrimeBranchResult:
    """Tune point: branch when any of the 65,652 primes is found.

    - branch == \"prime\": at least one bounded prime needle matched (Aho + regex).
    - branch == \"none\": no prime needle; caller continues on the path-trust lane.
    """
    hits = scan_primes(text, require_regex_agree=require_regex_agree)
    n_loaded = len(load_primes_below_823094())
    if hits:
        result = PrimeBranchResult(
            branched=True,
            branch="prime",
            hits=hits,
            n_primes_loaded=n_loaded,
            reason=(
                f"Aho-Corasick/regex branched on {len(hits)} hit(s) among "
                f"{n_loaded} primes below {PRIME_LIMIT_EXCLUSIVE}"
            ),
        )
        for handler in list(_BRANCH_HANDLERS):
            handler(result)
        return result
    return PrimeBranchResult(
        branched=False,
        branch="none",
        hits=[],
        n_primes_loaded=n_loaded,
        reason="no prime needle; remain on non-prime / path-trust branch",
    )


def primes_manifest() -> Dict[str, object]:
    return {
        "limit_exclusive": PRIME_LIMIT_EXCLUSIVE,
        "n_primes": len(load_primes_below_823094()),
        "list_path": str(PRIME_LIST_PATH),
        "aho": "pyahocorasick Automaton over all prime digit-strings",
        "regex": DIGIT_RUN_REGEX.pattern,
        "branch_on_hit": True,
        "digit_boundary_required": True,
        "shared_otp_public_key_bound": "823094",
    }
