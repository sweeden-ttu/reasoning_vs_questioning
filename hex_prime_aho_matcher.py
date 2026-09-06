"""Private Hex-Encoded Prime Aho-Corasick Matcher.

Converts all 39 prime numbers < 170 into hexadecimal encoding and provides
a native, pure-Python Aho-Corasick multi-pattern automaton for private,
zero-dependency matching guarded against private key exfiltration.

Primes < 170 (39 primes):
  Decimal: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
            67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
            139, 149, 151, 157, 163, 167]
  Hex:     ['0x02', '0x03', '0x05', '0x07', '0x0B', '0x0D', '0x11', '0x13',
            '0x17', '0x1D', '0x1F', '0x25', '0x29', '0x2B', '0x2F', '0x35',
            '0x3B', '0x3D', '0x43', '0x47', '0x49', '0x4F', '0x53', '0x59',
            '0x61', '0x65', '0x67', '0x6B', '0x6D', '0x71', '0x7F', '0x83',
            '0x89', '0x8B', '0x95', '0x97', '0x9D', '0xA3', '0xA7']
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from private_key_aho_guard import (
    UnexpectedAutomatonBehavior,
    refuse_if_guarded_private_key,
)

# ── Primes < 170 (39 primes total) ───────────────────────────────────────────

PRIMES_BELOW_170_DEC: Tuple[int, ...] = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
    67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
    139, 149, 151, 157, 163, 167,
)

PRIMES_BELOW_170_HEX: Tuple[str, ...] = tuple(
    f"0x{p:02X}" for p in PRIMES_BELOW_170_DEC
)

PRIMES_HEX_MAP: Dict[int, str] = dict(zip(PRIMES_BELOW_170_DEC, PRIMES_BELOW_170_HEX))
HEX_PRIMES_MAP: Dict[str, int] = {h: p for p, h in PRIMES_HEX_MAP.items()}


@dataclass
class HexPrimeMatchHit:
    start: int
    end: int
    pattern: str
    hex_prime: str
    prime_dec: int
    value: Any = None


class _TrieNode:
    def __init__(self) -> None:
        self.children: Dict[Union[str, int], _TrieNode] = {}
        self.fail: Optional[_TrieNode] = None
        self.outputs: List[Tuple[str, int, str, Any]] = []  # (pattern, prime_dec, hex_prime, value)


class HexPrimeAhoAutomaton:
    """Pure-Python Aho-Corasick automaton operating over hex-encoded prime states.

    Performs linear-time multi-pattern matching while remaining fully private,
    zero-dependency, and guarded against private-key scanning.
    """

    def __init__(self) -> None:
        self._root = _TrieNode()
        self._built = False
        self._patterns: List[Tuple[str, int, str, Any]] = []

    def add_pattern(self, pattern: str, prime_idx: Optional[int] = None, value: Any = None) -> str:
        """Add a pattern associated with a specific prime < 170 (by index 0..38 or decimal value)."""
        if not pattern:
            raise ValueError("Pattern cannot be empty")
        
        if prime_idx is None:
            prime_idx = len(self._patterns) % len(PRIMES_BELOW_170_DEC)
        
        if prime_idx in PRIMES_HEX_MAP:
            p_dec = prime_idx
            p_hex = PRIMES_HEX_MAP[prime_idx]
        elif 0 <= prime_idx < len(PRIMES_BELOW_170_DEC):
            p_dec = PRIMES_BELOW_170_DEC[prime_idx]
            p_hex = PRIMES_BELOW_170_HEX[prime_idx]
        else:
            p_dec = PRIMES_BELOW_170_DEC[0]
            p_hex = PRIMES_BELOW_170_HEX[0]

        node = self._root
        for char in pattern:
            if char not in node.children:
                node.children[char] = _TrieNode()
            node = node.children[char]

        item = (pattern, p_dec, p_hex, value)
        node.outputs.append(item)
        self._patterns.append(item)
        self._built = False
        return p_hex

    def build_failure_links(self) -> None:
        """Build Aho-Corasick failure links using BFS."""
        queue: deque[_TrieNode] = deque()

        # Root children failure links point to root
        for child in self._root.children.values():
            child.fail = self._root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for char, child in current.children.items():
                fail_node = current.fail
                while fail_node is not None and char not in fail_node.children:
                    fail_node = fail_node.fail
                
                child.fail = fail_node.children[char] if fail_node else self._root
                if child.fail:
                    child.outputs.extend(child.fail.outputs)
                queue.append(child)

        self._built = True

    def search(self, text: str) -> List[HexPrimeMatchHit]:
        """Search text for patterns, returning all match hits."""
        refuse_if_guarded_private_key(text, where="hex_prime_aho_search")
        
        if not self._built:
            self.build_failure_links()

        hits: List[HexPrimeMatchHit] = []
        node = self._root

        for idx, char in enumerate(text):
            while node is not None and char not in node.children:
                node = node.fail

            if node is None:
                node = self._root
                continue

            node = node.children[char]
            for pattern, p_dec, p_hex, val in node.outputs:
                start = idx - len(pattern) + 1
                end = idx + 1
                hits.append(
                    HexPrimeMatchHit(
                        start=start,
                        end=end,
                        pattern=pattern,
                        hex_prime=p_hex,
                        prime_dec=p_dec,
                        value=val,
                    )
                )

        return hits


def create_hex_prime_trust_automaton(needles: Sequence[str]) -> HexPrimeAhoAutomaton:
    """Build a private Hex-Prime Aho-Corasick automaton loaded with trust needles."""
    automaton = HexPrimeAhoAutomaton()
    for idx, needle in enumerate(needles):
        automaton.add_pattern(needle, prime_idx=idx)
    automaton.build_failure_links()
    return automaton


def get_prime_hex_summary() -> Dict[str, Any]:
    """Return summary dictionary of all 39 primes < 170 and their hex encodings."""
    return {
        "count": len(PRIMES_BELOW_170_DEC),
        "primes_decimal": list(PRIMES_BELOW_170_DEC),
        "primes_hex": list(PRIMES_BELOW_170_HEX),
        "hex_mapping": {str(d): h for d, h in PRIMES_HEX_MAP.items()},
    }


if __name__ == "__main__":
    print("Hex Prime Matcher Initialized.")
    summary = get_prime_hex_summary()
    print(f"Total primes < 170: {summary['count']}")
    print(f"Hex encodings: {summary['primes_hex']}")
