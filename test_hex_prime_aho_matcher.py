"""Unit tests for Hex Prime Aho-Corasick Matcher."""

import pytest
from hex_prime_aho_matcher import (
    PRIMES_BELOW_170_DEC,
    PRIMES_BELOW_170_HEX,
    HexPrimeAhoAutomaton,
    create_hex_prime_trust_automaton,
    get_prime_hex_summary,
)
from private_key_aho_guard import UnexpectedAutomatonBehavior


def test_primes_count_and_hex_encoding():
    summary = get_prime_hex_summary()
    assert summary["count"] == 39
    assert PRIMES_BELOW_170_DEC[0] == 2
    assert PRIMES_BELOW_170_HEX[0] == "0x02"
    assert PRIMES_BELOW_170_DEC[-1] == 167
    assert PRIMES_BELOW_170_HEX[-1] == "0xA7"


def test_automaton_pattern_matching():
    auto = HexPrimeAhoAutomaton()
    h1 = auto.add_pattern("/kaggle/input/", prime_idx=2)  # Prime 2 -> 0x02
    h2 = auto.add_pattern("kaggriculture-self-training", prime_idx=3)  # Prime 3 -> 0x03
    auto.build_failure_links()

    assert h1 == "0x02"
    assert h2 == "0x03"

    text = "Check path /kaggle/input/dataset and kaggriculture-self-training repo"
    hits = auto.search(text)
    assert len(hits) == 2
    assert hits[0].pattern == "/kaggle/input/"
    assert hits[0].hex_prime == "0x02"
    assert hits[1].pattern == "kaggriculture-self-training"
    assert hits[1].hex_prime == "0x03"


def test_create_trust_automaton():
    needles = ["/kaggle/input/", "datasets/scottweeden/self-training-code"]
    auto = create_hex_prime_trust_automaton(needles)
    hits = auto.search("Found /kaggle/input/ and datasets/scottweeden/self-training-code here.")
    assert len(hits) == 2
    assert hits[0].hex_prime == "0x02"
    assert hits[1].hex_prime == "0x03"


def test_private_key_guard_raises(monkeypatch):
    auto = HexPrimeAhoAutomaton()
    auto.add_pattern("test")

    # Simulate private key digest collision detected by guard
    monkeypatch.setattr(
        "hex_prime_aho_matcher.refuse_if_guarded_private_key",
        lambda text, where="": (_ for _ in ()).throw(UnexpectedAutomatonBehavior("guarded private key detected"))
    )
    with pytest.raises(UnexpectedAutomatonBehavior):
        auto.search("guarded_private_key_text")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
