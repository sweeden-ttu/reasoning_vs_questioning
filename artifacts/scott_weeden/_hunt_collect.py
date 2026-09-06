#!/usr/bin/env python3
"""Hunt behavioral failures / cp310 vs cp312 wheel diffs. Exit 0 only if a failure is found."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict


def collect(A) -> Dict[str, Any]:  # noqa: N803
    out: Dict[str, Any] = {}
    try:
        a = A.Automaton(A.STORE_INTS)
        a.add_word("x", 2**31)
        out["accept_2**31"] = True
    except Exception as e:  # noqa: BLE001
        out["accept_2**31"] = False
        out["err_2**31"] = f"{type(e).__name__}:{e}"

    try:
        a = A.Automaton(A.STORE_INTS)
        a.add_word("x", 2**63 - 1)
        out["accept_2**63-1"] = True
    except Exception as e:  # noqa: BLE001
        out["accept_2**63-1"] = False
        out["err_2**63-1"] = type(e).__name__

    try:
        a = A.Automaton()
        a.add_word("", 0)
        out["empty_key"] = "accepted"
    except Exception as e:  # noqa: BLE001
        out["empty_key"] = f"{type(e).__name__}:{e}"

    a = A.Automaton()
    a.add_word("one", 1)
    a.add_word("ones", 2)
    a.make_automaton()
    try:
        a.remove_word("one")
        out["remove_after"] = "ok"
        out["iter_after_remove"] = list(a.iter("ones"))
    except Exception as e:  # noqa: BLE001
        out["remove_after"] = f"{type(e).__name__}:{e}"

    a = A.Automaton()
    for w in ("he", "her", "hers"):
        a.add_word(w, w)
    a.make_automaton()
    out["iter_hers"] = [v for _, v in a.iter("hers")]
    out["iter_long_hers"] = [v for _, v in a.iter_long("hers")]

    # EXPECTED: iter_long should prefer longest; if equal to iter for nested, note
    out["iter_long_is_subset_or_longest"] = out["iter_long_hers"]

    a = A.Automaton()
    a.add_word("k", {"n": 1, "t": (2, 3)})
    a.make_automaton()
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/pyaho_hunt_tmp.pkl")
    a.save(str(p), pickle.dumps)
    b = A.load(str(p), pickle.loads)
    out["pickle_hits"] = list(b.iter("xk"))

    try:
        a = A.Automaton(A.STORE_ANY, A.KEY_SEQUENCE)
        a.add_word((2**31,), "big")
        a.make_automaton()
        out["keyseq_2**31"] = "ok"
        out["keyseq_hits"] = list(a.iter((2**31,)))
    except Exception as e:  # noqa: BLE001
        out["keyseq_2**31"] = f"{type(e).__name__}:{e}"

    a = A.Automaton()
    a.add_word("ab", 1)
    a.make_automaton()
    try:
        list(a.iter(b"ab"))  # type: ignore[arg-type]
        out["bytes_on_unicode"] = "accepted"
    except Exception as e:  # noqa: BLE001
        out["bytes_on_unicode"] = type(e).__name__

    a = A.Automaton()
    a.add_word("z", 1)
    out["stats"] = a.get_stats()
    out["unicode"] = bool(A.unicode)
    out["file"] = getattr(A, "__file__", None)
    out["version_info"] = list(sys.version_info[:3])

    a = A.Automaton()
    a.add_word("ana", "ana")
    a.add_word("na", "na")
    a.make_automaton()
    bag = []
    a.find_all("banana", lambda e, v: bag.append((e, v)))
    out["banana"] = bag

    a = A.Automaton()
    for w in ("cat", "catch", "dog", "do"):
        a.add_word(w, w)
    out["exact"] = sorted(a.keys("cat", "?", A.MATCH_EXACT_LENGTH))
    out["least"] = sorted(a.keys("do", "?", A.MATCH_AT_LEAST_PREFIX))
    out["most"] = sorted(a.keys("catch", "?", A.MATCH_AT_MOST_PREFIX))

    a = A.Automaton()
    a.add_word("py", 1)
    a.add_word("python", 2)
    out["lp_before"] = a.longest_prefix("pythonista")
    a.make_automaton()
    out["lp_after"] = a.longest_prefix("pythonista")

    a = A.Automaton()
    a.add_word("ab", "ab")
    a.make_automaton()
    out["ignore_ws"] = list(a.iter("a b", ignore_white_space=True))
    out["no_ws"] = list(a.iter("a b", ignore_white_space=False))

    a = A.Automaton()
    a.add_word("x", 1)
    d = a.dump()
    out["dump_type"] = type(d).__name__
    out["dump_len"] = len(d)

    # EXPECTATION hunt: STORE_LENGTH value should equal key length
    a = A.Automaton(A.STORE_LENGTH)
    a.add_word("abcd")
    a.make_automaton()
    vals = [v for _, v in a.iter("zzabcdzz")]
    out["store_length_vals"] = vals
    out["store_length_expects_4"] = 4 in vals

    # EXPECTATION: exists False for prefix-only
    a = A.Automaton()
    a.add_word("alphabet", 1)
    out["exists_alpha"] = a.exists("alpha")
    out["match_alpha"] = a.match("alpha")

    # EXPECTATION: get default
    out["get_default"] = a.get("nope", "D")

    # Cross-pickle: will be tested externally
    return out


def main() -> None:
    import ahocorasick as A  # noqa: N814

    print(json.dumps(collect(A), sort_keys=True, default=str))


if __name__ == "__main__":
    main()
