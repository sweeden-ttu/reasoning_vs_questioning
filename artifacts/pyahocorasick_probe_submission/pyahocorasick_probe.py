"""Comprehensive pyahocorasick / ahocorasick public-API probe for Kaggle.

Goals
-----
1. Detect whether ``import ahocorasick`` works in the host environment.
2. Exercise essentially all *public* module constants and ``Automaton`` methods.
3. Hit edge cases that historically diverge across wheel ABIs, especially:
   - manylinux2014_* (glibc baseline / older tag) vs manylinux_2_17_* (newer tag)
   - CPython 3.10 (cp310) vs CPython 3.12 (cp312)
   - x86_64 vs aarch64 (KEY_SEQUENCE / STORE_INTS width, pickle, unicode)

Does not require the package to be installable from this archive; it tests the
*environment's* installed pyahocorasick. Results are written to
``pyahocorasick_probe_report.json`` next to this file (or ``/kaggle/working``
when present).
"""

from __future__ import annotations

import json
import os
import pickle
import platform
import struct
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


EXPECTED_PUBLIC_MODULE_ATTRS = (
    "Automaton",
    "load",
    "unicode",
    "STORE_ANY",
    "STORE_INTS",
    "STORE_LENGTH",
    "KEY_STRING",
    "KEY_SEQUENCE",
    "EMPTY",
    "TRIE",
    "AHOCORASICK",
    "MATCH_EXACT_LENGTH",
    "MATCH_AT_MOST_PREFIX",
    "MATCH_AT_LEAST_PREFIX",
)

EXPECTED_AUTOMATON_METHODS = (
    "add_word",
    "remove_word",
    "pop",
    "clear",
    "exists",
    "match",
    "get",
    "keys",
    "values",
    "items",
    "iter",
    "iter_long",
    "find_all",
    "longest_prefix",
    "make_automaton",
    "get_stats",
    "dump",
    "save",
    # load is module-level; Automaton may also expose store/kind attrs
)


def _working_dir() -> Path:
    kaggle = Path("/kaggle/working")
    if kaggle.is_dir():
        return kaggle
    return Path(__file__).resolve().parent


def _record(
    results: List[Dict[str, Any]],
    name: str,
    ok: bool,
    *,
    detail: str = "",
    severity: str = "normal",
) -> None:
    results.append(
        {
            "name": name,
            "ok": bool(ok),
            "detail": detail,
            "severity": severity,  # normal | significant | critical
        }
    )


def _run(results: List[Dict[str, Any]], name: str, fn: Callable[[], Any], *, severity: str = "normal") -> Any:
    try:
        out = fn()
        _record(results, name, True, detail=repr(out)[:500], severity=severity)
        return out
    except Exception as exc:  # noqa: BLE001 — probe must capture all failures
        _record(
            results,
            name,
            False,
            detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}",
            severity=severity,
        )
        return None


def probe_environment() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    env = {
        "python_version": sys.version,
        "python_version_info": list(sys.version_info[:3]),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture": list(platform.architecture()),
        "byteorder": sys.byteorder,
        "pointer_bits": struct.calcsize("P") * 8,
        "maxsize": sys.maxsize,
        "executable": sys.executable,
        "ahocorasick_import": None,
        "ahocorasick_file": None,
        "ahocorasick_unicode": None,
        "wheel_hypothesis": (
            "Compare behavior to vendored wheels: "
            "pyahocorasick-2.3.1-cp310-*-manylinux2014_x86_64.manylinux_2_17_x86_64.whl "
            "vs pyahocorasick-2.3.1-cp312-*-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"
        ),
    }

    # ── Import (critical) ───────────────────────────────────────────────────
    try:
        import ahocorasick as AHO  # type: ignore
    except Exception as exc:  # noqa: BLE001
        _record(
            results,
            "import_ahocorasick",
            False,
            detail=f"{type(exc).__name__}: {exc}",
            severity="critical",
        )
        return {
            "package_present": False,
            "all_ok": False,
            "passed": 0,
            "failed": 1,
            "environment": env,
            "results": results,
        }

    env["ahocorasick_import"] = True
    env["ahocorasick_file"] = getattr(AHO, "__file__", None)
    env["ahocorasick_unicode"] = bool(getattr(AHO, "unicode", False))
    env["ahocorasick_version_attr"] = getattr(AHO, "__version__", None)
    _record(results, "import_ahocorasick", True, detail=str(env["ahocorasick_file"]), severity="critical")

    # ── Module public surface ───────────────────────────────────────────────
    missing_mod = [n for n in EXPECTED_PUBLIC_MODULE_ATTRS if not hasattr(AHO, n)]
    _record(
        results,
        "module_public_attrs",
        not missing_mod,
        detail=f"missing={missing_mod}",
        severity="significant",
    )
    for n in EXPECTED_PUBLIC_MODULE_ATTRS:
        if hasattr(AHO, n):
            _record(results, f"module.{n}", True, detail=repr(getattr(AHO, n)))

    # ── Automaton method surface ────────────────────────────────────────────
    proto = AHO.Automaton()
    missing_m = [n for n in EXPECTED_AUTOMATON_METHODS if not hasattr(proto, n)]
    _record(
        results,
        "automaton_public_methods",
        not missing_m,
        detail=f"missing={missing_m}",
        severity="significant",
    )

    # ── STORE_ANY basic trie → automaton (classic hers/she overlap) ─────────
    def case_store_any_overlap() -> Dict[str, Any]:
        a = AHO.Automaton(AHO.STORE_ANY)
        words = "he her hers she".split()
        for i, w in enumerate(words):
            a.add_word(w, (i, w))
        assert a.kind == AHO.TRIE or a.kind in (AHO.EMPTY, AHO.TRIE, AHO.AHOCORASICK)
        a.make_automaton()
        hay = "he hers she"
        found = list(a.iter(hay))
        # must include overlapping / nested hits
        values = [v for _, v in found]
        originals = {v[1] for v in values}
        assert "he" in originals and "hers" in originals and "she" in originals
        return {"hits": len(found), "originals": sorted(originals)}

    _run(results, "STORE_ANY_overlap_iter", case_store_any_overlap, severity="significant")

    # ── STORE_LENGTH ────────────────────────────────────────────────────────
    def case_store_length() -> Dict[str, Any]:
        a = AHO.Automaton(AHO.STORE_LENGTH)
        a.add_word("abc")
        a.add_word("abcd")
        a.make_automaton()
        hits = list(a.iter("zzabcdzz"))
        # values should be lengths
        vals = [v for _, v in hits]
        assert 3 in vals or 4 in vals
        return {"values": vals}

    _run(results, "STORE_LENGTH", case_store_length)

    # ── STORE_INTS + 32-bit edge (significant across arch / py versions) ─────
    def case_store_ints_edges() -> Dict[str, Any]:
        a = AHO.Automaton(AHO.STORE_INTS)
        a.add_word("lo", 0)
        a.add_word("mid", 2**31 - 1)
        a.add_word("neg", -1)
        a.make_automaton()
        got = {k: a.get(k) for k in ("lo", "mid", "neg")}
        assert got["lo"] == 0
        assert got["mid"] == 2**31 - 1
        assert got["neg"] == -1
        overflow_ok = False
        overflow_err = ""
        try:
            b = AHO.Automaton(AHO.STORE_INTS)
            b.add_word("hi", 2**31)  # often rejected / truncated on 32-bit store
            overflow_ok = True
        except Exception as exc:  # noqa: BLE001
            overflow_err = f"{type(exc).__name__}: {exc}"
        return {"got": got, "accepted_2**31": overflow_ok, "overflow_err": overflow_err}

    _run(results, "STORE_INTS_32bit_edges", case_store_ints_edges, severity="significant")

    # ── KEY_SEQUENCE (int tuple keys; width sensitive) ──────────────────────
    def case_key_sequence() -> Dict[str, Any]:
        a = AHO.Automaton(AHO.STORE_ANY, AHO.KEY_SEQUENCE)
        a.add_word((1, 2, 3), "a")
        a.add_word((1, 2, 3, 4), "b")
        a.make_automaton()
        # search with sequence haystack
        hits = list(a.iter((0, 1, 2, 3, 4, 0)))
        vals = [v for _, v in hits]
        assert "a" in vals and "b" in vals
        return {"vals": vals}

    _run(results, "KEY_SEQUENCE_iter", case_key_sequence, severity="significant")

    # ── exists / match / get / pop / remove_word ────────────────────────────
    def case_dict_ops() -> Dict[str, Any]:
        a = AHO.Automaton()
        a.add_word("alpha", 1)
        a.add_word("alphabet", 2)
        assert a.exists("alpha") is True
        assert a.exists("alp") is False
        # match: True if any word has the given prefix (trie sense)
        matched = a.match("alph")
        got = a.get("alpha")
        assert got == 1
        defaulted = a.get("missing", "D")
        assert defaulted == "D"
        raised = False
        try:
            a.get("missing")
        except KeyError:
            raised = True
        assert raised
        removed = a.remove_word("alpha")
        assert a.exists("alpha") is False
        a.add_word("beta", 3)
        popped = a.pop("beta")
        assert popped == 3
        return {"match_alph": matched, "removed": removed, "popped": popped}

    _run(results, "dict_ops_exists_match_get_pop_remove", case_dict_ops)

    # ── keys / values / items with MATCH_* and wildcard ─────────────────────
    def case_keys_match_modes() -> Dict[str, Any]:
        a = AHO.Automaton()
        for w in ("cat", "catch", "dog", "do"):
            a.add_word(w, w)
        # keys(prefix[, wildcard[, how]]) — positional only; wildcard must be 1 char.
        exact = sorted(a.keys("cat", "?", AHO.MATCH_EXACT_LENGTH))
        at_most = sorted(a.keys("catch", "?", AHO.MATCH_AT_MOST_PREFIX))
        at_least = sorted(a.keys("do", "?", AHO.MATCH_AT_LEAST_PREFIX))
        prefix_only = sorted(a.keys("ca"))
        wild = sorted(a.keys("c?t", "?", AHO.MATCH_EXACT_LENGTH))
        vals = sorted(a.values())
        items = sorted(a.items())
        return {
            "exact": exact,
            "at_most": at_most,
            "at_least": at_least,
            "prefix_only": prefix_only,
            "wildcard": wild,
            "values": vals,
            "items": items,
        }

    _run(results, "keys_values_items_MATCH_modes", case_keys_match_modes, severity="significant")

    # ── iter start/end slice + ignore_white_space ───────────────────────────
    def case_iter_slice_ws() -> Dict[str, Any]:
        a = AHO.Automaton()
        a.add_word("ab", "ab")
        a.make_automaton()
        full = list(a.iter("xxabxx"))
        sliced = list(a.iter("xxabxx", 2, 4))
        ws = []
        try:
            ws = list(a.iter("a b", ignore_white_space=True))
        except TypeError as exc:
            ws = [f"ERR:{exc}"]
        return {"full": full, "sliced": sliced, "ignore_ws": ws}

    _run(results, "iter_start_end_ignore_white_space", case_iter_slice_ws, severity="significant")

    # ── iter_long (longest matches) ─────────────────────────────────────────
    def case_iter_long() -> Dict[str, Any]:
        a = AHO.Automaton()
        for w in ("he", "her", "hers"):
            a.add_word(w, w)
        a.make_automaton()
        short = [v for _, v in a.iter("hers")]
        long = [v for _, v in a.iter_long("hers")]
        return {"iter": short, "iter_long": long}

    _run(results, "iter_vs_iter_long", case_iter_long, severity="significant")

    # ── find_all callback ───────────────────────────────────────────────────
    def case_find_all() -> List[Tuple[Any, Any]]:
        a = AHO.Automaton()
        a.add_word("na", "na")
        a.add_word("ana", "ana")
        a.make_automaton()
        bag: List[Tuple[Any, Any]] = []

        def cb(end_index, value):  # noqa: ANN001
            bag.append((end_index, value))

        a.find_all("banana", cb)
        assert bag
        return bag

    _run(results, "find_all_callback", case_find_all)

    # ── longest_prefix ──────────────────────────────────────────────────────
    def case_longest_prefix() -> Any:
        a = AHO.Automaton()
        a.add_word("py", 1)
        a.add_word("python", 2)
        # before or after make_automaton — both should be defined
        lp1 = a.longest_prefix("pythonista")
        a.make_automaton()
        lp2 = a.longest_prefix("pythonista")
        return {"before": lp1, "after": lp2}

    _run(results, "longest_prefix", case_longest_prefix)

    # ── get_stats / dump / clear ────────────────────────────────────────────
    def case_stats_dump_clear() -> Dict[str, Any]:
        a = AHO.Automaton()
        a.add_word("x", 1)
        a.add_word("xy", 2)
        stats = a.get_stats()
        dumped = a.dump()
        a.clear()
        assert list(a.keys()) == []
        return {"stats_keys": sorted(stats.keys()) if isinstance(stats, dict) else type(stats).__name__, "dump_type": type(dumped).__name__, "dump_len": len(dumped) if hasattr(dumped, "__len__") else None}

    _run(results, "get_stats_dump_clear", case_stats_dump_clear)

    # ── save / load round-trip (pickle ABI — significant) ───────────────────
    def case_save_load() -> Dict[str, Any]:
        a = AHO.Automaton()
        a.add_word("/kaggle/input/", "in")
        a.add_word("/kaggle/working/", "work")
        a.make_automaton()
        path = str(_working_dir() / "_probe_automaton.pkl")
        # Public API: save(path, serializer) / load(path, deserializer)
        # serializer must return bytes (pickle.dumps); not pickle.dump.
        a.save(path, pickle.dumps)
        b = AHO.load(path, pickle.loads)
        hits = list(b.iter("see /kaggle/input/datasets here"))
        try:
            Path(path).unlink()
        except OSError:
            pass
        return {"hits": hits, "kind": getattr(b, "kind", None)}

    _run(results, "save_load_roundtrip", case_save_load, severity="significant")

    # ── empty / whitespace / unicode needles (Kaggle path trust style) ──────
    def case_unicode_and_empty_edges() -> Dict[str, Any]:
        a = AHO.Automaton()
        needles = [
            "/kaggle/input/",
            "/kaggle/inputs/",
            "datasets/kaggle",
            "kaggriculture",
            "路径",  # non-ascii
            "🔥",  # emoji
        ]
        for i, n in enumerate(needles):
            a.add_word(n, i)
        a.make_automaton()
        text = "Public /kaggle/input/datasets/kaggle and 路径 and 🔥"
        hits = list(a.iter(text))
        empty_err = ""
        try:
            e = AHO.Automaton()
            e.add_word("", 0)
        except Exception as exc:  # noqa: BLE001
            empty_err = f"{type(exc).__name__}: {exc}"
        return {"hit_count": len(hits), "empty_add": empty_err or "accepted", "unicode_flag": bool(AHO.unicode)}

    _run(results, "unicode_kaggle_needles_and_empty_key", case_unicode_and_empty_edges, severity="significant")

    # ── bytes vs str (compile-time unicode flag) ────────────────────────────
    def case_bytes_vs_str() -> Dict[str, Any]:
        if AHO.unicode:
            a = AHO.Automaton()
            a.add_word("abc", 1)
            a.make_automaton()
            err = ""
            try:
                list(a.iter(b"abc"))  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
            return {"build": "unicode", "bytes_iter_error": err or "unexpectedly_accepted"}
        a = AHO.Automaton()
        a.add_word(b"abc", 1)
        a.make_automaton()
        hits = list(a.iter(b"zzabczz"))
        return {"build": "bytes", "hits": hits}

    _run(results, "bytes_vs_str_by_unicode_flag", case_bytes_vs_str, severity="significant")

    # ── remove_word after make_automaton (edge; may fail or invalidate) ─────
    def case_remove_after_automaton() -> Dict[str, Any]:
        a = AHO.Automaton()
        a.add_word("one", 1)
        a.add_word("ones", 2)
        a.make_automaton()
        err = ""
        removed = None
        try:
            removed = a.remove_word("one")
            # subsequent iter should still be defined
            _ = list(a.iter("ones"))
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        return {"removed": removed, "error": err}

    _run(results, "remove_word_after_make_automaton", case_remove_after_automaton, severity="significant")

    # ── very long pattern / haystack (memory / overflow edge) ───────────────
    def case_long_strings() -> Dict[str, Any]:
        a = AHO.Automaton()
        pat = "x" * 10_000
        a.add_word(pat, "long")
        a.make_automaton()
        hay = ("y" * 1000) + pat + ("z" * 1000)
        hits = list(a.iter(hay))
        assert hits and hits[0][1] == "long"
        return {"end_index": hits[0][0], "pat_len": len(pat)}

    _run(results, "long_pattern_10000", case_long_strings)

    # ── many patterns (Aho-Corasick fan-out) ─────────────────────────────────
    def case_many_patterns() -> int:
        a = AHO.Automaton()
        for i in range(500):
            a.add_word(f"p{i:04d}", i)
        a.make_automaton()
        text = " ".join(f"p{i:04d}" for i in range(0, 500, 17))
        return len(list(a.iter(text)))

    _run(results, "many_patterns_500", case_many_patterns)

    passed = sum(1 for r in results if r["ok"])
    failed = sum(1 for r in results if not r["ok"])
    critical_fail = any(not r["ok"] and r["severity"] == "critical" for r in results)
    significant_fail = any(not r["ok"] and r["severity"] == "significant" for r in results)

    report = {
        "package_present": True,
        "all_ok": failed == 0,
        "passed": passed,
        "failed": failed,
        "critical_fail": critical_fail,
        "significant_fail": significant_fail,
        "environment": env,
        "results": results,
        "verdict": (
            "PASS_ENV_HAS_PYAHOCORASICK"
            if failed == 0
            else (
                "FAIL_IMPORT"
                if critical_fail
                else "FAIL_BEHAVIORAL_DIFF_OR_BUG"
            )
        ),
    }
    return report


def main() -> int:
    report = probe_environment()
    out = _working_dir() / "pyahocorasick_probe_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("package_present", "verdict", "passed", "failed", "critical_fail", "significant_fail")}, sort_keys=True))
    print(f"report={out}")
    if not report["package_present"]:
        return 2
    if report["critical_fail"]:
        return 2
    if report["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
