# pyahocorasick Kaggle environment probe submission

This archive tests whether the **Kaggle runtime** provides `pyahocorasick`
(`import ahocorasick`) and whether its public API behaves correctly.

## Contents

| File | Role |
|------|------|
| `main.py` | Kaggle agent entry (`agent(obs, config)`); runs probe once, then PASS |
| `pyahocorasick_probe.py` | Full public-API + edge-case suite |
| `WHEEL_FINGERPRINTS.json` | Expected dual-arch wheel tags for 2.3.1 (cp310 x86_64 vs cp312 aarch64) |

## What is tested

- Import presence (critical)
- Module constants: `STORE_*`, `KEY_*`, `MATCH_*`, `EMPTY`/`TRIE`/`AHOCORASICK`, `unicode`, `load`
- Automaton methods: `add_word`, `remove_word`, `pop`, `clear`, `exists`, `match`, `get`,
  `keys`/`values`/`items`, `iter`/`iter_long`, `find_all`, `longest_prefix`,
  `make_automaton`, `get_stats`, `dump`, `save`
- Edge cases with **significant** ABI/platform consequence between
  `manylinux2014_*` / `manylinux_2_17_*` tags and **cp310 vs cp312** / **x86_64 vs aarch64**:
  - `STORE_INTS` 32-bit limits (`2**31-1`, `2**31`, `-1`)
  - `KEY_SEQUENCE` integer-tuple search
  - `save`/`load` pickle round-trip
  - unicode vs bytes haystacks (`ahocorasick.unicode`)
  - `iter(..., start, end)` and `ignore_white_space`
  - `iter` vs `iter_long` on overlapping patterns
  - `remove_word` after `make_automaton`
  - empty key, non-ASCII / emoji needles, 10k-char pattern, 500-pattern fan-out
  - `MATCH_EXACT_LENGTH` / `MATCH_AT_MOST_PREFIX` / `MATCH_AT_LEAST_PREFIX` (+ wildcard)

## Output

Writes `pyahocorasick_probe_report.json` to `/kaggle/working/` (or cwd) with
`verdict` ∈ {`PASS_ENV_HAS_PYAHOCORASICK`, `FAIL_IMPORT`, `FAIL_BEHAVIORAL_DIFF_OR_BUG`}.

## Local run

```bash
conda run -n kagg python main.py
# or
conda run -n kagg python pyahocorasick_probe.py
```
