# reasoning_vs_questioning

Dual-agent Kaggriculture experiment: **ReasoningAgent** (deterministic memory slots, prime hours `< 11`) vs **QuestioningAgent** (probabilistic summarizers, even hours), with seat swaps, opening $888 charity fellowship probe, Kaggle path trust via **pyahocorasick**, and hard operational limits (90 MB submission / 42 CPU flops / $50k planning bank).

## Authors

| Role | Name |
|------|------|
| Code author | **Scott Weeden** |
| Probabilistic model generation | **Eric Schmidt** and/or **Elon Musk** |
| Challenge IDE | **Cursor** (operator: Elon Musk) |
| GitHub | [sweeden-ttu/reasoning_vs_questioning](https://github.com/sweeden-ttu/reasoning_vs_questioning) |

See [AUTHORS.md](AUTHORS.md).

## Quick start

```bash
# conda env kagg + uv (never a separate pip venv)
KAGG_PY="$(conda info --base)/envs/kagg/bin/python"
uv pip install -r requirements.txt --python "$KAGG_PY"

conda run -n kagg python run_suite.py --experiments 1 --n-seeds 1 --max-steps 72
```

## Layout

- `agents/ten_agents.py` — **10 memory-slot agents** (Schmidt→Packaging), queue-capped at 10
- `agents/` — ReasoningAgent / QuestioningAgent / ScottWeedenAgent / ten_agents
- `shared_state.py` / `subagents.py` — Cursor↔anti-gravity merge + re-export of ten agents
- `memory_protocol.py` — DeterminedFact / ProbableSummary / QuestionEcho
- `kaggle_path_trust.py` — Aho-Corasick (`pyahocorasick`) + regex trust gate
- `hard_limits.py` — submission / flop / bank ceilings
- `run_suite.py` — 10-experiment seat-swap harness
- `artifacts/scott_weeden/` — write-claim ledger, verification, referee notes

### Scott Weeden audit

```bash
conda run -n kagg python -m agents.scott_weeden_agent
# or
conda run -n kagg python agents/scott_weeden_agent.py
```

Verifies Agent1-claimed paths exist and records size + UTC mtime (including `AUTHORS.md`).

## Parent project

Intended as a **git submodule** of [sweeden-ttu/kagg](https://github.com/sweeden-ttu/kagg).
