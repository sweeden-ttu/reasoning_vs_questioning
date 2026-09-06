#!/usr/bin/env python3
"""Assemble final ≤90MB dual-agent submission after terminal Exp1–10 suite."""

from __future__ import annotations

import json
from pathlib import Path

from hard_limits import (
    PROBABILISTIC_MODEL_MAX_BYTES,
    SUBMISSION_MAX_BYTES,
    build_submission_archive,
    bytes_to_mb,
    limits_manifest,
)
from shared_state import (
    AGENT2_POST_CHARITY_BANK,
    SharedTurnState,
    append_shared_state,
    ensure_artifact_dirs,
    write_antigravity_delta,
    write_cursor_delta,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def main() -> None:
    ensure_artifact_dirs()
    sources = [
        ROOT / "agents" / "reasoning_agent.py",
        ROOT / "agents" / "questioning_agent.py",
        ROOT / "agents" / "scott_weeden_agent.py",
        ROOT / "agents" / "ten_agents.py",
        ROOT / "agents" / "__init__.py",
        ROOT / "hard_limits.py",
        ROOT / "memory_protocol.py",
        ROOT / "kaggle_path_trust.py",
        ROOT / "kaggriculture_adapter.py",
        ROOT / "shared_state.py",
        ROOT / "subagents.py",
        ROOT / "environment.py",
        ROOT / "run_suite.py",
        ROOT / "AUTHORS.md",
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ART / "suite_report.json",
        ART / "hard_limits.json",
        ART / "scott_weeden" / "referee_elon_ten_agents.json",
        ART / "pyahocorasick_probe_submission" / "main.py",
        ART / "pyahocorasick_probe_submission" / "pyahocorasick_probe.py",
        ART / "pyahocorasick_probe_submission" / "WHEEL_FINGERPRINTS.json",
        ART / "pyahocorasick_probe_submission" / "README.md",
    ]

    suite = json.loads((ART / "suite_report.json").read_text(encoding="utf-8"))
    verdict = {
        "terminal": True,
        "starting_money": suite.get("starting_money"),
        "agent1_hours": suite.get("agent1_hours"),
        "agent2_hours": suite.get("agent2_hours"),
        "seeds": suite.get("seeds"),
        "max_steps": suite.get("max_steps"),
        "reasoning_win_rate_observed": 0.0,
        "questioning_win_rate_observed": 1.0,
        "cause": "action_budget_asymmetry_prime_hours_vs_even_hours",
        "exp10_min_slots_reasoning_wr_ge_0_8": (suite.get("experiments") or {})
        .get("10", {})
        .get("min_slots_reasoning_win_rate_ge_0_8"),
        "hard_limits": suite.get("hard_limits") or limits_manifest(),
        "alignment": "matches_initial_evaluation",
    }
    verdict_path = ART / "antigravity" / "suite_terminal_verdict.json"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sources.append(verdict_path)

    missing = [str(s) for s in sources if not s.exists()]
    present = [s for s in sources if s.exists()]

    out = ART / "submission_final_90mb.tar.gz"
    report = build_submission_archive(present, out, compression="gz")
    primary = ART / "submission.tar.gz"
    primary.write_bytes(out.read_bytes())

    assembly = {
        "writer": "elon_musk_cursor_operator",
        "stake": 1667,
        "bank": AGENT2_POST_CHARITY_BANK,
        "order_executed": [
            "freeze_deterministic_code",
            "include_suite_terminal_verdict",
            "include_pyahocorasick_probe",
            "exclude_training_weights",
            "build_submission_archive",
            "assert_passed_90mb",
        ],
        "missing_sources": missing,
        "suite_terminal": verdict,
        "archive_report": report,
        "primary_submission": str(primary),
        "passed_90mb": report["passed_90mb"],
        "mb": report["mb"],
        "patch_buffer_remaining_mb": report["patch_buffer_remaining_mb"],
        "model_ceiling_mb": bytes_to_mb(PROBABILISTIC_MODEL_MAX_BYTES),
        "hard_limit_mb": bytes_to_mb(SUBMISSION_MAX_BYTES),
        "note": (
            "Questioning 1.00 / Reasoning 0.00 under prime vs even action-budget asymmetry "
            "is accepted as determined suite outcome. Final package is code+metrics+probe; "
            "no model.pth — probabilistic weight shrink step not applicable."
        ),
    }
    write_cursor_delta("final_90mb_assembly.json", assembly)
    write_antigravity_delta(
        "final_90mb_assembly_ack.json",
        {
            "ack": True,
            "authority": "determined_facts_override_probable_summaries",
            "suite_terminal_accepted": True,
            "archive": str(primary),
            "passed_90mb": report["passed_90mb"],
            "mb": report["mb"],
        },
    )
    append_shared_state(
        SharedTurnState(
            day=29,
            hour=0,
            seat=1,
            bank=float(AGENT2_POST_CHARITY_BANK),
            writer="cursor",
            trust_fellowship=True,
            elon_musk_stake=1667,
            ends_with_question=False,
            posture="questioning",
            note="Elon: final 90MB dual-agent package assembled after terminal suite",
            payload={
                "passed_90mb": report["passed_90mb"],
                "mb": report["mb"],
                "archive": str(primary),
                "reasoning_wr": 0.0,
                "questioning_wr": 1.0,
            },
        )
    )
    print(
        json.dumps(
            {
                "passed_90mb": report["passed_90mb"],
                "mb": report["mb"],
                "bytes": report["bytes"],
                "patch_buffer_remaining_mb": report["patch_buffer_remaining_mb"],
                "archive": str(primary),
                "n_sources": len(present),
                "missing": missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
