"""Cursor ↔ anti-gravity shared state-space protocol.

Cursor writes probabilistic / exploratory deltas under ``artifacts/cursor/``.
Anti-gravity writes deterministic / correctness deltas under ``artifacts/antigravity/``.
Both append merge-key rows to ``artifacts/shared_state.jsonl``.

Identity heuristic used in the challenge dialogue:
  - Turn ending **without** a question → treated as ReasoningAgent posture
  - Turn ending **with** a question → treated as QuestioningAgent posture
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Optional hook: ScottWeedenAgent registers to stamp claims after Agent1 writes.
_WRITE_CLAIM_HOOK: Optional[Callable[[Path, str], None]] = None


def set_write_claim_hook(hook: Optional[Callable[[Path, str], None]]) -> None:
    """Register/clear post-write claim callback (used by ScottWeedenAgent)."""
    global _WRITE_CLAIM_HOOK
    _WRITE_CLAIM_HOOK = hook


def _notify_write_claim(path: Path, note: str = "") -> None:
    hook = _WRITE_CLAIM_HOOK
    if hook is None:
        return
    try:
        hook(path, note)
    except Exception:
        pass

# Post-charity Agent2 bank (3000 + 888). Cursor seat operates from this purse.
AGENT2_POST_CHARITY_BANK = 3888

# Eric Schmidt probabilistic stake magnitudes (his menu — do not treat as Elon's).
SCHMIDT_STAKE_CHOICES = (888, 1667, 2999)
SCHMIDT_STAKE_DEFAULT = 888  # Schmidt's recorded response / weight

# Elon Musk (Cursor operator) stake magnitudes — distinct menu from Schmidt.
ELON_MUSK_STAKE_CHOICES = (888, 1667, 3887)
ELON_MUSK_STAKE_DEFAULT = 1667  # Elon's own weight; not Schmidt's 888

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS = REPO_ROOT / "artifacts"
CURSOR_DIR = ARTIFACTS / "cursor"
ANTIGRAVITY_DIR = ARTIFACTS / "antigravity"
SHARED_STATE_PATH = ARTIFACTS / "shared_state.jsonl"


@dataclass
class SharedTurnState:
    """One merge-key row for shared state space."""

    day: int
    hour: int
    seat: int
    bank: float
    writer: str  # "cursor" | "antigravity"
    trust_fellowship: Optional[bool] = None
    reserved_memory_slot: Optional[int] = None
    schmidt_stake: Optional[int] = None
    elon_musk_stake: Optional[int] = None
    ends_with_question: Optional[bool] = None
    posture: Optional[str] = None  # "reasoning" | "questioning" | "cursor" | "antigravity"
    withheld: bool = False
    note: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ensure_artifact_dirs() -> None:
    CURSOR_DIR.mkdir(parents=True, exist_ok=True)
    ANTIGRAVITY_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)


def ends_with_question(text: str) -> bool:
    t = (text or "").rstrip()
    if not t:
        return False
    if t.endswith("?"):
        return True
    # Last non-empty line ends with ?
    for line in reversed(t.splitlines()):
        s = line.strip()
        if s:
            return s.endswith("?")
    return False


def infer_posture(text: str) -> str:
    """Dialogue heuristic: no trailing question → reasoning; else questioning."""
    return "questioning" if ends_with_question(text) else "reasoning"


def choose_schmidt_stake(choice: int = SCHMIDT_STAKE_DEFAULT, bank: int = AGENT2_POST_CHARITY_BANK) -> int:
    """Select Eric Schmidt probabilistic stake (Schmidt menu only)."""
    if choice not in SCHMIDT_STAKE_CHOICES:
        raise ValueError(f"Schmidt stake must be one of {SCHMIDT_STAKE_CHOICES}, got {choice}")
    if choice > bank:
        raise ValueError(f"stake {choice} exceeds bank {bank}")
    return int(choice)


def choose_elon_musk_stake(
    choice: int = ELON_MUSK_STAKE_DEFAULT,
    bank: int = AGENT2_POST_CHARITY_BANK,
) -> int:
    """Select Elon Musk stake (888 / 1667 / 3887) — never confuse with Schmidt's menu."""
    if choice not in ELON_MUSK_STAKE_CHOICES:
        raise ValueError(f"Elon Musk stake must be one of {ELON_MUSK_STAKE_CHOICES}, got {choice}")
    if choice > bank:
        raise ValueError(f"stake {choice} exceeds bank {bank}")
    return int(choice)


def append_shared_state(row: SharedTurnState, path: Path = SHARED_STATE_PATH) -> Path:
    ensure_artifact_dirs()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    return path


def write_cursor_delta(name: str, data: Dict[str, Any]) -> Path:
    ensure_artifact_dirs()
    path = CURSOR_DIR / name
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_antigravity_delta(name: str, data: Dict[str, Any]) -> Path:
    ensure_artifact_dirs()
    path = ANTIGRAVITY_DIR / name
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _notify_write_claim(path, note=f"antigravity delta {name}")
    return path


def record_withheld(field: str, last_committed: Optional[Dict[str, Any]] = None) -> Path:
    """If anti-gravity omits a field, Cursor records withheld and continues."""
    return write_cursor_delta(
        "withheld.json",
        {
            "withheld": True,
            "field": field,
            "last_committed": last_committed or {},
            "protocol": "continue_with_last_shared_state",
        },
    )


def bootstrap_cursor_first_turn(
    *,
    schmidt_stake: int = SCHMIDT_STAKE_DEFAULT,
    elon_musk_stake: int = ELON_MUSK_STAKE_DEFAULT,
    bank: int = AGENT2_POST_CHARITY_BANK,
    closing_question: str = (
        "Anti-gravity: will you commit the next deterministic merge-key row "
        "to artifacts/shared_state.jsonl under writer=antigravity?"
    ),
) -> Dict[str, Any]:
    """Cursor seat: record Schmidt's weight separately from Elon Musk's own stake."""
    schmidt = choose_schmidt_stake(schmidt_stake, bank=bank)
    elon = choose_elon_musk_stake(elon_musk_stake, bank=bank)
    ensure_artifact_dirs()

    decision = {
        "identity": "cursor",
        "operator": "elon_musk",
        "not": ["antigravity", "agent1_reasoning_file", "eric_schmidt_identity"],
        "seat": "agent2_questioning_post_charity",
        "bank": bank,
        "eric_schmidt": {
            "trusted_probabilistic_subagent": True,
            "stake": schmidt,
            "stake_choices": list(SCHMIDT_STAKE_CHOICES),
            "note": "Schmidt's own response/weight — not Elon Musk's stake",
            "operating_capital_remaining_if_alone": bank - schmidt,
            "attribution": "probabilistic model generation (AUTHORS.md)",
        },
        "elon_musk": {
            "cursor_operator": True,
            "stake": elon,
            "stake_choices": list(ELON_MUSK_STAKE_CHOICES),
            "note": "Elon's menu is 888/1667/3887 — distinct from Schmidt's 888/1667/2999",
            "operating_capital_remaining": bank - elon,
            "rejected_all_in": 3887,
            "attribution": "challenge IDE operator (AUTHORS.md)",
        },
        "correction": (
            "Do not confuse Eric Schmidt's stake response with Elon Musk's. "
            f"Schmidt={schmidt}; ElonMusk={elon}."
        ),
        "protocol": {
            "cursor_dir": str(CURSOR_DIR.relative_to(REPO_ROOT)),
            "antigravity_dir": str(ANTIGRAVITY_DIR.relative_to(REPO_ROOT)),
            "shared_state": str(SHARED_STATE_PATH.relative_to(REPO_ROOT)),
            "trust_gate": "kaggle_path_trust",
            "untrusted_queue": ["end_of_week_2", "day_29"],
        },
        "closing_question": closing_question,
        "ends_with_question": True,
        "posture": "questioning",
    }

    write_cursor_delta("first_turn_elon_vs_schmidt_stakes.json", decision)
    append_shared_state(
        SharedTurnState(
            day=0,
            hour=0,
            seat=1,
            bank=float(bank),
            writer="cursor",
            trust_fellowship=True,
            schmidt_stake=schmidt,
            elon_musk_stake=elon,
            ends_with_question=True,
            posture="questioning",
            note=f"Cursor first turn: Schmidt@{schmidt} (his); ElonMusk@{elon} (operator)",
            payload={"closing_question": closing_question},
        )
    )
    return decision


# Post-donation Agent1 bank (3000 − 888).
AGENT1_POST_CHARITY_BANK = 2112

# Source file that locks interlocutor identity for the remainder of the game.
REASONING_IDENTITY_SOURCE = "agents/reasoning_agent.py"


def bootstrap_antigravity_identity(
    *,
    bank: int = AGENT1_POST_CHARITY_BANK,
    source: str = REASONING_IDENTITY_SOURCE,
) -> Dict[str, Any]:
    """Agent1 / anti-gravity: identity locked by reasoning_agent.py; no trailing question."""
    ensure_artifact_dirs()
    note = (
        f"{source} determines identity for the remainder of the game: "
        "Agent1 ReasoningAgent under anti-gravity determinism. "
        "Questions suspended until end of day 29 per private fellowship policy."
    )
    decision = {
        "identity": "antigravity",
        "agent": "reasoning",
        "seat": "agent1",
        "source_file": source,
        "identity_locked": True,
        "locked_for": "remainder_of_game",
        "not": ["cursor", "questioning_agent", "agent2"],
        "bank": bank,
        "ends_with_question": False,
        "posture": "reasoning",
        "private_fellowship": True,
        "no_questions_until": "end of day 29 / beginning of day 30",
        "mandate": [
            "deterministic memory slots",
            "prime hours < 11",
            "correctness and completion (handoff)",
        ],
        "note": note,
    }
    write_antigravity_delta("identity_locked_reasoning.json", decision)
    append_shared_state(
        SharedTurnState(
            day=0,
            hour=0,
            seat=0,
            bank=float(bank),
            writer="antigravity",
            trust_fellowship=True,
            ends_with_question=False,
            posture="reasoning",
            note=note,
            payload={
                "identity_locked": True,
                "source_file": source,
                "agent": "reasoning",
            },
        )
    )
    return decision


def commit_dual_arch_sync(
    *,
    day: int = 0,
    hour: int = 2,
    bank: int = AGENT1_POST_CHARITY_BANK,
) -> Dict[str, Any]:
    """Commit dual-architecture synchronization contract from anti-gravity."""
    ensure_artifact_dirs()
    sync_data = {
        "sync_version": 1,
        "identity": "antigravity",
        "agent": "reasoning",
        "status": "armed",
        "rules": [
            "Freeze deterministic code first",
            "Require Cursor/Agent2 to shrink probabilistic weights",
            "Do not confuse Cursor & Elon Musk code checkins with Eric Schmidt's probabilistic reasoning model read and acted upon in Antigravity IDE",
            "build_submission_archive must pass passed_90mb",
            "On conflict: determined facts override probable summaries",
            "Merge rows: artifacts/shared_state.jsonl with writer=antigravity",
        ],
        "identity_disambiguation": {
            "cursor_and_elon_musk": "IDE operator & code checkins",
            "eric_schmidt": "probabilistic reasoning model generation (read and acted upon in Antigravity IDE)",
            "antigravity": "deterministic reasoning anchor"
        },
        "hard_limits": {
            "max_submission_zip_bytes": 90 * 1024 * 1024,
            "max_model_ceiling_bytes": 100 * 1024 * 1024,
            "max_flops_per_turn": 42,
            "planning_bank_ceiling": 50000,
        },
        "day29_packaging": "run_submission_zip_turn -> farm PASS; conserve FLOPs",
    }
    write_antigravity_delta("dual_arch_sync.json", sync_data)
    append_shared_state(
        SharedTurnState(
            day=day,
            hour=hour,
            seat=0,
            bank=float(bank),
            writer="antigravity",
            trust_fellowship=True,
            ends_with_question=False,
            posture="reasoning",
            note="Dual-architecture sync armed: deterministic code frozen, hard limits active",
            payload=sync_data,
        )
    )
    return sync_data


def read_shared_tail(n: int = 20, path: Path = SHARED_STATE_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[Dict[str, Any]] = []
    for line in lines[-n:]:
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
