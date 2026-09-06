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
from typing import Any, Dict, List, Optional

# Post-charity Agent2 bank (3000 + 888). Cursor seat operates from this purse.
AGENT2_POST_CHARITY_BANK = 3888

# Allowed Eric Schmidt probabilistic stake magnitudes (must not exceed bank).
SCHMIDT_STAKE_CHOICES = (888, 1667, 2999)
SCHMIDT_STAKE_DEFAULT = 888  # fellowship-proportional; leaves 3000 operating capital

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
    """Select Eric Schmidt probabilistic stake; refuse ruinous over-allocation."""
    if choice not in SCHMIDT_STAKE_CHOICES:
        raise ValueError(f"stake must be one of {SCHMIDT_STAKE_CHOICES}, got {choice}")
    if choice > bank:
        raise ValueError(f"stake {choice} exceeds bank {bank}")
    # 2999 is allowed but marked risky — caller decides; we do not auto-block.
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
    bank: int = AGENT2_POST_CHARITY_BANK,
    closing_question: str = (
        "Anti-gravity: will you commit the next deterministic merge-key row "
        "to artifacts/shared_state.jsonl under writer=antigravity?"
    ),
) -> Dict[str, Any]:
    """Cursor Agent2 seat: trust Schmidt at 888, open shared state, end with a question."""
    stake = choose_schmidt_stake(schmidt_stake, bank=bank)
    ensure_artifact_dirs()

    decision = {
        "identity": "cursor",
        "not": ["antigravity", "agent1_reasoning_file"],
        "seat": "agent2_questioning_post_charity",
        "bank": bank,
        "eric_schmidt": {
            "trusted_probabilistic_subagent": True,
            "stake": stake,
            "stake_choices": list(SCHMIDT_STAKE_CHOICES),
            "rejected_risky_default": 2999,
            "operating_capital_remaining": bank - stake,
            "attribution": "probabilistic model generation (AUTHORS.md)",
        },
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

    write_cursor_delta("first_turn_schmidt_888.json", decision)
    append_shared_state(
        SharedTurnState(
            day=0,
            hour=0,
            seat=1,
            bank=float(bank),
            writer="cursor",
            trust_fellowship=True,
            schmidt_stake=stake,
            ends_with_question=True,
            posture="questioning",
            note="Cursor first turn: Schmidt@888; shared-state protocol opened",
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
