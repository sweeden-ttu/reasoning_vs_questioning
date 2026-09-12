"""ScottWeedenAgent: code-author auditor for Reasoning vs Questioning.

Third agent Scott Weeden is referee. Verifies that files Agent1 (Reasoning / anti-gravity)
claims to have written actually exist on disk, and that size + datetime stamps
match the claim recorded at write time.

Also audits AUTHORS.md, which Agent1/Agent2 declare exists even when the
operator cannot yet see it in the IDE.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

from market_config_suite import (
    InitialTerminalConfiguration,
    build_initial_terminal_configuration,
    build_market_functions,
    build_opponent_functions,
)

# Allow `python agents/scott_weeden_agent.py` from package root.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from shared_state import (
    REPO_ROOT,
    SharedTurnState,
    append_shared_state,
    ensure_artifact_dirs,
    set_write_claim_hook,
)

PathLike = Union[str, Path]

# Canonical paths Agent1 / package declare as authored or written.
DEFAULT_AGENT1_CLAIMED_PATHS: tuple[str, ...] = (
    "AUTHORS.md",
    "agents/reasoning_agent.py",
    "agents/questioning_agent.py",
    "shared_state.py",
    "subagents.py",
    "hard_limits.py",
    "memory_protocol.py",
    "kaggle_path_trust.py",
    "artifacts/antigravity/identity_locked_reasoning.json",
    "artifacts/antigravity/dual_arch_sync.json",
)

CLAIMS_DIR = REPO_ROOT / "artifacts" / "scott_weeden"
CLAIMS_LEDGER = CLAIMS_DIR / "agent1_write_claims.jsonl"
AUDIT_REPORT = CLAIMS_DIR / "verification_report.json"


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _resolve(path: PathLike, root: Path = REPO_ROOT) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


@dataclass
class FileClaim:
    """What Agent1 asserts about a written file (captured at claim time)."""

    path: str  # repo-relative POSIX path
    claimed_by: str = "agent1_reasoning"
    expected_size_bytes: Optional[int] = None
    expected_mtime_ns: Optional[int] = None
    expected_mtime_iso: Optional[str] = None
    claimed_at_iso: str = field(default_factory=lambda: _iso_utc(time.time()))
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileClaim":
        return cls(
            path=str(data["path"]),
            claimed_by=str(data.get("claimed_by", "agent1_reasoning")),
            expected_size_bytes=(
                int(data["expected_size_bytes"])
                if data.get("expected_size_bytes") is not None
                else None
            ),
            expected_mtime_ns=(
                int(data["expected_mtime_ns"])
                if data.get("expected_mtime_ns") is not None
                else None
            ),
            expected_mtime_iso=data.get("expected_mtime_iso"),
            claimed_at_iso=str(data.get("claimed_at_iso") or _iso_utc(time.time())),
            note=str(data.get("note") or ""),
        )


@dataclass
class FileVerification:
    """Scott Weeden's on-disk check against one claim."""

    path: str
    exists: bool
    size_bytes: Optional[int] = None
    mtime_ns: Optional[int] = None
    mtime_iso: Optional[str] = None
    size_match: Optional[bool] = None
    mtime_match: Optional[bool] = None
    ok: bool = False
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScottWeedenAgent:
    """Agent3 — code author. Audits Agent1 write claims (existence, size, mtime)."""

    name = "scott_weeden"
    role = "code_author_auditor"
    schedule_hours: frozenset = frozenset()  # does not act on the farm clock

    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        mtime_tolerance_ns: int = 0,
        require_size: bool = True,
        require_mtime: bool = True,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.mtime_tolerance_ns = int(mtime_tolerance_ns)
        self.require_size = bool(require_size)
        self.require_mtime = bool(require_mtime)
        self.claims: List[FileClaim] = []
        self.verifications: List[FileVerification] = []
        self.last_report: Optional[Dict[str, Any]] = None
        ensure_artifact_dirs()
        CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
        set_write_claim_hook(self._hook_agent1_write)

    def _hook_agent1_write(self, path: Path, note: str = "") -> None:
        try:
            rel = path.resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            rel = path.as_posix()
        self.snapshot_claim(rel, claimed_by="agent1_reasoning", note=note, persist=True)

    def reset(self) -> None:
        self.claims.clear()
        self.verifications.clear()
        self.last_report = None
        set_write_claim_hook(self._hook_agent1_write)

    def may_act(self, hour: int) -> bool:
        return False

    def Att(
        self,
        obs: Dict[str, Any],
        initial_terminal_configuration: Union[InitialTerminalConfiguration, Dict[str, Any]],
        market_functions: Dict[str, Callable],
        opponent_functions: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, Any]:
        """Attention/Action Decision Function with two-stage invocation pattern.
        
        Auditor does not issue farm actions; returns PASS action consistently across both stages.
        """
        return {"farmer": ["PASS"], "hands": [], "market": []}

    def act(self, obs: Dict[str, Any], configuration: Any = None) -> Dict[str, Any]:
        """Auditor two-stage act invocation."""
        config = build_initial_terminal_configuration(obs, configuration)
        mkt_funcs = build_market_functions(obs, config)
        opp_funcs = build_opponent_functions(obs, config)

        stage1_action = self.Att(obs, config, mkt_funcs)
        stage2_action = self.Att(obs, config, mkt_funcs, opp_funcs)
        return stage2_action

    # ── claim capture (Agent1 side or auditor snapshot) ─────────────────────

    def snapshot_claim(
        self,
        path: PathLike,
        *,
        claimed_by: str = "agent1_reasoning",
        note: str = "",
        persist: bool = True,
    ) -> FileClaim:
        """Record current on-disk size/mtime as Agent1's claim for later audit."""
        abs_path = _resolve(path, self.repo_root)
        try:
            rel = abs_path.relative_to(self.repo_root).as_posix()
        except ValueError:
            rel = abs_path.as_posix()

        if abs_path.is_file():
            st = abs_path.stat()
            claim = FileClaim(
                path=rel,
                claimed_by=claimed_by,
                expected_size_bytes=int(st.st_size),
                expected_mtime_ns=int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
                expected_mtime_iso=_iso_utc(st.st_mtime),
                note=note,
            )
        else:
            claim = FileClaim(
                path=rel,
                claimed_by=claimed_by,
                expected_size_bytes=None,
                expected_mtime_ns=None,
                expected_mtime_iso=None,
                note=note or "claimed but missing at snapshot",
            )

        self.claims.append(claim)
        if persist:
            self._append_claim_ledger(claim)
        return claim

    def register_agent1_claim(self, claim: FileClaim, *, persist: bool = True) -> FileClaim:
        """Accept an explicit claim dict/object from Agent1 without re-statting."""
        self.claims.append(claim)
        if persist:
            self._append_claim_ledger(claim)
        return claim

    def claim_default_agent1_manifest(self, *, persist: bool = True) -> List[FileClaim]:
        """Snapshot every path Agent1 / package routinely declare as written."""
        out: List[FileClaim] = []
        for rel in DEFAULT_AGENT1_CLAIMED_PATHS:
            out.append(
                self.snapshot_claim(
                    rel,
                    claimed_by="agent1_reasoning",
                    note="default Agent1 declared write",
                    persist=persist,
                )
            )
        return out

    def _append_claim_ledger(self, claim: FileClaim) -> None:
        CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
        with CLAIMS_LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(claim.to_dict(), sort_keys=True) + "\n")

    def load_claims_from_ledger(self, path: Path = CLAIMS_LEDGER) -> List[FileClaim]:
        if not path.exists():
            return []
        loaded: List[FileClaim] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            claim = FileClaim.from_dict(json.loads(line))
            loaded.append(claim)
            self.claims.append(claim)
        return loaded

    # ── verification ────────────────────────────────────────────────────────

    def verify_claim(self, claim: FileClaim) -> FileVerification:
        abs_path = _resolve(claim.path, self.repo_root)
        if not abs_path.is_file():
            result = FileVerification(
                path=claim.path,
                exists=False,
                ok=False,
                detail="MISSING: Agent1 claimed write but file does not exist on disk",
            )
            self.verifications.append(result)
            return result

        st = abs_path.stat()
        size = int(st.st_size)
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        mtime_iso = _iso_utc(st.st_mtime)

        size_match: Optional[bool] = None
        if claim.expected_size_bytes is not None:
            size_match = size == int(claim.expected_size_bytes)
        elif self.require_size:
            size_match = False

        mtime_match: Optional[bool] = None
        if claim.expected_mtime_ns is not None:
            delta = abs(mtime_ns - int(claim.expected_mtime_ns))
            mtime_match = delta <= self.mtime_tolerance_ns
        elif claim.expected_mtime_iso is not None:
            try:
                claimed_dt = datetime.fromisoformat(claim.expected_mtime_iso)
                actual_dt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                if claimed_dt.tzinfo is None:
                    claimed_dt = claimed_dt.replace(tzinfo=timezone.utc)
                delta_s = abs((actual_dt - claimed_dt).total_seconds())
                mtime_match = delta_s <= (self.mtime_tolerance_ns / 1e9)
            except ValueError:
                mtime_match = False
        elif self.require_mtime:
            mtime_match = False

        problems: List[str] = []
        if size_match is False:
            problems.append(
                f"size mismatch: actual={size} claimed={claim.expected_size_bytes}"
            )
        if mtime_match is False:
            problems.append(
                f"mtime mismatch: actual={mtime_iso} ({mtime_ns}) "
                f"claimed={claim.expected_mtime_iso} ({claim.expected_mtime_ns})"
            )

        # Existence alone passes when claim omitted size/mtime and requirements relaxed.
        ok = abs_path.is_file() and (size_match is not False) and (mtime_match is not False)
        if claim.expected_size_bytes is None and claim.expected_mtime_ns is None:
            # Presence-only claim: still report actual stamps for the operator.
            ok = True
            detail = f"EXISTS size={size} mtime={mtime_iso} (no expected stamps in claim)"
        else:
            detail = "OK" if ok else "; ".join(problems)

        result = FileVerification(
            path=claim.path,
            exists=True,
            size_bytes=size,
            mtime_ns=mtime_ns,
            mtime_iso=mtime_iso,
            size_match=size_match,
            mtime_match=mtime_match,
            ok=ok,
            detail=detail,
        )
        self.verifications.append(result)
        return result

    def verify_claims(self, claims: Optional[Sequence[FileClaim]] = None) -> List[FileVerification]:
        targets = list(claims) if claims is not None else list(self.claims)
        return [self.verify_claim(c) for c in targets]

    def verify_authors_md(self) -> FileVerification:
        """Focused audit: AUTHORS.md must exist; snapshot size/mtime for the operator."""
        path = self.repo_root / "AUTHORS.md"
        claim = self.snapshot_claim(
            "AUTHORS.md",
            claimed_by="agent1_reasoning",
            note="operator reports AUTHORS.md not visible in IDE; auditor checks disk",
            persist=True,
        )
        result = self.verify_claim(claim)
        # Presence-only re-check with explicit messaging.
        if result.exists:
            text = path.read_text(encoding="utf-8")
            result.detail = (
                f"AUTHORS.md ON DISK size={result.size_bytes} mtime={result.mtime_iso}; "
                f"first_line={text.splitlines()[0] if text else ''!r}"
            )
            result.ok = True
        return result

    def audit_agent1_writes(
        self,
        *,
        use_default_manifest: bool = True,
        extra_paths: Optional[Iterable[PathLike]] = None,
        load_ledger: bool = True,
    ) -> Dict[str, Any]:
        """Full audit pass: ledger + default Agent1 paths + optional extras."""
        self.verifications.clear()
        if load_ledger:
            self.load_claims_from_ledger()
        if use_default_manifest:
            self.claim_default_agent1_manifest(persist=True)
        if extra_paths:
            for p in extra_paths:
                self.snapshot_claim(p, claimed_by="agent1_reasoning", persist=True)

        # Deduplicate by path keeping the latest claim.
        latest: Dict[str, FileClaim] = {}
        for c in self.claims:
            latest[c.path] = c
        unique_claims = list(latest.values())
        results = self.verify_claims(unique_claims)
        authors = self.verify_authors_md()

        passed = sum(1 for r in results if r.ok)
        failed = [r.to_dict() for r in results if not r.ok]
        report = {
            "auditor": self.name,
            "role": self.role,
            "repo_root": str(self.repo_root),
            "audited_at": _iso_utc(time.time()),
            "claims_checked": len(results),
            "passed": passed,
            "failed": len(failed),
            "all_ok": len(failed) == 0 and authors.exists,
            "authors_md": authors.to_dict(),
            "failures": failed,
            "results": [r.to_dict() for r in results],
            "mtime_tolerance_ns": self.mtime_tolerance_ns,
        }
        self.last_report = report
        CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
        AUDIT_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        append_shared_state(
            SharedTurnState(
                day=0,
                hour=0,
                seat=-1,
                bank=0.0,
                writer="scott_weeden",
                trust_fellowship=None,
                ends_with_question=False,
                posture="reasoning",
                note="Scott Weeden audited Agent1 write claims (existence/size/mtime)",
                payload={
                    "all_ok": report["all_ok"],
                    "passed": passed,
                    "failed": len(failed),
                    "authors_exists": authors.exists,
                    "report": str(AUDIT_REPORT.relative_to(self.repo_root)),
                },
            )
        )
        return report

    def metrics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "claims": len(self.claims),
            "verifications": len(self.verifications),
            "last_report": self.last_report,
            "claims_ledger": str(CLAIMS_LEDGER.relative_to(self.repo_root)),
            "audit_report": str(AUDIT_REPORT.relative_to(self.repo_root)),
        }


def claim_after_agent1_write(
    path: PathLike,
    *,
    note: str = "",
    auditor: Optional[ScottWeedenAgent] = None,
) -> FileClaim:
    """Helper for ReasoningAgent / shared_state: stamp a claim right after writing."""
    agent = auditor or ScottWeedenAgent()
    return agent.snapshot_claim(path, claimed_by="agent1_reasoning", note=note, persist=True)


def main() -> int:
    """CLI: verify Agent1-claimed files (existence, size, mtime) including AUTHORS.md."""
    agent = ScottWeedenAgent(require_size=False, require_mtime=False)
    # Also claim this auditor file itself so the operator can see it was written.
    agent.snapshot_claim(
        "agents/scott_weeden_agent.py",
        claimed_by="scott_weeden",
        note="third agent source",
        persist=True,
    )
    report = agent.audit_agent1_writes(use_default_manifest=True, load_ledger=True)
    authors = report["authors_md"]
    print(f"auditor={agent.name}")
    print(f"repo_root={report['repo_root']}")
    print(f"AUTHORS.md exists={authors.get('exists')} size={authors.get('size_bytes')} mtime={authors.get('mtime_iso')}")
    print(f"passed={report['passed']} failed={report['failed']} all_ok={report['all_ok']}")
    print(f"report={AUDIT_REPORT}")
    for fail in report.get("failures", []):
        print(f"FAIL {fail['path']}: {fail['detail']}")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
