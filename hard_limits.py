"""Hard operational limits for Kaggriculture dual-agent challenge.

Hard limits (enforced):
  - Probabilistic model payload ceiling: 100 MB
  - Final submission (code + model zip): 90 MB (10 MB patch buffer under 100 MB)
  - Per-turn sub-processing flops: 42 (one per CPU before disk-observable handoff)
  - Bank planning ceiling: $50,000 (Kaggle test purse assumption)
  - Final submission window: day 29, 5 turns to zip and submit
  - Post-training model size gate: after 24 hours of training, final weights must
    fit the submission budget (training corpus / checkpoint growth is NOT capped)

Soft / schedule context (documented, not size-capped):
  - Calendar season is 30 days, but strategy discovery budget is < 2 weeks (14 days)
  - Test hardware profile: 42 CPUs + 3 GPUs (NVLink, single-GPU pipeline, no inforom)
  - Deterministic agent code must share the submission size budget with the model
"""

from __future__ import annotations

import json
import logging
import tarfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ── Hard limits ─────────────────────────────────────────────────────────────

MB = 1024 * 1024

PROBABILISTIC_MODEL_MAX_BYTES: int = 100 * MB
SUBMISSION_MAX_BYTES: int = 90 * MB  # leave 10 MB under the 100 MB ceiling
SUBMISSION_PATCH_BUFFER_BYTES: int = 10 * MB
assert SUBMISSION_MAX_BYTES + SUBMISSION_PATCH_BUFFER_BYTES == PROBABILISTIC_MODEL_MAX_BYTES

MAX_SUBPROCESS_FLOPS_PER_TURN: int = 42  # one flop-unit per CPU before observability
MAX_PLANNING_BANK_COINS: int = 50_000  # do not assume true banks above Kaggle purse

TRAINING_WALL_CLOCK_HOURS_BEFORE_SIZE_GATE: float = 24.0
# Training data volume and intermediate checkpoint size: NOT hard-limited.

STRATEGY_DISCOVERY_DAYS: int = 14  # < 2 weeks to find a winning strategy
SEASON_DAYS: int = 30
SUBMISSION_DAY: int = 29
SUBMISSION_ZIP_TURNS: int = 5  # only 5 turns on day 29 to package and submit

# ── Hardware profile (test cluster) ─────────────────────────────────────────

TEST_CPUS: int = 42
TEST_GPUS: int = 3
NVLINK_SINGLE_PIPELINE: bool = True  # GPU↔GPU limited to one pipeline; no inforom


@dataclass
class HardLimits:
    probabilistic_model_max_bytes: int = PROBABILISTIC_MODEL_MAX_BYTES
    submission_max_bytes: int = SUBMISSION_MAX_BYTES
    submission_patch_buffer_bytes: int = SUBMISSION_PATCH_BUFFER_BYTES
    max_subprocess_flops_per_turn: int = MAX_SUBPROCESS_FLOPS_PER_TURN
    max_planning_bank_coins: int = MAX_PLANNING_BANK_COINS
    training_hours_before_size_gate: float = TRAINING_WALL_CLOCK_HOURS_BEFORE_SIZE_GATE
    strategy_discovery_days: int = STRATEGY_DISCOVERY_DAYS
    season_days: int = SEASON_DAYS
    submission_day: int = SUBMISSION_DAY
    submission_zip_turns: int = SUBMISSION_ZIP_TURNS
    test_cpus: int = TEST_CPUS
    test_gpus: int = TEST_GPUS
    nvlink_single_pipeline: bool = NVLINK_SINGLE_PIPELINE
    # Explicit non-limits (documented for auditors / configs)
    training_data_size_capped: bool = False
    training_checkpoint_size_capped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_LIMITS = HardLimits()


def bytes_to_mb(n: int) -> float:
    return float(n) / MB


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def clamp_planning_bank(money: float, limits: HardLimits = DEFAULT_LIMITS) -> float:
    """Cap believed/planned bank at the Kaggle purse assumption ($50k)."""
    return min(float(money), float(limits.max_planning_bank_coins))


def within_strategy_window(day: int, limits: HardLimits = DEFAULT_LIMITS) -> bool:
    """True if still inside the <2-week strategy-discovery window (days 0..13)."""
    return 0 <= int(day) < int(limits.strategy_discovery_days)


def in_submission_zip_window(
    day: int,
    hour: int,
    limits: HardLimits = DEFAULT_LIMITS,
) -> bool:
    """Day 29, first ``submission_zip_turns`` hours are the zip/submit window."""
    return int(day) == int(limits.submission_day) and 0 <= int(hour) < int(limits.submission_zip_turns)


@dataclass
class TurnComputeBudget:
    """Hard cap of 42 sub-processing flops before the turn must hit disk."""

    max_flops: int = MAX_SUBPROCESS_FLOPS_PER_TURN
    used: int = 0
    spilled_to_disk: bool = False
    log: List[str] = field(default_factory=list)

    def remaining(self) -> int:
        return max(0, self.max_flops - self.used)

    def consume(self, units: int = 1, label: str = "") -> bool:
        """Spend flop units. Returns False if budget exhausted (must observe/disk)."""
        units = max(0, int(units))
        if self.used + units > self.max_flops:
            self.spilled_to_disk = True
            self.log.append(f"DENIED:{label or 'anon'}:need={units}:left={self.remaining()}")
            return False
        self.used += units
        self.log.append(f"OK:{label or 'anon'}:used={units}:total={self.used}")
        return True

    def force_observable(self) -> None:
        """Mark that the turn is now system-observable (written / opponent-visible)."""
        self.spilled_to_disk = True
        self.log.append("OBSERVABLE")

    def reset(self) -> None:
        self.used = 0
        self.spilled_to_disk = False
        self.log.clear()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "max_flops": self.max_flops,
            "used": self.used,
            "remaining": self.remaining(),
            "spilled_to_disk": self.spilled_to_disk,
            "events": list(self.log),
        }


def check_model_size(
    model_path: Path,
    *,
    limits: HardLimits = DEFAULT_LIMITS,
    after_training_hours: Optional[float] = None,
) -> Dict[str, Any]:
    """Enforce probabilistic model ≤ 100 MB.

    If ``after_training_hours`` >= 24, this is the mandatory final-size gate
    (training growth before that gate is unrestricted).
    """
    size = path_size_bytes(Path(model_path))
    gated = (
        after_training_hours is not None
        and after_training_hours >= limits.training_hours_before_size_gate
    )
    ok = size <= limits.probabilistic_model_max_bytes
    # After 24h, final model must also leave room for deterministic code in the
    # 90 MB submission package — prefer model alone under submission budget when gated.
    submission_fit = size <= limits.submission_max_bytes
    result = {
        "path": str(model_path),
        "bytes": size,
        "mb": round(bytes_to_mb(size), 3),
        "limit_bytes": limits.probabilistic_model_max_bytes,
        "limit_mb": bytes_to_mb(limits.probabilistic_model_max_bytes),
        "within_100mb": ok,
        "within_90mb_submission_budget": submission_fit,
        "size_gate_active": bool(gated),
        "passed": ok if not gated else (ok and submission_fit),
    }
    if gated and not result["passed"]:
        logger.error(
            "Post-24h model size gate FAILED: %s is %.2f MB (need ≤ %.0f MB for zip budget)",
            model_path,
            result["mb"],
            bytes_to_mb(limits.submission_max_bytes),
        )
    return result


def build_submission_archive(
    sources: Sequence[Path],
    archive_path: Path,
    *,
    limits: HardLimits = DEFAULT_LIMITS,
    compression: str = "gz",
) -> Dict[str, Any]:
    """Zip/tar submission artifacts and enforce the 90 MB hard limit (+10 MB buffer).

    Training data is excluded by caller — only code + final model should be listed.
    """
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()

    if compression == "zip" or archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for src in sources:
                src = Path(src)
                if not src.exists():
                    continue
                if src.is_file():
                    zf.write(src, arcname=src.name)
                else:
                    for f in src.rglob("*"):
                        if f.is_file():
                            zf.write(f, arcname=str(f.relative_to(src.parent)))
    else:
        mode = "w:gz" if compression in ("gz", "tgz") else "w"
        with tarfile.open(archive_path, mode) as tf:
            for src in sources:
                src = Path(src)
                if src.exists():
                    tf.add(src, arcname=src.name)

    size = path_size_bytes(archive_path)
    headroom = limits.probabilistic_model_max_bytes - size
    passed = size <= limits.submission_max_bytes
    report = {
        "archive": str(archive_path),
        "bytes": size,
        "mb": round(bytes_to_mb(size), 3),
        "hard_limit_mb": bytes_to_mb(limits.submission_max_bytes),
        "ceiling_mb": bytes_to_mb(limits.probabilistic_model_max_bytes),
        "patch_buffer_remaining_mb": round(bytes_to_mb(max(0, headroom)), 3),
        "passed_90mb": passed,
        "within_100mb_ceiling": size <= limits.probabilistic_model_max_bytes,
        "sources": [str(s) for s in sources],
    }
    if not passed:
        logger.error(
            "Submission archive %.2f MB exceeds 90 MB hard limit (100 MB ceiling, 10 MB patch buffer)",
            report["mb"],
        )
    return report


class TrainingSizeGate:
    """Track wall-clock training; enforce model size only after 24 hours.

    Does NOT limit training dataset size or intermediate checkpoint growth.
    """

    def __init__(self, limits: HardLimits = DEFAULT_LIMITS):
        self.limits = limits
        self.started_at: Optional[float] = None
        self.last_check: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        self.started_at = time.time()

    def elapsed_hours(self) -> float:
        if self.started_at is None:
            return 0.0
        return (time.time() - self.started_at) / 3600.0

    def check(self, model_path: Path) -> Dict[str, Any]:
        hours = self.elapsed_hours()
        self.last_check = check_model_size(
            model_path,
            limits=self.limits,
            after_training_hours=hours,
        )
        self.last_check["training_hours"] = round(hours, 4)
        self.last_check["training_data_capped"] = False
        self.last_check["checkpoint_growth_capped"] = False
        return self.last_check


def assert_turn_budget(
    budget: TurnComputeBudget,
    *,
    write_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Finalize a turn: remaining work is observable; optionally write audit to disk."""
    budget.force_observable()
    snap = budget.snapshot()
    if write_path is not None:
        write_path = Path(write_path)
        write_path.parent.mkdir(parents=True, exist_ok=True)
        write_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        snap["disk_path"] = str(write_path)
    return snap


def limits_manifest(limits: HardLimits = DEFAULT_LIMITS) -> Dict[str, Any]:
    """JSON-serializable manifest for experiment / submission reports."""
    d = limits.to_dict()
    d["notes"] = {
        "probabilistic_model": "Hard ≤ 100 MB",
        "submission_zip": "Hard ≤ 90 MB (10 MB last-minute patch buffer under 100 MB)",
        "subprocess_flops": "Hard ≤ 42 per turn before disk-observable handoff",
        "planning_bank": "Hard planning ceiling $50,000",
        "training_data": "NOT size-capped",
        "training_checkpoints": "NOT size-capped during training",
        "final_model_gate": "After 24h training, final weights must fit submission budget",
        "strategy_window": "< 14 days to discover winning strategy (season still 30 days)",
        "submission_window": "Day 29, 5 turns to zip and submit",
        "hardware": "42 CPUs + 3 GPUs NVLink single pipeline (no inforom)",
        "deterministic_code": "Must fit inside the same 90 MB submission package",
    }
    return d
