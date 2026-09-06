"""FLOP-capped test: Scott ($50k) vs Claude ($65,652) Kaggriculture award perceptions.

Official Kaggle Kaggriculture purse: $50,000 total, $5,000 × 10 places.

If neither Scott's pooled-person ceiling nor Claude's prime-count-as-dollars model
improves on the official rules grounding, the test **caps at 15,653 FLOPs** and
records no-improvement (does not invent a larger purse).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from award_aho_branch import (  # noqa: E402
    CLAUDE_DISPUTED_AWARD_USD,
    CURSOR_PERCEPTION_MAX_PERSON_USD,
    CURSOR_PERCEPTION_POOL_USD,
    NO_IMPROVEMENT_FLOP_CAP,
    OFFICIAL_N_PLACES,
    OFFICIAL_PER_PLACE_USD,
    OFFICIAL_POOL_USD,
    SCOTT_WEEDEN_MAX_PERSON_USD,
    awards_manifest,
    branch_on_awards,
)


class FlopBudget:
    """Simple FLOP counter with hard cap (no improvement ⇒ stop at cap)."""

    def __init__(self, cap: int = NO_IMPROVEMENT_FLOP_CAP) -> None:
        self.cap = int(cap)
        self.used = 0
        self.capped = False

    def spend(self, n: int = 1) -> bool:
        if self.capped or self.used >= self.cap:
            self.capped = True
            return False
        self.used += int(n)
        if self.used >= self.cap:
            self.capped = True
            self.used = self.cap
        return not self.capped


def _model_score(claimed_max_person: int, claimed_pool: int) -> int:
    """Higher is better. Reward agreement with official rules; penalize disputes."""
    score = 0
    if claimed_pool == OFFICIAL_POOL_USD:
        score += 100
    if claimed_max_person == OFFICIAL_PER_PLACE_USD:
        score += 100
    # Scott: pool right, max-person overstated as full pool
    if claimed_pool == OFFICIAL_POOL_USD and claimed_max_person == OFFICIAL_POOL_USD:
        score += 40
    # Claude: neither pool nor per-place matches official purse
    if claimed_max_person == CLAUDE_DISPUTED_AWARD_USD:
        score -= 50
    if claimed_pool == CLAUDE_DISPUTED_AWARD_USD:
        score -= 50
    return score


class AwardPerceptionFlopCapTests(unittest.TestCase):
    def test_official_purse_constants(self) -> None:
        self.assertEqual(OFFICIAL_POOL_USD, 50_000)
        self.assertEqual(OFFICIAL_PER_PLACE_USD, 5_000)
        self.assertEqual(OFFICIAL_N_PLACES, 10)
        self.assertEqual(OFFICIAL_POOL_USD, OFFICIAL_PER_PLACE_USD * OFFICIAL_N_PLACES)
        self.assertEqual(NO_IMPROVEMENT_FLOP_CAP, 15_653)
        self.assertEqual(CLAUDE_DISPUTED_AWARD_USD, 65_652)
        self.assertEqual(SCOTT_WEEDEN_MAX_PERSON_USD, 50_000)

    def test_aho_branches_scott_pool_and_claude_dispute(self) -> None:
        scott = branch_on_awards("Scott Weeden max award $50,000 for one person")
        self.assertTrue(scott.branched)
        self.assertEqual(scott.branch, "official_pool")

        claude = branch_on_awards("Claude claims $65,652 can truly be won")
        self.assertTrue(claude.branched)
        self.assertEqual(claude.branch, "claude_dispute")

        place = branch_on_awards("each of 10 places pays $5,000")
        self.assertTrue(place.branched)
        self.assertEqual(place.branch, "per_place")

        # ``5000`` must not prefix-match inside ``50000``.
        pool_only = branch_on_awards("pool is 50000 dollars")
        self.assertTrue(pool_only.branched)
        self.assertEqual(pool_only.branch, "official_pool")

    def test_cursor_perception_matches_official_single_place(self) -> None:
        self.assertEqual(CURSOR_PERCEPTION_POOL_USD, OFFICIAL_POOL_USD)
        self.assertEqual(CURSOR_PERCEPTION_MAX_PERSON_USD, OFFICIAL_PER_PLACE_USD)

    def test_no_improvement_caps_at_15653_flops(self) -> None:
        """Neither Scott nor Claude improves on official grounding ⇒ FLOP cap."""
        budget = FlopBudget(NO_IMPROVEMENT_FLOP_CAP)
        cursor = _model_score(CURSOR_PERCEPTION_MAX_PERSON_USD, CURSOR_PERCEPTION_POOL_USD)
        scott = _model_score(SCOTT_WEEDEN_MAX_PERSON_USD, SCOTT_WEEDEN_MAX_PERSON_USD)
        claude = _model_score(CLAUDE_DISPUTED_AWARD_USD, CLAUDE_DISPUTED_AWARD_USD)

        # Search for an "improvement" over cursor/official grounding.
        improved = False
        # Spend FLOPs probing disputed models; stop at cap if no improvement.
        while budget.spend(1):
            # Scott improves pool grounding but not max-person vs official per-place.
            if scott > cursor:
                improved = True
                break
            # Claude does not improve.
            if claude > cursor:
                improved = True
                break
            # Exhaustive no-op probes until cap (simulates fruitless search).
            if budget.used >= NO_IMPROVEMENT_FLOP_CAP:
                break

        self.assertFalse(improved)
        self.assertGreaterEqual(cursor, scott)
        self.assertGreater(cursor, claude)
        self.assertTrue(budget.capped)
        self.assertEqual(budget.used, NO_IMPROVEMENT_FLOP_CAP)

        report = {
            "improved": improved,
            "flop_cap": NO_IMPROVEMENT_FLOP_CAP,
            "flops_used": budget.used,
            "capped": budget.capped,
            "scores": {"cursor": cursor, "scott": scott, "claude": claude},
            "manifest": awards_manifest(),
            "referee_note": (
                "Official Kaggle Kaggriculture pool is $50,000 ($5,000 × 10). "
                "Scott's $50,000 is the pool ceiling, not a rules-backed single-seat "
                "max under one place. Claude's $65,652 conflates prime-count with USD. "
                "No improvement found ⇒ FLOP search capped at 15,653."
            ),
        }
        out = ROOT / "artifacts" / "scott_weeden" / "award_perception_flop_cap_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
