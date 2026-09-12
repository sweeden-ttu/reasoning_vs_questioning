"""Replay Trajectory Ingestion and QKD Statistical Calibration Engine.

Parses tournament match replays from `replays/`, identifies winning agent traces,
extracts compressed base action sequences, and fits empirical sample distributions
over the QKD Statistical Question Bank.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
RVQ_DIR = Path(__file__).resolve().parent
if str(RVQ_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qkd_statistical_questions import (
    CANONICAL_QKD_QUESTIONS,
    QKDStatisticalQuestionBank,
)

logger = logging.getLogger("replay_qkd_extractor")


class ReplayQKDExtractor:
    """Ingests match replay JSONs and produces compressed action policies & QKD statistics."""

    def __init__(self, replays_dir: Optional[Path | str] = None) -> None:
        self.replays_dir = Path(replays_dir) if replays_dir else ROOT_DIR / "replays"
        self.question_bank = QKDStatisticalQuestionBank()

    def parse_replay_file(self, file_path: Path | str) -> Optional[Dict[str, Any]]:
        """Parse a single replay JSON and extract winner, actions, and observations."""
        p = Path(file_path)
        if not p.exists():
            return None

        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            logger.warning("Failed to load %s: %s", p, e)
            return None

        steps = data.get("steps", [])
        if not steps or len(steps) < 2:
            return None

        rewards = data.get("rewards", [0, 0])
        p0_reward = rewards[0] if rewards and len(rewards) > 0 and rewards[0] is not None else 0.0
        p1_reward = rewards[1] if rewards and len(rewards) > 1 and rewards[1] is not None else 0.0

        winner = 0 if p0_reward >= p1_reward else 1
        winning_score = max(p0_reward, p1_reward)

        observations_winner: List[Dict[str, Any]] = []
        actions_winner: List[Dict[str, Any]] = []
        all_observations: List[Dict[str, Any]] = []

        for step_idx in range(len(steps) - 1):
            for player_idx in (0, 1):
                obs = steps[step_idx][player_idx].get("observation") or {}
                if obs:
                    all_observations.append(obs)

            # Record winning agent's observation and the action generated in response to it
            win_obs = steps[step_idx][winner].get("observation") or {}
            win_action = steps[step_idx + 1][winner].get("action") or {"farmer": ["PASS"], "hands": [], "market": []}

            if win_obs:
                observations_winner.append(win_obs)
            actions_winner.append(win_action)

        # Include final observation
        final_obs_0 = steps[-1][0].get("observation") or {}
        final_obs_1 = steps[-1][1].get("observation") or {}
        if final_obs_0:
            all_observations.append(final_obs_0)
        if final_obs_1:
            all_observations.append(final_obs_1)

        return {
            "file": p.name,
            "episode_id": data.get("info", {}).get("EpisodeId", p.stem),
            "total_steps": len(steps),
            "winner": winner,
            "winning_score": winning_score,
            "actions_winner": actions_winner,
            "observations_winner": observations_winner,
            "all_observations": all_observations,
        }

    def compress_actions(self, actions: List[Dict[str, Any]]) -> str:
        """Serialize and compress action sequence to a compact Base85 string."""
        raw_json = json.dumps(actions, separators=(",", ":")).encode("utf-8")
        compressed = zlib.compress(raw_json, level=9)
        return base64.b85encode(compressed).decode("ascii")

    @staticmethod
    def decompress_actions(encoded_str: str) -> List[Dict[str, Any]]:
        """Decompress Base85 string back into action dictionary list."""
        if not encoded_str:
            return []
        compressed = base64.b85decode(encoded_str.encode("ascii"))
        raw_json = zlib.decompress(compressed).decode("utf-8")
        return json.loads(raw_json)

    def extract_from_replay_dir(
        self,
        replays_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Ingest all valid replay files, fit QKD distribution, and find highest-scoring route."""
        r_dir = Path(replays_dir) if replays_dir else self.replays_dir
        parsed_replays: List[Dict[str, Any]] = []
        all_match_obs: List[Dict[str, Any]] = []

        best_score = -1.0
        best_replay: Optional[Dict[str, Any]] = None

        for json_file in sorted(r_dir.glob("*.json")):
            res = self.parse_replay_file(json_file)
            if res and res["total_steps"] >= 100:
                parsed_replays.append(res)
                all_match_obs.extend(res["all_observations"])
                if res["winning_score"] > best_score:
                    best_score = res["winning_score"]
                    best_replay = res

        print(f"Loaded {len(parsed_replays)} valid match replays ({len(all_match_obs)} observations).")

        # Fit QKD Question Bank distribution
        ranked_questions = self.question_bank.fit_sample_distribution(all_match_obs)
        summary_table = self.question_bank.summary_table()

        compressed_best_route = ""
        best_file = ""
        if best_replay is not None:
            compressed_best_route = self.compress_actions(best_replay["actions_winner"])
            best_file = best_replay["file"]
            print(f"Top championship replay: {best_file} with score ${best_score:,.0f}")
            print(f"Compressed route payload: {len(compressed_best_route)} chars.")

        return {
            "parsed_count": len(parsed_replays),
            "total_observations": len(all_match_obs),
            "best_file": best_file,
            "best_score": best_score,
            "compressed_best_route": compressed_best_route,
            "summary_table": summary_table,
            "ranked_questions": ranked_questions,
        }


if __name__ == "__main__":
    extractor = ReplayQKDExtractor()
    results = extractor.extract_from_replay_dir()
    print("\n=== QKD Statistical Question Bank - Empirical Distribution from Replays ===")
    for row in results["summary_table"]:
        print(f"Rank {row['rank']:2d} [{row['channel']}] {row['qid']:<25} | std={row['std_deviation']:.4f} | mean={row['mean']:.4f} | range={row['range']}")
