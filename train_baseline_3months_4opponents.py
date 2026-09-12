"""Train & Benchmark Tripartite Baseline over 3 Months & 4 Opponents.

Runs 3 full 30-day seasons (2,160 turns per opponent x 2 seat positions = 17,280 match steps)
against 4 reference opponents:
  1. fallow_finn    (Tier 0 - baseline reward floor)
  2. wheat_walter   (Tier 1 - early wheat mono-rusher)
  3. rotation_rosa  (Tier 2 - crop rotation with 4 workers)
  4. homestead_hana (Tier 3 - multi-quadrant staple mix with 8 workers)

Indexes all state embeddings, tactical responses, and market outcomes into
Strategy 3's QuantizedVectorMemoryBank (INT8 quantization, <90 MB payload).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure proper path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
RVQ_DIR = Path(__file__).resolve().parent
if str(RVQ_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval import (
    _make_kaggle_env,
    _normalize_states,
    compare_episode_outcome,
    discover_reference_opponents,
    load_kaggle_agent_policy,
    parse_observation,
)
from hard_limits import (
    MAX_SUBPROCESS_FLOPS_PER_TURN,
    PROBABILISTIC_MODEL_MAX_BYTES,
    SUBMISSION_MAX_BYTES,
)
from reasoning_vs_questioning.agents import (
    TripartiteReasoningAgent,
    build_opponent_kmap_mask,
)
from vector_memory_bank import (
    DEFAULT_EMBEDDING_DIM,
    QuantizedVectorMemoryBank,
    StateVectorEncoder,
)

logger = logging.getLogger("baseline_trainer")


TARGET_OPPONENTS = [
    "fallow_finn",
    "wheat_walter",
    "rotation_rosa",
    "homestead_hana",
]


class BaselineTrainer:
    """Trainer orchestrating multi-month baseline training and vector memory bank ingestion."""

    def __init__(
        self,
        output_dir: Path,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        seasons_per_opp: int = 3,  # 3 months = 3 seasons of 30 days
        turns_per_season: int = 720,  # 30 days * 24 turns
        base_seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.seasons_per_opp = seasons_per_opp
        self.turns_per_season = turns_per_season
        self.base_seed = base_seed

        self.vector_bank = QuantizedVectorMemoryBank(dim=embedding_dim)
        self.encoder = StateVectorEncoder(dim=embedding_dim)
        self.kmap_mask = build_opponent_kmap_mask(dim=embedding_dim)

    def load_opponents(self, opponents_dir: Optional[Path] = None) -> Dict[str, Callable]:
        opp_dir = opponents_dir or (ROOT_DIR / "opponents")
        all_opps = dict(discover_reference_opponents(opp_dir))
        selected = {}
        for name in TARGET_OPPONENTS:
            if name in all_opps:
                selected[name] = all_opps[name]
            else:
                opp_file = opp_dir / f"{name}.py"
                if opp_file.exists():
                    selected[name] = load_kaggle_agent_policy(opp_file)
                else:
                    raise FileNotFoundError(f"Cannot find opponent: {name} in {opp_dir}")
        return selected

    def run_match_season(
        self,
        agent: TripartiteReasoningAgent,
        opp_fn: Callable,
        opp_name: str,
        season_idx: int,
        agent_seat: int = 0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Execute one 30-day (720-step) competitive season, indexing states into vector bank."""
        env = _make_kaggle_env(max_steps=self.turns_per_season, seed=seed, turns_per_day=24)
        states = _normalize_states(env.reset())
        obs_p0 = parse_observation(states[0], player_id=0)
        obs_p1 = parse_observation(states[1], player_id=1)

        agent.reset()
        steps = 0
        done = False
        transitions_indexed = 0

        prev_money_agent = 3000.0

        while not done and steps < self.turns_per_season:
            obs_agent = obs_p0 if agent_seat == 0 else obs_p1
            obs_opp = obs_p1 if agent_seat == 0 else obs_p0

            # Act via Tripartite decision engine
            act_agent = agent.act(obs_agent)
            act_opp = opp_fn(obs_opp)

            actions = [act_agent, act_opp] if agent_seat == 0 else [act_opp, act_agent]
            states = _normalize_states(env.step(actions))
            obs_p0 = parse_observation(states[0], player_id=0)
            obs_p1 = parse_observation(states[1], player_id=1)

            obs_agent_next = obs_p0 if agent_seat == 0 else obs_p1
            cur_money_agent = float(
                (obs_agent_next.get("farms", []) or [{}])[agent_seat].get("money", 0.0) or 0.0
            )
            reward = (cur_money_agent - prev_money_agent) / 100.0
            prev_money_agent = cur_money_agent

            # Index daily boundaries and high-yield strategic moves into QuantizedVectorMemoryBank
            if steps % 24 == 0 or reward > 0.5:
                tag = f"{opp_name}_season{season_idx}_d{steps//24}_s{steps%24}"
                self.vector_bank.add_entry(
                    obs=obs_agent,
                    action=act_agent,
                    strategic_value=float(reward),
                    archetype_tag=opp_name,
                    dialogue_response=f"Strategic state transition vs {opp_name} in season {season_idx}",
                )
                transitions_indexed += 1

            status = states[0].get("status", "ACTIVE")
            done = status in ("DONE", "TIMEOUT", "INVALID")
            steps += 1

        w0, w1, ties = compare_episode_outcome(obs_p0, obs_p1)
        agent_win = bool(w0 if agent_seat == 0 else w1)
        opp_win = bool(w1 if agent_seat == 0 else w0)
        tie = bool(ties)

        agent_final_money = float(
            (obs_p0 if agent_seat == 0 else obs_p1).get("farms", [])[agent_seat].get("money", 0.0)
        )
        opp_final_money = float(
            (obs_p1 if agent_seat == 0 else obs_p0).get("farms", [])[1 - agent_seat].get("money", 0.0)
        )

        return {
            "opponent": opp_name,
            "season": season_idx + 1,
            "agent_seat": agent_seat,
            "agent_win": agent_win,
            "opp_win": opp_win,
            "tie": tie,
            "agent_money": agent_final_money,
            "opp_money": opp_final_money,
            "money_margin": agent_final_money - opp_final_money,
            "steps": steps,
            "transitions_indexed": transitions_indexed,
        }

    def train_baseline(self) -> Dict[str, Any]:
        """Execute baseline training over 3 months x 4 opponents across both seats."""
        opponents = self.load_opponents()
        print(f"=== Starting Baseline Training: 3 Months (3 Seasons x 720 turns) across 4 Opponents ===")
        print(f"Opponents: {list(opponents.keys())}\n")

        # Initialize base Tripartite agent with shared vector bank
        agent = TripartiteReasoningAgent(
            vector_bank=self.vector_bank,
            kmap_mask=self.kmap_mask,
            act_all_hours=True,
            target_hands=4,
        )

        all_match_results: List[Dict[str, Any]] = []
        summary_by_opp: Dict[str, Any] = {}

        total_steps = 0
        t0_train = time.time()

        for opp_name, opp_fn in opponents.items():
            print(f"--- Training vs {opp_name} (3 Months / 3 Seasons) ---")
            opp_wins = 0
            opp_losses = 0
            opp_ties = 0
            opp_agent_money: List[float] = []
            opp_money_list: List[float] = []

            for season in range(self.seasons_per_opp):
                # Play both as Seat 0 and Seat 1 for fairness and comprehensive experience
                for seat in (0, 1):
                    seed = self.base_seed + (len(all_match_results) * 17)
                    res = self.run_match_season(
                        agent=agent,
                        opp_fn=opp_fn,
                        opp_name=opp_name,
                        season_idx=season,
                        agent_seat=seat,
                        seed=seed,
                    )
                    all_match_results.append(res)
                    total_steps += res["steps"]

                    if res["agent_win"]:
                        opp_wins += 1
                    elif res["opp_win"]:
                        opp_losses += 1
                    else:
                        opp_ties += 1

                    opp_agent_money.append(res["agent_money"])
                    opp_money_list.append(res["opp_money"])

                    print(
                        f"  Month {season+1} (Seat P{seat}): "
                        f"{'WIN' if res['agent_win'] else ('TIE' if res['tie'] else 'LOSS')} | "
                        f"Agent: ${res['agent_money']:.0f} vs Opp: ${res['opp_money']:.0f} "
                        f"(Indexed +{res['transitions_indexed']} vectors)"
                    )

            n_matches = len(opp_agent_money)
            summary_by_opp[opp_name] = {
                "matches": n_matches,
                "wins": opp_wins,
                "losses": opp_losses,
                "ties": opp_ties,
                "win_rate": opp_wins / max(1, n_matches),
                "avg_agent_money": float(np.mean(opp_agent_money)),
                "avg_opp_money": float(np.mean(opp_money_list)),
                "avg_margin": float(np.mean(np.array(opp_agent_money) - np.array(opp_money_list))),
            }
            print(
                f"-> Summary vs {opp_name}: Win Rate={summary_by_opp[opp_name]['win_rate']:.1%}, "
                f"Avg Money: ${summary_by_opp[opp_name]['avg_agent_money']:.0f} vs ${summary_by_opp[opp_name]['avg_opp_money']:.0f}\n"
            )

        t_elapsed = time.time() - t0_train

        # Save trained QuantizedVectorMemoryBank
        bank_path = self.output_dir / "policy_vector_bank.npz"
        self.vector_bank.save(bank_path)
        bank_size_bytes = os.path.getsize(bank_path)
        bank_size_mb = bank_size_bytes / (1024 * 1024)

        # Retrieval latency benchmark
        dummy_query_obs = {"day": 10, "hour": 12, "farms": [{"money": 8000.0}, {"money": 6000.0}]}
        t_q0 = time.perf_counter()
        for _ in range(500):
            _ = self.vector_bank.query_top_k(dummy_query_obs, k=3)
        t_q1 = time.perf_counter()
        query_latency_ms = ((t_q1 - t_q0) / 500) * 1000

        total_wins = sum(s["wins"] for s in summary_by_opp.values())
        total_matches = sum(s["matches"] for s in summary_by_opp.values())
        overall_win_rate = total_wins / max(1, total_matches)

        report = {
            "training_summary": {
                "opponents_trained": TARGET_OPPONENTS,
                "seasons_per_opponent": self.seasons_per_opp,
                "turns_per_season": self.turns_per_season,
                "total_simulated_steps": total_steps,
                "wall_clock_time_sec": round(t_elapsed, 2),
                "overall_win_rate": round(overall_win_rate, 4),
                "total_matches": total_matches,
            },
            "opponent_breakdown": summary_by_opp,
            "vector_memory_bank": {
                "total_vectors_indexed": len(self.vector_bank),
                "embedding_dim": self.embedding_dim,
                "payload_size_bytes": bank_size_bytes,
                "payload_size_mb": round(bank_size_mb, 3),
                "payload_limit_mb": 90.0,
                "under_payload_ceiling": bank_size_bytes <= SUBMISSION_MAX_BYTES,
                "avg_query_latency_ms": round(query_latency_ms, 3),
            },
            "hard_limits_compliance": {
                "max_flops_per_turn": MAX_SUBPROCESS_FLOPS_PER_TURN,
                "vector_bank_under_90mb": bank_size_bytes <= SUBMISSION_MAX_BYTES,
                "latency_under_5ms": query_latency_ms < 5.0,
            },
            "match_history": all_match_results,
        }

        metrics_path = self.output_dir / "baseline_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("=== Baseline Training Complete ===")
        print(f"Overall Win Rate: {overall_win_rate:.1%}")
        print(f"Total Vectors Indexed: {len(self.vector_bank)}")
        print(f"Vector Bank Size: {bank_size_mb:.2f} MB (Hard limit: 90 MB)")
        print(f"Query Latency: {query_latency_ms:.3f} ms (Target: < 5 ms)")
        print(f"Saved artifacts to {self.output_dir}")

        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline agent over 3 months and 4 opponents")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "experiments" / "baseline_3months_4opponents",
        help="Directory to save model artifacts and metrics",
    )
    parser.add_argument("--seasons", type=int, default=3, help="Number of 30-day seasons (months) per opponent")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    trainer = BaselineTrainer(
        output_dir=args.output_dir,
        seasons_per_opp=args.seasons,
        base_seed=args.seed,
    )
    trainer.train_baseline()


if __name__ == "__main__":
    main()
