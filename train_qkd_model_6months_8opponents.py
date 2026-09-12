"""Train & Benchmark Unified QKD Vector Model over 6 Months & 8 Opponents.

Trains the tripartite QKD (Questions + K-Map Observations + Decisions) model across
6 full 30-day seasons (4,320 turns per opp x 2 seat positions = 8,640 turns per opp)
against 8 reference ladder opponents:
  1. fallow_finn    (Tier 0 - baseline reward floor)
  2. wheat_walter   (Tier 1 - early wheat mono-rusher)
  3. rotation_rosa  (Tier 2 - crop rotation with 4 workers)
  4. homestead_hana (Tier 3 - multi-quadrant staple mix with 8 workers)
  5. melon_mateo    (Tier 4 - premium melon fertilization & glut control)
  6. rancher_rita   (Tier 5 - livestock herd with wheat supply chain)
  7. broker_bea     (Tier 6 - meta line with cash-position timing)
  8. closer_cleo    (Tier 9 - meta line with in-place sell queue preservation)

Total simulated match steps: 8 opps x 6 seasons x 2 seats x 720 = 69,120 turns.
Serializes the full QKD model (Q_Bank, K_Bank, D_Bank) into qkd_model.npz (<= 90 MB).
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
from qkd_vector_model import QKDVectorModel, QKDQueryInsight
from reasoning_vs_questioning.agents import TripartiteReasoningAgent

logger = logging.getLogger("qkd_trainer")

TARGET_OPPONENTS_8 = [
    "fallow_finn",
    "wheat_walter",
    "rotation_rosa",
    "homestead_hana",
    "melon_mateo",
    "rancher_rita",
    "broker_bea",
    "closer_cleo",
]


class QKDBaselineTrainer:
    """Trainer orchestrating 6-month x 8-opponent QKD model training."""

    def __init__(
        self,
        output_dir: Path,
        embedding_dim: int = 256,
        seasons_per_opp: int = 6,  # 6 months = 6 seasons of 30 days
        turns_per_season: int = 720,  # 30 days * 24 turns
        base_seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.seasons_per_opp = seasons_per_opp
        self.turns_per_season = turns_per_season
        self.base_seed = base_seed

        self.qkd_model = QKDVectorModel(dim=embedding_dim)

    def load_opponents(self, opponents_dir: Optional[Path] = None) -> Dict[str, Callable]:
        opp_dir = opponents_dir or (ROOT_DIR / "opponents")
        all_opps = dict(discover_reference_opponents(opp_dir))
        selected = {}
        for name in TARGET_OPPONENTS_8:
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
        """Execute one 30-day (720-step) season, updating Q, K, D stores."""
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

            # 1. Query QKD insight
            qkd_insight = self.qkd_model.query_qkd(obs_agent)

            # 2. Act via Tripartite Agent
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

            # 3. Add experiences into Q, K, D stores at day boundaries or significant yields
            if steps % 24 == 0 or reward > 0.5:
                q_text = f"What is the expected strategic counter for {opp_name} at day {steps//24}?"
                self.qkd_model.add_experience(
                    obs=obs_agent,
                    action=act_agent,
                    reward=float(reward),
                    opp_name=opp_name,
                    question_text=q_text,
                    hypothesis=f"posture_{opp_name}_m{season_idx+1}_d{steps//24}",
                    confidence=min(1.0, max(0.5, 0.5 + reward * 0.1)),
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
        """Execute baseline training over 6 months x 8 opponents across both seats."""
        opponents = self.load_opponents()
        print("=== Starting QKD Model Training: 6 Months (6 Seasons x 720 turns) across 8 Opponents ===")
        print(f"Opponents ({len(opponents)}): {list(opponents.keys())}\n")

        agent = TripartiteReasoningAgent(
            vector_bank=self.qkd_model.k_store,
            kmap_mask=self.qkd_model.kmap_mask,
            act_all_hours=True,
            target_hands=4,
        )

        all_match_results: List[Dict[str, Any]] = []
        summary_by_opp: Dict[str, Any] = {}

        total_steps = 0
        t0_train = time.time()

        for opp_name, opp_fn in opponents.items():
            print(f"--- Training vs {opp_name} (6 Months / 6 Seasons) ---")
            opp_wins = 0
            opp_losses = 0
            opp_ties = 0
            opp_agent_money: List[float] = []
            opp_money_list: List[float] = []

            for season in range(self.seasons_per_opp):
                for seat in (0, 1):
                    seed = self.base_seed + (len(all_match_results) * 23)
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
                        f"(QKD Indexed +{res['transitions_indexed']} tuples)"
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

        # Save trained QKDVectorModel
        model_path = self.output_dir / "qkd_model.npz"
        self.qkd_model.save(model_path)
        model_size_bytes = os.path.getsize(model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)

        # Retrieval latency benchmark
        dummy_obs = {"day": 15, "hour": 8, "farms": [{"money": 15000.0}, {"money": 12000.0}]}
        t_q0 = time.perf_counter()
        for _ in range(500):
            _ = self.qkd_model.query_qkd(dummy_obs)
        t_q1 = time.perf_counter()
        query_latency_ms = ((t_q1 - t_q0) / 500) * 1000

        total_wins = sum(s["wins"] for s in summary_by_opp.values())
        total_matches = sum(s["matches"] for s in summary_by_opp.values())
        overall_win_rate = total_wins / max(1, total_matches)

        report = {
            "training_summary": {
                "architecture": "QKD_Tripartite_Vector_Model (Questions + K-Map Observations + Decisions)",
                "opponents_trained": TARGET_OPPONENTS_8,
                "seasons_per_opponent": self.seasons_per_opp,
                "turns_per_season": self.turns_per_season,
                "total_simulated_steps": total_steps,
                "wall_clock_time_sec": round(t_elapsed, 2),
                "overall_win_rate": round(overall_win_rate, 4),
                "total_matches": total_matches,
            },
            "opponent_breakdown": summary_by_opp,
            "qkd_vector_model": {
                "total_vectors_indexed": self.qkd_model.total_vectors(),
                "q_store_vectors": len(self.qkd_model.q_store),
                "k_store_vectors": len(self.qkd_model.k_store),
                "d_store_vectors": len(self.qkd_model.d_store),
                "embedding_dim": self.embedding_dim,
                "payload_size_bytes": model_size_bytes,
                "payload_size_mb": round(model_size_mb, 3),
                "payload_limit_mb": 90.0,
                "under_payload_ceiling": model_size_bytes <= SUBMISSION_MAX_BYTES,
                "avg_query_latency_ms": round(query_latency_ms, 3),
            },
            "hard_limits_compliance": {
                "max_flops_per_turn": MAX_SUBPROCESS_FLOPS_PER_TURN,
                "vector_bank_under_90mb": model_size_bytes <= SUBMISSION_MAX_BYTES,
                "latency_under_5ms": query_latency_ms < 5.0,
            },
            "match_history": all_match_results,
        }

        metrics_path = self.output_dir / "baseline_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("=== 6-Month 8-Opponent QKD Model Training Complete ===")
        print(f"Overall Win Rate: {overall_win_rate:.1%}")
        print(f"Total Vectors Indexed across Q, K, D: {self.qkd_model.total_vectors()}")
        print(f"QKD Model Size: {model_size_mb:.2f} MB (Hard limit: 90 MB)")
        print(f"Query Latency: {query_latency_ms:.3f} ms (Target: < 5 ms)")
        print(f"Saved artifacts to {self.output_dir}")

        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QKD model over 6 months and 8 opponents")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "experiments" / "qkd_model_6months_8opponents",
        help="Directory to save QKD model and metrics",
    )
    parser.add_argument("--seasons", type=int, default=6, help="Number of 30-day seasons (months) per opponent")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    trainer = QKDBaselineTrainer(
        output_dir=args.output_dir,
        seasons_per_opp=args.seasons,
        base_seed=args.seed,
    )
    trainer.train_baseline()


if __name__ == "__main__":
    main()
