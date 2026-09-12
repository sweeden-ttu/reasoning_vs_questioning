"""Benchmark Replay-Guided Statistical QKD RL Agent against Ladder Reference Opponents."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Path resolution
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
from reasoning_vs_questioning.agents.qkd_replay_rl_agent import (
    QKDReplayRLAgent,
    agent as qkd_replay_agent_fn,
)

logger = logging.getLogger("eval_qkd_replay")


def run_match(
    agent_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    opp_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    agent_seat: int = 0,
    max_steps: int = 720,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a single 720-step match between agent and opponent."""
    env = _make_kaggle_env(max_steps=max_steps, seed=seed, turns_per_day=24)
    states = _normalize_states(env.reset())
    obs_p0 = parse_observation(states[0], player_id=0)
    obs_p1 = parse_observation(states[1], player_id=1)

    steps = 0
    done = False
    turn_times: List[float] = []

    while not done and steps < max_steps:
        obs_agent = obs_p0 if agent_seat == 0 else obs_p1
        obs_opp = obs_p1 if agent_seat == 0 else obs_p0

        t0 = time.perf_counter()
        act_agent = agent_fn(obs_agent)
        t_agent = (time.perf_counter() - t0) * 1000.0
        turn_times.append(t_agent)

        act_opp = opp_fn(obs_opp)

        actions = [act_agent, act_opp] if agent_seat == 0 else [act_opp, act_agent]
        states = _normalize_states(env.step(actions))
        obs_p0 = parse_observation(states[0], player_id=0)
        obs_p1 = parse_observation(states[1], player_id=1)

        status = states[0].get("status", "ACTIVE")
        done = status in ("DONE", "TIMEOUT", "INVALID", "ERROR")
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
        "agent_win": agent_win,
        "opp_win": opp_win,
        "tie": tie,
        "agent_money": agent_final_money,
        "opp_money": opp_final_money,
        "money_margin": agent_final_money - opp_final_money,
        "steps": steps,
        "mean_turn_ms": float(np.mean(turn_times)) if turn_times else 0.0,
        "max_turn_ms": float(np.max(turn_times)) if turn_times else 0.0,
    }


def evaluate_against_ladder(
    target_opponents: Optional[List[str]] = None,
    matches_per_seat: int = 1,
    base_seed: int = 42,
) -> Dict[str, Any]:
    import numpy as np

    opponents_dict = dict(discover_reference_opponents(ROOT_DIR / "opponents"))
    selected_opps = target_opponents or ["fallow_finn", "wheat_walter", "rotation_rosa"]

    print("================================================================================")
    print(" Benchmarking Replay-Guided Statistical QKD RL Agent against Reference Ladder")
    print("================================================================================")

    results: Dict[str, Any] = {}
    total_wins, total_losses, total_ties = 0, 0, 0

    for opp_name in selected_opps:
        if opp_name not in opponents_dict:
            opp_file = ROOT_DIR / "opponents" / f"{opp_name}.py"
            if opp_file.exists():
                opp_fn = load_kaggle_agent_policy(opp_file)
            else:
                print(f"Skipping {opp_name} (file not found)")
                continue
        else:
            opp_fn = opponents_dict[opp_name]

        opp_wins, opp_losses, opp_ties = 0, 0, 0
        agent_scores, opp_scores, turn_latencies = [], [], []

        for m_idx in range(matches_per_seat):
            for seat in (0, 1):
                seed = base_seed + len(results) * 100 + m_idx * 10 + seat
                res = run_match(
                    agent_fn=qkd_replay_agent_fn,
                    opp_fn=opp_fn,
                    agent_seat=seat,
                    seed=seed,
                )
                if res["agent_win"]:
                    opp_wins += 1
                    total_wins += 1
                elif res["opp_win"]:
                    opp_losses += 1
                    total_losses += 1
                else:
                    opp_ties += 1
                    total_ties += 1

                agent_scores.append(res["agent_money"])
                opp_scores.append(res["opp_money"])
                turn_latencies.append(res["mean_turn_ms"])

                outcome = "WIN" if res["agent_win"] else ("TIE" if res["tie"] else "LOSS")
                print(
                    f"vs {opp_name:<15} [Seat P{seat} | Match {m_idx+1}]: {outcome:<4} | "
                    f"Agent: ${res['agent_money']:>8,.0f} | Opp: ${res['opp_money']:>8,.0f} | "
                    f"Margin: ${res['money_margin']:>8,.0f} | Turn: {res['mean_turn_ms']:.2f}ms"
                )

        results[opp_name] = {
            "wins": opp_wins,
            "losses": opp_losses,
            "ties": opp_ties,
            "win_rate": opp_wins / max(1, opp_wins + opp_losses + opp_ties),
            "avg_agent_money": float(np.mean(agent_scores)),
            "avg_opp_money": float(np.mean(opp_scores)),
            "avg_turn_ms": float(np.mean(turn_latencies)),
        }

    total_games = total_wins + total_losses + total_ties
    print("\n--------------------------------------------------------------------------------")
    print(f"Summary: {total_wins} Wins / {total_losses} Losses / {total_ties} Ties ({total_wins/max(1, total_games)*100:.1f}% Win Rate)")
    print("--------------------------------------------------------------------------------")
    return results


if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--opponents", nargs="+", default=["fallow_finn", "wheat_walter", "rotation_rosa"])
    parser.add_argument("--matches", type=int, default=1)
    args = parser.parse_args()

    evaluate_against_ladder(target_opponents=args.opponents, matches_per_seat=args.matches)
