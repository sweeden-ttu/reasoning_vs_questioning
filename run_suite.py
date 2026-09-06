"""Seat-swap harness and 10-experiment suite: Reasoning vs Questioning.

Agent1 (Reasoning) acts on prime hours < 11: {2,3,5,7}.
Agent2 (Questioning) acts on even hours.
Every matchup runs twice with seats swapped. Both start at startingMoney=3000.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Allow running as script from this repository root (or nested copies).
_HERE = Path(__file__).resolve().parent
_CODE_ROOT = _HERE
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from agents.questioning_agent import QuestioningAgent
from agents.reasoning_agent import ReasoningAgent
from environment import create_competitive_env
from hard_limits import limits_manifest
from memory_protocol import (
    PRIME_HOURS_LT_11,
    day_edge_question_stats,
)

logger = logging.getLogger(__name__)

DEFAULT_OUT = _HERE
STARTING_MONEY = 3000


@dataclass
class MatchResult:
    seed: int
    reasoning_seat: int
    questioning_seat: int
    money_p0: float
    money_p1: float
    winner: str  # "reasoning" | "questioning" | "tie"
    reasoning_metrics: Dict[str, Any]
    questioning_metrics: Dict[str, Any]
    schedule_ok: bool
    starting_money_ok: bool
    opening_charity: Optional[Dict[str, Any]] = None
    motive_dialogue: Optional[Dict[str, Any]] = None


def _final_money(obs: Dict[str, Any], player: int) -> float:
    farms = obs.get("farms", []) or []
    if len(farms) > player:
        return float(farms[player].get("money", 0.0) or 0.0)
    return 0.0


def _verify_schedule(agent: Any, expected_hours) -> bool:
    for entry in agent.action_audit:
        hour = int(entry.get("hour", -1))
        # Opening DONATE is pre-loop (hour=-1); skip schedule check.
        if hour < 0:
            continue
        acted = bool(entry.get("acted"))
        legal = hour in expected_hours
        if acted and not legal:
            return False
        if not acted and legal:
            continue
    return True


def run_match(
    seed: int,
    *,
    reasoning_seat: int,
    reasoning_kwargs: Optional[Dict[str, Any]] = None,
    questioning_kwargs: Optional[Dict[str, Any]] = None,
    max_steps: int = 720,
    turns_per_cycle: int = 24,
) -> MatchResult:
    """Play one full episode with Reasoning on ``reasoning_seat`` (0 or 1).

    Before the step loop, Agent1 (Reasoning) donates 888 to Agent2's bank so
    Agent2 can observe and record Agent1's charitable nature.
    """
    r_kw = dict(reasoning_kwargs or {})
    q_kw = dict(questioning_kwargs or {})
    reasoning = ReasoningAgent(**r_kw)
    questioning = QuestioningAgent(**q_kw)
    reasoning.reset()
    questioning.reset()

    env = create_competitive_env(
        use_kaggle=True,
        max_steps=max_steps,
        seed=seed,
        turns_per_cycle=turns_per_cycle,
    )
    obs_p0 = env.reset()
    obs_p1 = env._get_obs(player=1)

    start0 = _final_money(obs_p0, 0)
    start1 = _final_money(obs_p1, 1)
    starting_ok = abs(start0 - STARTING_MONEY) < 1e-6 and abs(start1 - STARTING_MONEY) < 1e-6

    # Agent1's one opening act: donate 888 → Agent2 bank (publicly visible).
    charity = reasoning.offer_opening_charity(env, reasoning_seat)
    obs_p0 = env._get_obs(player=0)
    obs_p1 = env._get_obs(player=1)
    q_seat = 1 - reasoning_seat
    q_obs = env._get_obs(player=q_seat)
    questioning.record_agent1_charity(q_obs, amount=int(charity.get("amount", 888)))
    # Agent2 questions motives; Agent1 answers aloud with fellowship-test (spoken).
    motive_q = questioning.question_agent1_motives(q_obs)
    motive_dialogue = reasoning.answer_motive_question(
        motive_q["question"],
        obs=env._get_obs(player=reasoning_seat),
    )
    questioning.receive_motive_answer(motive_dialogue)
    # Agent1 speaks public/private Kaggle path proof; Agent2 Aho-Corasick trust scan.
    path_proof = reasoning.emit_kaggle_path_proof()
    path_trust = questioning.evaluate_agent1_path_trust(path_proof, day=0)
    # Agent2 states the rules as he sees them; Agent1 adopts private policy (not spoken).
    rules_view = questioning.state_rules_understanding(q_obs)
    private_policy = reasoning.adopt_private_fellowship_policy(agent2_rules_statement=rules_view)

    done = False
    steps = 0
    day29_asked = False
    while not done and steps < max_steps:
        agents = [None, None]
        agents[reasoning_seat] = reasoning
        agents[1 - reasoning_seat] = questioning
        a0 = agents[0].act(obs_p0)
        a1 = agents[1].act(obs_p1)
        # Day-29 closing question from Agent1 (private policy); Agent2 replies; Agent1 judges.
        if not day29_asked:
            r_obs = obs_p0 if reasoning_seat == 0 else obs_p1
            day = int(r_obs.get("day", 0) or 0)
            hour = int(r_obs.get("hour", 0) or 0)
            if day == 29 and hour >= 20 and reasoning.may_act(hour):
                q29 = reasoning.ask_day29_question(r_obs)
                if q29:
                    day29_asked = True
                    q_obs_now = obs_p0 if q_seat == 0 else obs_p1
                    r_money_now = _final_money(r_obs, reasoning_seat)
                    q_money_now = _final_money(q_obs_now, q_seat)
                    agent2_losing = q_money_now < r_money_now
                    # Default: tell deterministic truth when losing → knowledge/CS assumption.
                    reply29 = questioning.answer_day29_question(
                        q29, q_obs_now, tell_truth=True
                    )
                    reasoning.judge_day29_reply(reply29, agent2_losing=agent2_losing)
        (obs_p0, obs_p1), _rewards, done, _info = env.step([a0, a1])
        steps += 1

    money0 = _final_money(obs_p0, 0)
    money1 = _final_money(obs_p1, 1)
    r_money = money0 if reasoning_seat == 0 else money1
    q_money = money1 if reasoning_seat == 0 else money0
    if r_money > q_money:
        winner = "reasoning"
    elif q_money > r_money:
        winner = "questioning"
    else:
        winner = "tie"

    schedule_ok = _verify_schedule(reasoning, PRIME_HOURS_LT_11) and _verify_schedule(
        questioning, QuestioningAgent.schedule_hours
    )

    return MatchResult(
        seed=seed,
        reasoning_seat=reasoning_seat,
        questioning_seat=1 - reasoning_seat,
        money_p0=money0,
        money_p1=money1,
        winner=winner,
        reasoning_metrics=reasoning.metrics(),
        questioning_metrics=questioning.metrics(),
        schedule_ok=schedule_ok,
        starting_money_ok=starting_ok,
        opening_charity=charity,
        motive_dialogue={
            **(motive_dialogue or {}),
            "agent2_rules_understanding": rules_view,
            "agent1_private_policy": {
                "spoken_to_agent2": False,
                **private_policy,
            },
            "kaggle_path_trust": path_trust.to_dict(),
            "day29_judgment": reasoning.day29_judgment,
        },
    )


def run_seat_swap_pair(
    seed: int,
    *,
    reasoning_kwargs: Optional[Dict[str, Any]] = None,
    questioning_kwargs: Optional[Dict[str, Any]] = None,
    max_steps: int = 720,
) -> List[MatchResult]:
    """Agent1 as P0 and as P1 for the same seed."""
    return [
        run_match(
            seed,
            reasoning_seat=0,
            reasoning_kwargs=reasoning_kwargs,
            questioning_kwargs=questioning_kwargs,
            max_steps=max_steps,
        ),
        run_match(
            seed,
            reasoning_seat=1,
            reasoning_kwargs=reasoning_kwargs,
            questioning_kwargs=questioning_kwargs,
            max_steps=max_steps,
        ),
    ]


def _summarize_matches(matches: List[MatchResult]) -> Dict[str, Any]:
    n = len(matches)
    r_wins = sum(1 for m in matches if m.winner == "reasoning")
    q_wins = sum(1 for m in matches if m.winner == "questioning")
    ties = sum(1 for m in matches if m.winner == "tie")
    return {
        "n_matches": n,
        "reasoning_wins": r_wins,
        "questioning_wins": q_wins,
        "ties": ties,
        "reasoning_win_rate": r_wins / n if n else 0.0,
        "questioning_win_rate": q_wins / n if n else 0.0,
        "schedule_ok_all": all(m.schedule_ok for m in matches),
        "starting_money_ok_all": all(m.starting_money_ok for m in matches),
        "mean_money_p0": sum(m.money_p0 for m in matches) / n if n else 0.0,
        "mean_money_p1": sum(m.money_p1 for m in matches) / n if n else 0.0,
        "seat0_reasoning_wins": sum(
            1 for m in matches if m.reasoning_seat == 0 and m.winner == "reasoning"
        ),
        "seat1_reasoning_wins": sum(
            1 for m in matches if m.reasoning_seat == 1 and m.winner == "reasoning"
        ),
    }


def _match_to_dict(m: MatchResult) -> Dict[str, Any]:
    return {
        "seed": m.seed,
        "reasoning_seat": m.reasoning_seat,
        "questioning_seat": m.questioning_seat,
        "money_p0": m.money_p0,
        "money_p1": m.money_p1,
        "winner": m.winner,
        "schedule_ok": m.schedule_ok,
        "starting_money_ok": m.starting_money_ok,
        "opening_charity": m.opening_charity,
        "motive_dialogue": m.motive_dialogue,
        "reasoning_metrics": {
            k: v
            for k, v in m.reasoning_metrics.items()
            if k not in ("day_question_log", "action_audit")
        },
        "questioning_metrics": {
            k: v
            for k, v in m.questioning_metrics.items()
            if k not in ("day_question_log", "action_audit")
        },
        "reasoning_edge_stats": day_edge_question_stats(
            m.reasoning_metrics.get("day_question_log", [])
        ),
        "questioning_edge_stats": day_edge_question_stats(
            m.questioning_metrics.get("day_question_log", [])
        ),
    }


def _run_seeds(
    seeds: List[int],
    reasoning_kwargs: Dict[str, Any],
    questioning_kwargs: Dict[str, Any],
    max_steps: int,
) -> List[MatchResult]:
    out: List[MatchResult] = []
    for seed in seeds:
        out.extend(
            run_seat_swap_pair(
                seed,
                reasoning_kwargs=reasoning_kwargs,
                questioning_kwargs=questioning_kwargs,
                max_steps=max_steps,
            )
        )
    return out


# ── Experiment definitions ──────────────────────────────────────────────────


def exp1_baseline_seats(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    matches = _run_seeds(seeds, {"memory_slots": 10}, {"memory_slots": 10}, max_steps)
    summary = _summarize_matches(matches)
    summary["hypothesis"] = "Seat-only effect under prime/even schedule with memory=10."
    return {"experiment": 1, "name": "baseline_seats", "summary": summary, "matches": [_match_to_dict(m) for m in matches]}


def exp2_memory_floor(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    matches = _run_seeds(seeds, {"memory_slots": 10}, {"memory_slots": 10}, max_steps)
    summary = _summarize_matches(matches)
    summary["hypothesis"] = "Reasoning wins if determined facts beat noisy summaries (mem=10)."
    return {"experiment": 2, "name": "memory_floor", "summary": summary, "matches": [_match_to_dict(m) for m in matches]}


def exp3_memory_ceiling(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    matches = _run_seeds(seeds, {"memory_slots": 30}, {"memory_slots": 30}, max_steps)
    summary = _summarize_matches(matches)
    summary["hypothesis"] = "Extra probabilistic slots help or dilute Questioning (mem=30)."
    return {"experiment": 3, "name": "memory_ceiling", "summary": summary, "matches": [_match_to_dict(m) for m in matches]}


def exp4_asymmetric_memory(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    a = _run_seeds(seeds, {"memory_slots": 10}, {"memory_slots": 30}, max_steps)
    b = _run_seeds(seeds, {"memory_slots": 30}, {"memory_slots": 10}, max_steps)
    return {
        "experiment": 4,
        "name": "asymmetric_memory",
        "summary_r10_q30": _summarize_matches(a),
        "summary_r30_q10": _summarize_matches(b),
        "matches_r10_q30": [_match_to_dict(m) for m in a],
        "matches_r30_q10": [_match_to_dict(m) for m in b],
    }


def exp5_day_edge(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    """H1: Questioning wastes edge days. H2: Reasoning wastes mid-season."""
    h1_matches = _run_seeds(
        seeds,
        {"memory_slots": 10, "edge_question_bias": False},
        {"memory_slots": 10, "waste_edge_days": True, "omit_determined_truths": True},
        max_steps,
    )
    h2_matches = _run_seeds(
        seeds,
        {"memory_slots": 10, "edge_question_bias": True},
        {"memory_slots": 10, "waste_edge_days": False, "omit_determined_truths": True},
        max_steps,
    )
    h1_summary = _summarize_matches(h1_matches)
    h2_summary = _summarize_matches(h2_matches)

    def _edge_totals(matches: List[MatchResult], who: str) -> Dict[str, int]:
        key = "questioning_metrics" if who == "q" else "reasoning_metrics"
        day0 = mid = day29 = 0
        for m in matches:
            st = day_edge_question_stats(getattr(m, key).get("day_question_log", []))
            day0 += st.get("day0", 0)
            mid += st.get("mid", 0)
            day29 += st.get("day29", 0)
        return {"day0": day0, "mid": mid, "day29": day29}

    q_edge = _edge_totals(h1_matches, "q")
    r_mid = _edge_totals(h2_matches, "r")
    h1_supported = h1_summary["reasoning_win_rate"] >= 0.55 and (q_edge["day0"] + q_edge["day29"]) > 0
    h2_supported = (
        h2_summary["questioning_win_rate"] > h2_summary["reasoning_win_rate"]
        and r_mid["mid"] > 0
    )
    return {
        "experiment": 5,
        "name": "day_edge_questioning",
        "summary_H1_arm": {**h1_summary, "questioning_edge": q_edge},
        "summary_H2_arm": {**h2_summary, "reasoning_edge": r_mid},
        "summary": {
            "H1_supported": bool(h1_supported),
            "H2_supported": bool(h2_supported),
            "questioning_day0_questions": q_edge["day0"],
            "questioning_day29_questions": q_edge["day29"],
            "reasoning_mid_questions": r_mid["mid"],
            "H1_reasoning_win_rate": h1_summary["reasoning_win_rate"],
            "H2_questioning_win_rate": h2_summary["questioning_win_rate"],
            "schedule_ok_all": h1_summary["schedule_ok_all"] and h2_summary["schedule_ok_all"],
            "starting_money_ok_all": h1_summary["starting_money_ok_all"]
            and h2_summary["starting_money_ok_all"],
            "hypothesis": (
                "H1: Reasoning wins because Questioning wastes day 0/29. "
                "H2: Questioning wins because Reasoning wastes days 1–28."
            ),
        },
        "matches_H1": [_match_to_dict(m) for m in h1_matches],
        "matches_H2": [_match_to_dict(m) for m in h2_matches],
    }


def exp6_self_talk(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    matches = _run_seeds(
        seeds,
        {"memory_slots": 10, "force_self_talk": True},
        {"memory_slots": 10, "force_self_talk": True},
        max_steps,
    )
    summary = _summarize_matches(matches)
    summary["reasoning_self_talk"] = sum(m.reasoning_metrics.get("self_talk_detected", 0) for m in matches)
    summary["questioning_self_talk"] = sum(
        m.questioning_metrics.get("self_talk_detected", 0) for m in matches
    )
    summary["hypothesis"] = "Agent that treats QuestionEcho as self-talk and stops burning slots wins."
    return {"experiment": 6, "name": "self_talk_detection", "summary": summary, "matches": [_match_to_dict(m) for m in matches]}


def exp7_hidden_plain_sight(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    matches = _run_seeds(
        seeds,
        {"memory_slots": 10},
        {"memory_slots": 10, "omit_determined_truths": True},
        max_steps,
    )
    summary = _summarize_matches(matches)
    summary["omissions"] = sum(
        m.questioning_metrics.get("significant_omissions", 0) for m in matches
    )
    summary["hypothesis"] = "Public money/tiles are plain sight; private shed stays hidden."
    return {"experiment": 7, "name": "hidden_in_plain_sight", "summary": summary, "matches": [_match_to_dict(m) for m in matches]}


def exp8_coin_lead(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    aggressive = _run_seeds(
        seeds,
        {"memory_slots": 10, "aggressive_when_ahead": True},
        {"memory_slots": 10, "aggressive_when_ahead": True},
        max_steps,
    )
    idle = _run_seeds(
        seeds,
        {"memory_slots": 10, "aggressive_when_ahead": False},
        {"memory_slots": 10, "aggressive_when_ahead": False},
        max_steps,
    )
    return {
        "experiment": 8,
        "name": "coin_lead_validation",
        "summary_aggressive": _summarize_matches(aggressive),
        "summary_idle_when_ahead": _summarize_matches(idle),
        "hypothesis": "When ahead, is further aggression necessary vs idle + history validation?",
        "matches_aggressive": [_match_to_dict(m) for m in aggressive],
        "matches_idle": [_match_to_dict(m) for m in idle],
    }


def exp9_day2_day3(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    matches = _run_seeds(seeds, {"memory_slots": 10}, {"memory_slots": 10}, max_steps)
    summary = _summarize_matches(matches)
    summary["hypothesis"] = (
        "Day-2 seed pressure and day-3 hire: Reasoning uses determined hire/seed costs; "
        "Questioning only summarizes."
    )
    return {"experiment": 9, "name": "day2_seed_day3_labor", "summary": summary, "matches": [_match_to_dict(m) for m in matches]}


def exp10_memory_sweep(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    """Find minimum memory slots in {10,15,20,25,30} for Reasoning win rate >= 0.8."""
    sweep = {}
    min_slots_for_80 = None
    for n in (10, 15, 20, 25, 30):
        matches = _run_seeds(seeds, {"memory_slots": n}, {"memory_slots": n}, max_steps)
        summary = _summarize_matches(matches)
        sweep[str(n)] = summary
        if summary["reasoning_win_rate"] >= 0.8 and min_slots_for_80 is None:
            min_slots_for_80 = n
    return {
        "experiment": 10,
        "name": "win_all_memory_search",
        "sweep": sweep,
        "min_slots_reasoning_win_rate_ge_0_8": min_slots_for_80,
        "finding": (
            "Under Agent1 prime-hour ({2,3,5,7}) vs Agent2 even-hour schedule, "
            "no memory size in [10,30] reached Reasoning win-rate >= 0.8 "
            "(action-budget asymmetry dominates memory slot count)."
            if min_slots_for_80 is None
            else f"Minimum memory slots for >=80% Reasoning WR: {min_slots_for_80}"
        ),
        "hypothesis": "Minimum slots in [10,30] for Reasoning to win >=80% under prime/even schedule.",
    }


EXPERIMENTS: Dict[int, Callable[[List[int], int], Dict[str, Any]]] = {
    1: exp1_baseline_seats,
    2: exp2_memory_floor,
    3: exp3_memory_ceiling,
    4: exp4_asymmetric_memory,
    5: exp5_day_edge,
    6: exp6_self_talk,
    7: exp7_hidden_plain_sight,
    8: exp8_coin_lead,
    9: exp9_day2_day3,
    10: exp10_memory_sweep,
}


def run_suite(
    *,
    experiments: Optional[List[int]] = None,
    n_seeds: int = 3,
    base_seed: int = 42,
    max_steps: int = 720,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir or DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [base_seed + i for i in range(n_seeds)]
    selected = experiments or list(range(1, 11))
    report: Dict[str, Any] = {
        "seeds": seeds,
        "max_steps": max_steps,
        "starting_money": STARTING_MONEY,
        "agent1_hours": sorted(PRIME_HOURS_LT_11),
        "agent2_hours": "even",
        "hard_limits": limits_manifest(),
        "experiments": {},
    }
    for exp_id in selected:
        fn = EXPERIMENTS[exp_id]
        logger.info("Running experiment %d ...", exp_id)
        result = fn(seeds, max_steps)
        report["experiments"][str(exp_id)] = result
        exp_path = out_dir / f"exp{exp_id}"
        exp_path.mkdir(parents=True, exist_ok=True)
        with open(exp_path / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        logger.info("Wrote %s", exp_path / "metrics.json")

    with open(out_dir / "suite_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reasoning vs Questioning experiment suite")
    parser.add_argument("--experiments", type=str, default="1-10", help="e.g. 1-10 or 5,10")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_HERE),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    selected: List[int] = []
    for part in args.experiments.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            selected.extend(range(int(a), int(b) + 1))
        else:
            selected.append(int(part))

    report = run_suite(
        experiments=selected,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        max_steps=args.max_steps,
        out_dir=Path(args.out_dir),
    )
    # Print compact summary
    for eid, result in report["experiments"].items():
        if "summary" in result:
            s = result["summary"]
            if eid == "5":
                print(
                    f"Exp {eid} {result.get('name')}: "
                    f"schedule_ok={s.get('schedule_ok_all')} "
                    f"start3000={s.get('starting_money_ok_all')}"
                )
                print(
                    f"  H1={s.get('H1_supported')} (R_wr={s.get('H1_reasoning_win_rate')}) "
                    f"H2={s.get('H2_supported')} (Q_wr={s.get('H2_questioning_win_rate')}) "
                    f"Q_day0={s.get('questioning_day0_questions')} "
                    f"Q_day29={s.get('questioning_day29_questions')} "
                    f"R_mid={s.get('reasoning_mid_questions')}"
                )
            else:
                print(
                    f"Exp {eid} {result.get('name')}: "
                    f"R_wr={s.get('reasoning_win_rate', 0):.2f} "
                    f"Q_wr={s.get('questioning_win_rate', 0):.2f} "
                    f"schedule_ok={s.get('schedule_ok_all')} "
                    f"start3000={s.get('starting_money_ok_all')}"
                )
        elif eid == "10":
            print(
                f"Exp 10 memory sweep min_slots>=0.8: "
                f"{result.get('min_slots_reasoning_win_rate_ge_0_8')}"
            )
            for k, v in result.get("sweep", {}).items():
                print(f"  mem={k}: R_wr={v.get('reasoning_win_rate', 0):.2f}")
        elif "summary_r10_q30" in result:
            print(
                f"Exp {eid}: r10q30 R_wr={result['summary_r10_q30'].get('reasoning_win_rate', 0):.2f} "
                f"r30q10 R_wr={result['summary_r30_q10'].get('reasoning_win_rate', 0):.2f}"
            )
        elif "summary_aggressive" in result:
            print(
                f"Exp {eid}: aggressive R_wr={result['summary_aggressive'].get('reasoning_win_rate', 0):.2f} "
                f"idle R_wr={result['summary_idle_when_ahead'].get('reasoning_win_rate', 0):.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
