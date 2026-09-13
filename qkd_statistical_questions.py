"""Statistical Questioning Engine over the Q-K-D Vector Space.

Approach 1 (significant-questions K-map): expand to ≥100 probes across
10 Gray-adjacent families, rank by σ on trusted trajectories, fold 10×10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from proportional_kmap import generate_gray_code

logger = logging.getLogger("qkd_questions")

FAMILIES: List[str] = [
    "time",
    "capital",
    "labor",
    "crops",
    "market",
    "opponent",
    "risk",
    "poison_trust",
    "schedule_flops",
    "fellowship",
]

CHANNEL_BY_FAMILY: Dict[str, str] = {
    "time": "Q",
    "capital": "D",
    "labor": "D",
    "crops": "K",
    "market": "K",
    "opponent": "K",
    "risk": "Q",
    "poison_trust": "Q",
    "schedule_flops": "Q",
    "fellowship": "D",
}


@dataclass
class QKDStatisticalQuestion:
    """Individual strategic question probe over Q-K-D vector observations."""

    qid: str
    channel: str  # "Q" / "K" / "D"
    text: str
    description: str
    evaluator: Callable[[Dict[str, Any], np.ndarray, np.ndarray], float]
    family: str = "time"
    mean: float = 0.0
    std: float = 0.0
    variance: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    sample_count: int = 0


def _my_farm(obs: Dict[str, Any]) -> Dict[str, Any]:
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms", []) or []
    return farms[player] if len(farms) > player else {}


def _opp_farm(obs: Dict[str, Any]) -> Dict[str, Any]:
    player = int(obs.get("player", 0) or 0)
    opp = 1 - player
    farms = obs.get("farms", []) or []
    return farms[opp] if len(farms) > opp else {}


def _tiles(farm: Dict[str, Any]) -> List[Any]:
    return farm.get("tiles", []) or []


def _count_tiles(farm: Dict[str, Any], pred: Callable[[Any], bool]) -> float:
    n = 0
    for row in _tiles(farm):
        for cell in row or []:
            if pred(cell):
                n += 1
    return float(n)


def _market_prices(obs: Dict[str, Any]) -> Dict[str, float]:
    market = obs.get("market", {}) or {}
    prices = market.get("prices", {}) or {}
    return {str(k): float(v or 0.0) for k, v in prices.items()}


def _seeds(obs: Dict[str, Any]) -> Dict[str, float]:
    private = obs.get("private", {}) or {}
    seeds = private.get("seeds", {}) or {}
    return {str(k): float(v or 0.0) for k, v in seeds.items()}


def _shed(obs: Dict[str, Any]) -> Dict[str, float]:
    private = obs.get("private", {}) or {}
    shed = private.get("shed", {}) or {}
    return {str(k): float(v or 0.0) for k, v in shed.items()}


def _norm_log_money(x: float) -> float:
    return float(np.log1p(max(0.0, x)) / 12.0)


def _mk(
    qid: str,
    family: str,
    text: str,
    description: str,
    evaluator: Callable[[Dict[str, Any], np.ndarray, np.ndarray], float],
) -> QKDStatisticalQuestion:
    if not text.endswith("?"):
        raise ValueError(f"Question text must end with '?': {qid}")
    return QKDStatisticalQuestion(
        qid=qid,
        channel=CHANNEL_BY_FAMILY[family],
        text=text,
        description=description,
        evaluator=evaluator,
        family=family,
    )


# ── Family evaluators (lightweight; obs-primary) ───────────────────────────────


def _eval_days_remaining(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    day = float(obs.get("day", 0) or 0)
    return max(0.0, (30.0 - day) / 30.0)


def _eval_hour_norm(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(obs.get("hour", 0) or 0) / 24.0


def _eval_step_norm(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(obs.get("step", obs.get("hour", 0) or 0) or 0) / 720.0


def _eval_day_frac(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(obs.get("day", 0) or 0) / 30.0


def _eval_turns_into_day(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(obs.get("hour", 0) or 0) / 24.0


def _eval_season_pressure(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    day = float(obs.get("day", 0) or 0)
    hour = float(obs.get("hour", 0) or 0)
    return (day * 24.0 + hour) / (30.0 * 24.0)


def _eval_late_day_flag(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return 1.0 if float(obs.get("hour", 0) or 0) >= 18 else 0.0


def _eval_early_day_flag(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return 1.0 if float(obs.get("hour", 0) or 0) <= 4 else 0.0


def _eval_midseason_flag(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    day = float(obs.get("day", 0) or 0)
    return 1.0 if 10 <= day <= 20 else 0.0


def _eval_endgame_flag(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return 1.0 if float(obs.get("day", 0) or 0) >= 25 else 0.0


def _eval_my_money(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return _norm_log_money(float(_my_farm(obs).get("money", 0.0) or 0.0))


def _eval_opp_money(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return _norm_log_money(float(_opp_farm(obs).get("money", 0.0) or 0.0))


def _eval_money_gap(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    mine = float(_my_farm(obs).get("money", 0.0) or 0.0)
    theirs = float(_opp_farm(obs).get("money", 0.0) or 0.0)
    return float(np.tanh((mine - theirs) / 3000.0))


def _eval_money_ratio(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    mine = float(_my_farm(obs).get("money", 0.0) or 0.0)
    theirs = float(_opp_farm(obs).get("money", 0.0) or 0.0)
    return float(mine / (mine + theirs + 1.0))


def _eval_starting_bank_drawdown(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    mine = float(_my_farm(obs).get("money", 0.0) or 0.0)
    return float(np.clip((3000.0 - mine) / 3000.0, 0.0, 2.0))


def _eval_collaborative_surplus(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    mine = float(_my_farm(obs).get("money", 0.0) or 0.0)
    return float(np.clip((mine - 2112.0) / 2112.0, -1.0, 2.0))


def _eval_post_charity_gap(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    mine = float(_my_farm(obs).get("money", 0.0) or 0.0)
    return float(np.tanh((mine - 3888.0) / 1000.0))


def _eval_liquid_reserve(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return 1.0 if float(_my_farm(obs).get("money", 0.0) or 0.0) >= 1500 else 0.0


def _eval_capital_stress(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return 1.0 if float(_my_farm(obs).get("money", 0.0) or 0.0) < 888 else 0.0


def _eval_schmidt_stake_cover(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(min(1.0, float(_my_farm(obs).get("money", 0.0) or 0.0) / 888.0))


def _eval_hands_count(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(len(_my_farm(obs).get("hands", []) or [])) / 10.0


def _eval_hires_today(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(_my_farm(obs).get("hires_today", 0) or 0) / 5.0


def _eval_opp_hands(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(len(_opp_farm(obs).get("hands", []) or [])) / 10.0


def _eval_labor_advantage(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    mine = float(len(_my_farm(obs).get("hands", []) or []))
    theirs = float(len(_opp_farm(obs).get("hands", []) or []))
    return float(np.tanh((mine - theirs) / 3.0))


def _eval_farmer_x(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    farmer = _my_farm(obs).get("farmer", [0, 0]) or [0, 0]
    return float(farmer[0]) / 9.0 if len(farmer) > 0 else 0.0


def _eval_farmer_y(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    farmer = _my_farm(obs).get("farmer", [0, 0]) or [0, 0]
    return float(farmer[1]) / 9.0 if len(farmer) > 1 else 0.0


def _eval_workforce_capex(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    hands = float(len(_my_farm(obs).get("hands", []) or []))
    money = float(_my_farm(obs).get("money", 0.0) or 0.0)
    return float(np.tanh(hands * 100.0 / (money + 1.0)))


def _eval_idle_labor_risk(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    hands = float(len(_my_farm(obs).get("hands", []) or []))
    planted = _count_tiles(_my_farm(obs), lambda c: c not in (None, "LOCKED") and c is not None)
    return float(np.tanh(hands / (planted + 1.0)))


def _eval_hire_pressure(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    hires = float(_my_farm(obs).get("hires_today", 0) or 0)
    hour = float(obs.get("hour", 0) or 0)
    return float(np.tanh(hires * (1.0 + hour / 24.0)))


def _eval_unlocked_quads(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(len(_my_farm(obs).get("unlocked_quadrants", []) or [])) / 4.0


def _crop_seed(name: str) -> Callable[[Dict[str, Any], np.ndarray, np.ndarray], float]:
    def _fn(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
        return float(_seeds(obs).get(name, 0.0)) / 50.0

    return _fn


def _crop_shed(name: str) -> Callable[[Dict[str, Any], np.ndarray, np.ndarray], float]:
    def _fn(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
        return float(_shed(obs).get(name, 0.0)) / 50.0

    return _fn


def _eval_planted_tiles(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return _count_tiles(_my_farm(obs), lambda c: c not in (None, "LOCKED") and bool(c)) / 100.0


def _eval_locked_tiles(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return _count_tiles(_my_farm(obs), lambda c: c == "LOCKED") / 100.0


def _eval_empty_tiles(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return _count_tiles(_my_farm(obs), lambda c: c is None) / 100.0


def _eval_seed_diversity(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    seeds = _seeds(obs)
    return float(sum(1 for v in seeds.values() if v > 0)) / max(1, len(seeds))


def _eval_shed_fill(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    shed = _shed(obs)
    return float(sum(shed.values())) / 200.0


def _price(name: str) -> Callable[[Dict[str, Any], np.ndarray, np.ndarray], float]:
    def _fn(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
        prices = _market_prices(obs)
        return float(prices.get(name, 0.0)) / 250.0

    return _fn


def _eval_price_dispersion(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    vals = list(_market_prices(obs).values())
    if not vals:
        return 0.0
    return float(np.std(vals) / (np.mean(vals) + 1e-6))


def _eval_price_max(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    vals = list(_market_prices(obs).values())
    return float(max(vals) / 250.0) if vals else 0.0


def _eval_price_min(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    vals = list(_market_prices(obs).values())
    return float(min(vals) / 250.0) if vals else 0.0


def _eval_inventory_wheat(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    inv = (obs.get("market", {}) or {}).get("inventory", {}) or {}
    return float(inv.get("WHEAT", 0) or 0) / 10000.0


def _eval_shops_unlocked(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    town = obs.get("town", {}) or {}
    return float(len(town.get("unlocked_shops", []) or [])) / 5.0


def _eval_player_seat(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(obs.get("player", 0) or 0)


def _eval_opp_unlocked(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(len(_opp_farm(obs).get("unlocked_quadrants", []) or [])) / 4.0


def _eval_opp_planted(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return _count_tiles(_opp_farm(obs), lambda c: c not in (None, "LOCKED") and bool(c)) / 100.0


def _eval_opp_empty(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return _count_tiles(_opp_farm(obs), lambda c: c is None) / 100.0


def _eval_opp_farmer_x(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    farmer = _opp_farm(obs).get("farmer", [0, 0]) or [0, 0]
    return float(farmer[0]) / 9.0 if farmer else 0.0


def _eval_mirror_money(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    mine = float(_my_farm(obs).get("money", 0.0) or 0.0)
    theirs = float(_opp_farm(obs).get("money", 0.0) or 0.0)
    return 1.0 if abs(mine - theirs) < 1.0 else 0.0


def _eval_spatial_separation(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    a = _my_farm(obs).get("farmer", [0, 0]) or [0, 0]
    b = _opp_farm(obs).get("farmer", [0, 0]) or [0, 0]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    dist = abs(float(a[0]) - float(b[0])) + abs(float(a[1]) - float(b[1]))
    return float(dist / 18.0)


def _eval_overage_time(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(obs.get("remainingOverageTime", 0) or 0) / 60.0


def _eval_status_error(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return 1.0 if str(obs.get("_status", "")).upper() == "ERROR" else 0.0


def _eval_status_active(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return 1.0 if str(obs.get("_status", "")).upper() == "ACTIVE" else 0.0


def _eval_action_missing(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return 1.0 if obs.get("_action") is None else 0.0


def _eval_reward_null(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return 1.0 if obs.get("_reward") is None else 0.0


def _eval_poison_id_guard(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    eid = str(obs.get("_episode_id", "") or "")
    return 0.0 if eid.endswith("107982856") else 1.0


def _eval_trusted_id_match(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    eid = str(obs.get("_episode_id", "") or "")
    return 1.0 if "107982855" in eid else 0.0


def _eval_private_leak_proxy(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    private = obs.get("private", {}) or {}
    return 1.0 if private else 0.0


def _eval_buffer_integrity(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(obs.get("_trusted", 1.0) or 0.0)


def _eval_flop_budget_proxy(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    # TurnComputeBudget ≤ 42; normalize residual by hour parity as schedule proxy.
    hour = int(obs.get("hour", 0) or 0)
    used = hour % 43
    return float((42 - used) / 42.0)


def _eval_fib_hour(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    fib = {1, 4, 5, 13, 21}
    return 1.0 if int(obs.get("hour", 0) or 0) in fib else 0.0


def _eval_prime_hour(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    primes = {0, 2, 3, 5, 7, 11, 13, 17, 19, 23}
    return 1.0 if int(obs.get("hour", 0) or 0) in primes else 0.0


def _eval_zip_window(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)
    return 1.0 if day == 29 and 0 <= hour <= 4 else 0.0


def _eval_seat_trade_day(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    day = int(obs.get("day", 0) or 0)
    return 1.0 if day > 0 and day % 3 == 0 else 0.0


def _eval_day29_qa_reopen(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return 1.0 if int(obs.get("day", 0) or 0) >= 29 else 0.0


def _eval_schedule_conflict(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    hour = int(obs.get("hour", 0) or 0)
    fib = {1, 4, 5, 13, 21}
    primes = {0, 2, 3, 5, 7, 11, 13, 17, 19, 23}
    return 1.0 if hour in fib and hour in primes else 0.0


def _eval_pass_farm_mode(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)
    return 1.0 if day == 29 and hour <= 4 else 0.0


def _eval_musk_stake_cover(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(min(1.0, float(_my_farm(obs).get("money", 0.0) or 0.0) / 1667.0))


def _eval_gift_888_ratio(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(min(2.0, float(_my_farm(obs).get("money", 0.0) or 0.0) / 888.0))


def _eval_bank_612_floor(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return 1.0 if float(_my_farm(obs).get("money", 0.0) or 0.0) >= 612 else 0.0


def _eval_bank_2388_target(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(np.tanh(float(_my_farm(obs).get("money", 0.0) or 0.0) / 2388.0))


def _eval_joint_bank(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    mine = float(_my_farm(obs).get("money", 0.0) or 0.0)
    theirs = float(_opp_farm(obs).get("money", 0.0) or 0.0)
    return _norm_log_money(mine + theirs)


def _eval_charity_identity_day3(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return 1.0 if int(obs.get("day", 0) or 0) >= 3 else 0.0


def _eval_questioning_posture(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    # Agent2 fib hours → questioning posture proxy
    return _eval_fib_hour(obs, s_self, s_opp)


def _eval_reasoning_posture(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return _eval_prime_hour(obs, s_self, s_opp)


def _eval_shared_state_sync(obs: Dict[str, Any], s_self: np.ndarray, s_opp: np.ndarray) -> float:
    return float(obs.get("_sync_ok", 1.0) or 0.0)


def _build_question_bank() -> List[QKDStatisticalQuestion]:
    qs: List[QKDStatisticalQuestion] = [
        # time (10)
        _mk("TIME_DAYS_REMAINING", "time", "How many days are remaining in the season?", "Normalized days remaining.", _eval_days_remaining),
        _mk("TIME_HOUR_NORM", "time", "What fraction of the day has elapsed by hour?", "Hour / 24.", _eval_hour_norm),
        _mk("TIME_STEP_NORM", "time", "Where is the episode step within the season clock?", "Step normalized by 720.", _eval_step_norm),
        _mk("TIME_DAY_FRAC", "time", "What fraction of the 30-day season is complete?", "Day / 30.", _eval_day_frac),
        _mk("TIME_TURNS_INTO_DAY", "time", "How far into the current day have turns progressed?", "Hour fraction.", _eval_turns_into_day),
        _mk("TIME_SEASON_PRESSURE", "time", "How compressed is remaining season pressure?", "Day-hour continuum.", _eval_season_pressure),
        _mk("TIME_LATE_DAY", "time", "Is the clock in the late-day decision window?", "Hour >= 18.", _eval_late_day_flag),
        _mk("TIME_EARLY_DAY", "time", "Is the clock in the early-day Fib/prime overlap band?", "Hour <= 4.", _eval_early_day_flag),
        _mk("TIME_MIDSEASON", "time", "Are we inside the midseason crop decision band?", "Days 10-20.", _eval_midseason_flag),
        _mk("TIME_ENDGAME", "time", "Has the season entered the endgame harvest window?", "Day >= 25.", _eval_endgame_flag),
        # capital (10)
        _mk("CAP_MY_MONEY", "capital", "What is the balance of my subagents' wallets?", "Self money log-norm.", _eval_my_money),
        _mk("CAP_OPP_MONEY", "capital", "What is the balance of the opponent's wallet?", "Opp money log-norm.", _eval_opp_money),
        _mk("CAP_MONEY_GAP", "capital", "How large is the capital gap versus the opponent?", "Tanh money delta.", _eval_money_gap),
        _mk("CAP_MONEY_RATIO", "capital", "What share of joint liquid capital do I hold?", "Mine/(mine+theirs).", _eval_money_ratio),
        _mk("CAP_DRAWDOWN", "capital", "How far has my bank drawn down from the 3000 start?", "Drawdown vs 3000.", _eval_starting_bank_drawdown),
        _mk("CAP_COLLAB_SURPLUS", "capital", "Am I above the collaborative 2112 post-gift floor?", "Surplus vs 2112.", _eval_collaborative_surplus),
        _mk("CAP_POST_CHARITY", "capital", "How close is my bank to the post-charity 3888 target?", "Gap to 3888.", _eval_post_charity_gap),
        _mk("CAP_LIQUID_RESERVE", "capital", "Do I still hold a competitive 1500 liquid reserve?", "Money >= 1500.", _eval_liquid_reserve),
        _mk("CAP_STRESS", "capital", "Is capital stress below the 888 opening-gift threshold?", "Money < 888.", _eval_capital_stress),
        _mk("CAP_SCHMIDT_STAKE", "capital", "Can my bank cover Schmidt's 888 stake response?", "Money/888 capped.", _eval_schmidt_stake_cover),
        # labor (10)
        _mk("LAB_HANDS", "labor", "How many farm hands are currently employed?", "Hands/10.", _eval_hands_count),
        _mk("LAB_HIRES_TODAY", "labor", "How many hires were made earlier today?", "Hires_today/5.", _eval_hires_today),
        _mk("LAB_OPP_HANDS", "labor", "How large is the opponent's employed workforce?", "Opp hands/10.", _eval_opp_hands),
        _mk("LAB_ADVANTAGE", "labor", "Do I hold a labor advantage over the opponent?", "Hands delta tanh.", _eval_labor_advantage),
        _mk("LAB_FARMER_X", "labor", "Where is my farmer on the X tile axis?", "Farmer x / 9.", _eval_farmer_x),
        _mk("LAB_FARMER_Y", "labor", "Where is my farmer on the Y tile axis?", "Farmer y / 9.", _eval_farmer_y),
        _mk("LAB_CAPEX", "labor", "Is workforce cost high relative to liquid capital?", "Hands/money tanh.", _eval_workforce_capex),
        _mk("LAB_IDLE_RISK", "labor", "Is hired labor idle relative to planted tiles?", "Hands/planted.", _eval_idle_labor_risk),
        _mk("LAB_HIRE_PRESSURE", "labor", "How intense is same-day hiring pressure by hour?", "Hires*hour factor.", _eval_hire_pressure),
        _mk("LAB_QUADS", "labor", "How many farm quadrants have been unlocked?", "Unlocked quads/4.", _eval_unlocked_quads),
        # crops (12)
        _mk("CRP_PLANTED", "crops", "What fraction of my tiles are planted or occupied?", "Non-empty non-locked.", _eval_planted_tiles),
        _mk("CRP_LOCKED", "crops", "What fraction of my tiles remain locked?", "Locked tiles/100.", _eval_locked_tiles),
        _mk("CRP_EMPTY", "crops", "What fraction of my tiles are empty fallow soil?", "None tiles/100.", _eval_empty_tiles),
        _mk("CRP_SEED_DIV", "crops", "How diverse is my current seed portfolio?", "Nonzero seed types.", _eval_seed_diversity),
        _mk("CRP_SHED_FILL", "crops", "How full is the shed inventory across commodities?", "Shed sum/200.", _eval_shed_fill),
        _mk("CRP_WHEAT_SEED", "crops", "How many wheat seeds remain available to plant?", "Wheat seeds/50.", _crop_seed("WHEAT")),
        _mk("CRP_CARROT_SEED", "crops", "How many carrot seeds remain available to plant?", "Carrot seeds/50.", _crop_seed("CARROT")),
        _mk("CRP_TOMATO_SEED", "crops", "How many tomato seeds remain available to plant?", "Tomato seeds/50.", _crop_seed("TOMATO")),
        _mk("CRP_BERRY_SEED", "crops", "How many strawberry seeds remain available to plant?", "Berry seeds/50.", _crop_seed("STRAWBERRY")),
        _mk("CRP_MELON_SEED", "crops", "How many melon seeds remain available to plant?", "Melon seeds/50.", _crop_seed("MELON")),
        _mk("CRP_WHEAT_SHED", "crops", "How much wheat is stored in the shed?", "Wheat shed/50.", _crop_shed("WHEAT")),
        _mk("CRP_FERT_SHED", "crops", "How much fertilizer is stored in the shed?", "Fertilizer shed/50.", _crop_shed("FERTILIZER")),
        # market (11)
        _mk("MKT_WHEAT_PX", "market", "What is the current wheat market price level?", "Wheat/250.", _price("WHEAT")),
        _mk("MKT_CARROT_PX", "market", "What is the current carrot market price level?", "Carrot/250.", _price("CARROT")),
        _mk("MKT_TOMATO_PX", "market", "What is the current tomato market price level?", "Tomato/250.", _price("TOMATO")),
        _mk("MKT_BERRY_PX", "market", "What is the current strawberry market price level?", "Berry/250.", _price("STRAWBERRY")),
        _mk("MKT_MELON_PX", "market", "What is the current melon market price level?", "Melon/250.", _price("MELON")),
        _mk("MKT_EGG_PX", "market", "What is the current egg market price level?", "Egg/250.", _price("EGG")),
        _mk("MKT_MILK_PX", "market", "What is the current milk market price level?", "Milk/250.", _price("MILK")),
        _mk("MKT_WOOL_PX", "market", "What is the current wool market price level?", "Wool/250.", _price("WOOL")),
        _mk("MKT_DISP", "market", "How dispersed are commodity prices across the board?", "Price CV.", _eval_price_dispersion),
        _mk("MKT_MAX", "market", "Which commodity currently sits at the price ceiling?", "Max price/250.", _eval_price_max),
        _mk("MKT_SHOPS", "market", "How many town shops are unlocked for selling?", "Shops/5.", _eval_shops_unlocked),
        # opponent (10)
        _mk("OPP_SEAT", "opponent", "Am I seated as player 1 rather than player 0?", "Player index.", _eval_player_seat),
        _mk("OPP_WALLET", "opponent", "What is the opponent wallet strength on a log scale?", "Opp money.", _eval_opp_money),
        _mk("OPP_QUADS", "opponent", "How many quadrants has the opponent unlocked?", "Opp quads/4.", _eval_opp_unlocked),
        _mk("OPP_PLANTED", "opponent", "What fraction of opponent tiles are planted?", "Opp planted.", _eval_opp_planted),
        _mk("OPP_EMPTY", "opponent", "What fraction of opponent tiles are still empty?", "Opp empty.", _eval_opp_empty),
        _mk("OPP_HANDS", "opponent", "How many hands has the opponent hired?", "Opp hands.", _eval_opp_hands),
        _mk("OPP_FARMER_X", "opponent", "Where is the opponent farmer on the X axis?", "Opp farmer x.", _eval_opp_farmer_x),
        _mk("OPP_MIRROR", "opponent", "Are our banks currently mirrored within one unit?", "Money equality.", _eval_mirror_money),
        _mk("OPP_SEP", "opponent", "How far apart are the two farmers in Manhattan distance?", "Spatial sep.", _eval_spatial_separation),
        _mk("OPP_GAP", "opponent", "Is the opponent ahead of me in liquid capital?", "Negated gap proxy.", lambda o, a, b: float(1.0 - (0.5 + 0.5 * _eval_money_gap(o, a, b)))),
        # risk (10)
        _mk("RSK_OVERAGE", "risk", "How much remaining overage time buffer do I hold?", "Overage/60.", _eval_overage_time),
        _mk("RSK_ERROR", "risk", "Has the agent status flipped into ERROR?", "Status ERROR.", _eval_status_error),
        _mk("RSK_ACTIVE", "risk", "Is the agent status still ACTIVE?", "Status ACTIVE.", _eval_status_active),
        _mk("RSK_NO_ACTION", "risk", "Is the recorded action missing on this step?", "Action None.", _eval_action_missing),
        _mk("RSK_NULL_REWARD", "risk", "Is the step reward null or undefined?", "Reward None.", _eval_reward_null),
        _mk("RSK_CAP_STRESS", "risk", "Is capital stress signaling a bankruptcy path?", "Money < 888.", _eval_capital_stress),
        _mk("RSK_LOCKED", "risk", "How locked-down is my farm map as expansion risk?", "Locked tiles.", _eval_locked_tiles),
        _mk("RSK_IDLE_LABOR", "risk", "Is idle labor creating burn risk?", "Idle labor.", _eval_idle_labor_risk),
        _mk("RSK_ENDGAME", "risk", "Does endgame timing amplify downside risk?", "Endgame flag.", _eval_endgame_flag),
        _mk("RSK_PRICE_DISP", "risk", "Does price dispersion imply market risk?", "Price CV.", _eval_price_dispersion),
        # poison_trust (10)
        _mk("PT_POISON_GUARD", "poison_trust", "Did we reject the poison episode 107982856?", "Poison id guard.", _eval_poison_id_guard),
        _mk("PT_TRUSTED_MATCH", "poison_trust", "Are we scoring the trusted episode 107982855?", "Trusted id match.", _eval_trusted_id_match),
        _mk("PT_PRIVATE", "poison_trust", "Is private observation state present for this seat?", "Private present.", _eval_private_leak_proxy),
        _mk("PT_BUFFER", "poison_trust", "Is the buffer marked trusted for this trajectory?", "Trusted flag.", _eval_buffer_integrity),
        _mk("PT_ERROR", "poison_trust", "Does ERROR status indicate untrusted completion?", "Status ERROR.", _eval_status_error),
        _mk("PT_ACTIVE", "poison_trust", "Does ACTIVE status support trajectory trust?", "Status ACTIVE.", _eval_status_active),
        _mk("PT_ACTION", "poison_trust", "Is a concrete action present to audit intent?", "Action present inverse.", lambda o, a, b: 1.0 - _eval_action_missing(o, a, b)),
        _mk("PT_REWARD", "poison_trust", "Is reward defined enough to trust the step label?", "Reward defined.", lambda o, a, b: 1.0 - _eval_reward_null(o, a, b)),
        _mk("PT_PLAYER", "poison_trust", "Which player seat is emitting the trust probe?", "Player seat.", _eval_player_seat),
        _mk("PT_SYNC", "poison_trust", "Is dual-arch shared-state sync marked healthy?", "Sync ok.", _eval_shared_state_sync),
        # schedule_flops (10)
        _mk("SCH_FLOPS", "schedule_flops", "How much of the 42-FLOP turn budget remains?", "Budget residual.", _eval_flop_budget_proxy),
        _mk("SCH_FIB", "schedule_flops", "Is the current hour on the Agent2 Fibonacci schedule?", "Fib hour.", _eval_fib_hour),
        _mk("SCH_PRIME", "schedule_flops", "Is the current hour on the Agent1 prime-hour schedule?", "Prime hour.", _eval_prime_hour),
        _mk("SCH_ZIP", "schedule_flops", "Are we inside the day-29 hours 0-4 zip window?", "Zip window.", _eval_zip_window),
        _mk("SCH_SEAT_TRADE", "schedule_flops", "Is today a mid-episode seat-trade day (every 3)?", "Day % 3.", _eval_seat_trade_day),
        _mk("SCH_DAY29_QA", "schedule_flops", "Has day 29 reopened cross-agent Q&A?", "Day >= 29.", _eval_day29_qa_reopen),
        _mk("SCH_CONFLICT", "schedule_flops", "Do Fib and prime schedules collide this hour?", "Schedule overlap.", _eval_schedule_conflict),
        _mk("SCH_PASS_FARM", "schedule_flops", "Should we be in PASS-farm packaging mode now?", "Pass farm mode.", _eval_pass_farm_mode),
        _mk("SCH_EARLY", "schedule_flops", "Is the hour still inside the early packaging band?", "Hour <= 4.", _eval_early_day_flag),
        _mk("SCH_HOUR", "schedule_flops", "What is the normalized hour used for schedule gates?", "Hour/24.", _eval_hour_norm),
        # fellowship (11)
        _mk("FEL_MUSK_STAKE", "fellowship", "Can the bank cover Elon's 1667 stake menu choice?", "Money/1667.", _eval_musk_stake_cover),
        _mk("FEL_GIFT_888", "fellowship", "How many opening 888 gifts does current capital imply?", "Money/888.", _eval_gift_888_ratio),
        _mk("FEL_BANK_612", "fellowship", "Is the competitive 612 residual bank intact?", "Money >= 612.", _eval_bank_612_floor),
        _mk("FEL_BANK_2388", "fellowship", "How close are we to the gifted 2388 bank path?", "Tanh vs 2388.", _eval_bank_2388_target),
        _mk("FEL_JOINT", "fellowship", "What is the joint fellowship bank on a log scale?", "Joint money.", _eval_joint_bank),
        _mk("FEL_DAY3", "fellowship", "Has day 3 charity-identity fork already occurred?", "Day >= 3.", _eval_charity_identity_day3),
        _mk("FEL_QUESTIONING", "fellowship", "Is questioning posture scheduled for this hour?", "Fib posture.", _eval_questioning_posture),
        _mk("FEL_REASONING", "fellowship", "Is reasoning posture scheduled for this hour?", "Prime posture.", _eval_reasoning_posture),
        _mk("FEL_SYNC", "fellowship", "Is fellowship sync healthy for dual-arch claims?", "Sync ok.", _eval_shared_state_sync),
        _mk("FEL_POST_CHARITY", "fellowship", "Are we tracking the collaborative 3888 post-charity bank?", "Gap to 3888.", _eval_post_charity_gap),
        _mk("FEL_SCHMIDT", "fellowship", "Can fellowship funds cover Schmidt's 888 response stake?", "Schmidt cover.", _eval_schmidt_stake_cover),
    ]
    return qs


CANONICAL_QKD_QUESTIONS: List[QKDStatisticalQuestion] = _build_question_bank()


def map_observation_to_questions(obs: Dict[str, Any]) -> Dict[str, float]:
    """Map a raw observation dictionary directly to all question probe responses."""
    z = np.zeros(1, dtype=np.float32)
    return {q.qid: float(q.evaluator(obs, z, z)) for q in CANONICAL_QKD_QUESTIONS}


def score_questions_on_observations(
    observations: Sequence[Dict[str, Any]],
    questions: Optional[Sequence[QKDStatisticalQuestion]] = None,
) -> List[QKDStatisticalQuestion]:
    """Fit mean/std on observations without requiring the vector encoder."""
    qs = list(questions) if questions is not None else [q for q in CANONICAL_QKD_QUESTIONS]
    if not observations:
        return qs
    z = np.zeros(1, dtype=np.float32)
    n = len(observations)
    matrix = np.zeros((len(qs), n), dtype=np.float32)
    for j, obs in enumerate(observations):
        for i, q in enumerate(qs):
            matrix[i, j] = float(q.evaluator(obs, z, z))
    for i, q in enumerate(qs):
        vals = matrix[i, :]
        q.mean = float(np.mean(vals))
        q.std = float(np.std(vals))
        q.variance = float(np.var(vals))
        q.min_val = float(np.min(vals))
        q.max_val = float(np.max(vals))
        q.sample_count = n
    return qs


def fold_significant_questions_10x10(
    questions: Sequence[QKDStatisticalQuestion],
    families: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Fold top-10 σ per family into a clear 10×10 matrix (family × σ-rank)."""
    fams = list(families) if families is not None else list(FAMILIES)
    if len(fams) != 10:
        raise ValueError(f"Expected 10 families, got {len(fams)}")

    # Locked Gray-adjacent family order from design; annotate with 4-bit Gray labels.
    gray = generate_gray_code(4)[:10]
    row_families = list(fams)
    gray_labels = [format(g, "04b") for g in gray]

    by_family: Dict[str, List[QKDStatisticalQuestion]] = {f: [] for f in fams}
    for q in questions:
        if q.family in by_family:
            by_family[q.family].append(q)
    for f in fams:
        by_family[f].sort(key=lambda qq: qq.std, reverse=True)

    matrix: List[List[Dict[str, Any]]] = []
    for r, family in enumerate(row_families):
        ranked = by_family[family]
        row: List[Dict[str, Any]] = []
        for c in range(10):
            if c < len(ranked):
                q = ranked[c]
                cell = {
                    "row": r,
                    "col": c,
                    "family": family,
                    "rank_in_family": c,
                    "qid": q.qid,
                    "text": q.text,
                    "channel": q.channel,
                    "sigma": float(q.std),
                    "mean": float(q.mean),
                    "n": int(q.sample_count),
                    "significant": bool(q.std > 0.0),
                }
            else:
                cell = {
                    "row": r,
                    "col": c,
                    "family": family,
                    "rank_in_family": c,
                    "qid": None,
                    "text": "INSUFFICIENT?",
                    "channel": None,
                    "sigma": 0.0,
                    "mean": 0.0,
                    "n": 0,
                    "significant": False,
                }
            row.append(cell)
        matrix.append(row)

    sigmas = [cell["sigma"] for row in matrix for cell in row]
    texts = [cell["text"] for row in matrix for cell in row]
    nonempty = sum(1 for t in texts if t and not t.startswith("INSUFFICIENT"))
    family_coverage = sum(
        1
        for row in matrix
        if all(cell.get("qid") for cell in row) and all(str(cell.get("text", "")).endswith("?") for cell in row)
    )
    return {
        "kind": "significant_questions_kmap_10x10",
        "role": "trained_qkd_rank_collapse",
        "families": row_families,
        "shape": [10, 10],
        "matrix": matrix,
        "metrics": {
            "nonempty_cells": int(nonempty),
            "mean_sigma": float(np.mean(sigmas)) if sigmas else 0.0,
            "family_coverage": int(family_coverage),
            "bank_size": len(list(questions)),
            "gray_row_order": row_families,
            "gray_labels": gray_labels,
        },
    }


class QKDStatisticalQuestionBank:
    """Statistical Question Evaluator & Variance Ranker for Q-K-D Vectors."""

    def __init__(
        self,
        questions: Optional[Sequence[QKDStatisticalQuestion]] = None,
    ) -> None:
        self.questions: List[QKDStatisticalQuestion] = (
            list(questions) if questions is not None else list(CANONICAL_QKD_QUESTIONS)
        )

    def evaluate_observation(self, obs: Dict[str, Any]) -> Dict[str, float]:
        return map_observation_to_questions(obs)

    def fit_sample_distribution(
        self,
        observations: Sequence[Dict[str, Any]],
    ) -> List[QKDStatisticalQuestion]:
        self.questions = score_questions_on_observations(observations, self.questions)
        self.questions.sort(key=lambda q: q.std, reverse=True)
        return self.questions

    def get_ranked_questions(self) -> List[QKDStatisticalQuestion]:
        return sorted(self.questions, key=lambda q: q.std, reverse=True)

    def get_top_k_significant_questions(self, k: int = 10) -> List[QKDStatisticalQuestion]:
        return self.get_ranked_questions()[:k]

    def summary_table(self) -> List[Dict[str, Any]]:
        ranked = self.get_ranked_questions()
        table = []
        for rank, q in enumerate(ranked, start=1):
            table.append(
                {
                    "rank": rank,
                    "qid": q.qid,
                    "family": q.family,
                    "channel": q.channel,
                    "std_deviation": round(q.std, 4),
                    "variance": round(q.variance, 4),
                    "mean": round(q.mean, 4),
                    "range": [round(q.min_val, 3), round(q.max_val, 3)],
                    "question": q.text,
                    "description": q.description,
                }
            )
        return table
