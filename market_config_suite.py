"""Market Configuration and Future Price Functional Suite for Kaggriculture.

Provides:
- InitialTerminalConfiguration (number of days, amount of money, start price of every commodity)
- build_initial_terminal_configuration: Extract configuration from obs/env
- build_market_functions: Series of future price and demand elasticity functions
- build_opponent_functions: Opponent trajectory prediction and adversarial modeling functions
- Standardized market order ranking and elasticity helpers
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

_PRICE_FLOOR = 1
_DEMAND_ALPHA = 0.25
_MARKET_PARAMS = {
    "WHEAT": (25, 10000, 400, "sqrt", 0.8, "log", 0.2),
    "CARROT": (35, 10000, 450, "log", 0.2, "sqrt", 0.7),
    "TOMATO": (60, 10000, 200, "linear", 0.4, "sqrt", 0.6),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.7, "linear", 1.6),
    "MELON": (250, 10000, 300, "log", 0.2, "sq", 3.6),
    "EGG": (50, 10000, 332, "linear", 0.4, "log", 0.2),
    "MILK": (160, 10000, 122, "sqrt", 0.6, "linear", 1.6),
    "WOOL": (200, 10000, 105, "log", 0.2, "sq", 3.2),
    "FERTILIZER": (100, 10000, 200, "linear", 0.4, "linear", 0.4),
}
_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _copy_action(action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    action = copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [list(order or ["PASS"]) for order in (action.get("hands") or [])],
        "market": [list(order) for order in (action.get("market") or [])],
    }


def _seat(obs: Dict[str, Any]) -> int:
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _farm(obs: Dict[str, Any], seat: int) -> Dict[str, Any]:
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _align_hands(action: Dict[str, Any], obs: Dict[str, Any]) -> Dict[str, Any]:
    action = _copy_action(action)
    expected = len(_get(_farm(obs, _seat(obs)), "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [list(order or ["PASS"]) for order in hands[:expected]]
    return action


def _shape(name: str, value: float) -> float:
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1.0 + value)
    raise ValueError(name)


def market_price(item: str, inventory: int) -> int:
    """Compute official market clearing price given current market inventory."""
    if item not in _MARKET_PARAMS:
        return _PRICE_FLOOR
    base, equilibrium, scale, below_func, below_target, above_func, above_target = (
        _MARKET_PARAMS[item]
    )
    if inventory < equilibrium:
        amplitude = below_target * base / _shape(below_func, scale)
        price = base + amplitude * _shape(below_func, equilibrium - inventory)
    else:
        amplitude = above_target * base / _shape(above_func, scale)
        price = base - amplitude * _shape(above_func, inventory - equilibrium)
    return max(_PRICE_FLOOR, int(round(price)))


def is_sell_order(order: Any) -> bool:
    return (
        isinstance(order, (list, tuple))
        and len(order) >= 3
        and order[0] == "SELL"
        and order[1] in _MARKET_PARAMS
    )


def impact_score(obs: Dict[str, Any], order: Sequence[Any]) -> float:
    if not is_sell_order(order):
        return float("-inf")
    item = str(order[1])
    try:
        quantity = max(0, int(order[2]))
    except (TypeError, ValueError):
        return 0.0
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    current_quote = float(
        _get(prices, item, market_price(item, current_inventory)) or 0
    )
    later_quote = float(market_price(item, current_inventory + quantity))
    return float(quantity) * max(0.0, current_quote - later_quote)


def demand_per_day(obs: Dict[str, Any], configuration: Any, item: str) -> float:
    town = _get(obs, "town", {}) or {}
    shops = list(_get(town, "unlocked_shops", []) or [])
    turns_per_day = int(_get(configuration, "turnsPerDay", 24) or 24)
    shop_interval = max(
        1, int(_get(configuration, "townShopSellInterval", 4) or 4)
    )
    demand = 0.0
    for shop in shops:
        products = _SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += (turns_per_day / shop_interval) * (
                2 if len(products) == 1 else 1
            )
    if item != "FERTILIZER":
        center_interval = max(
            1,
            int(_get(configuration, "townCenterSellInterval", 24) or 24),
        )
        demand += (turns_per_day / center_interval)
    return demand


def order_score(obs: Dict[str, Any], configuration: Any, order: Sequence[Any]) -> float:
    score = impact_score(obs, order)
    if score <= 0 or not is_sell_order(order):
        return score
    item = str(order[1])
    quantity = max(0, int(order[2]))
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    current_inventory = int(_get(inventory, item, 10000) or 0)
    demand = max(0.25, demand_per_day(obs, configuration, item))
    excess = max(0.0, current_inventory + quantity - 10000)
    urgency = min(1.0, (excess / demand) / 10.0)
    return score * (1.0 + _DEMAND_ALPHA * urgency)


def rank_sell_slots(obs: Dict[str, Any], action: Dict[str, Any], configuration: Any) -> Dict[str, Any]:
    action = _copy_action(action)
    market = list(action.get("market") or [])
    rows = [
        (order_score(obs, configuration, order), -index, list(order))
        for index, order in enumerate(market)
        if is_sell_order(order)
    ]
    if len(rows) < 2:
        return action
    rows.sort(reverse=True)
    ranked = iter(row[2] for row in rows)
    action["market"] = [next(ranked) if is_sell_order(order) else order for order in market]
    return action


# ── Initial Terminal Configuration ─────────────────────────────────────────────

@dataclass
class InitialTerminalConfiguration:
    """Initial and terminal competition configuration parameters.
    
    Contains:
      - number_of_days (and alias .days)
      - amount_of_money (and alias .initial_money)
      - start_price_of_every_commodity (and alias .start_prices)
      - terminal_money, turns_per_day, episode_steps
    """
    number_of_days: int = 30
    amount_of_money: float = 3000.0
    start_price_of_every_commodity: Dict[str, float] = field(default_factory=lambda: {
        "WHEAT": 25.0,
        "CARROT": 35.0,
        "TOMATO": 60.0,
        "STRAWBERRY": 120.0,
        "MELON": 250.0,
        "EGG": 50.0,
        "MILK": 160.0,
        "WOOL": 200.0,
        "FERTILIZER": 100.0,
    })
    terminal_money: float = 100000.0
    turns_per_day: int = 24
    episode_steps: int = 720
    town_center_sell_interval: int = 24
    town_shop_sell_interval: int = 4

    @property
    def days(self) -> int:
        return self.number_of_days

    @days.setter
    def days(self, val: int) -> None:
        self.number_of_days = int(val)

    @property
    def initial_money(self) -> float:
        return self.amount_of_money

    @initial_money.setter
    def initial_money(self, val: float) -> None:
        self.amount_of_money = float(val)

    @property
    def start_prices(self) -> Dict[str, float]:
        return self.start_price_of_every_commodity

    @start_prices.setter
    def start_prices(self, val: Dict[str, float]) -> None:
        self.start_price_of_every_commodity = dict(val)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number_of_days": self.number_of_days,
            "amount_of_money": self.amount_of_money,
            "start_price_of_every_commodity": dict(self.start_price_of_every_commodity),
            "days": self.number_of_days,
            "initial_money": self.amount_of_money,
            "start_prices": dict(self.start_price_of_every_commodity),
            "terminal_money": self.terminal_money,
            "turns_per_day": self.turns_per_day,
            "episode_steps": self.episode_steps,
            "town_center_sell_interval": self.town_center_sell_interval,
            "town_shop_sell_interval": self.town_shop_sell_interval,
        }


def build_initial_terminal_configuration(
    obs: Dict[str, Any],
    configuration: Any = None,
) -> InitialTerminalConfiguration:
    """Build standardized InitialTerminalConfiguration from game environment and observation."""
    turns_per_day = int(_get(configuration, "turnsPerDay", 24) or 24)
    episode_steps = int(_get(configuration, "episodeSteps", 720) or 720)
    total_days = max(1, episode_steps // max(1, turns_per_day))

    farms = _get(obs, "farms", []) or []
    init_money = 3000.0
    if farms and isinstance(farms[0], dict):
        init_money = float(farms[0].get("money", 3000.0) or 3000.0)

    start_prices = {
        "WHEAT": 25.0,
        "CARROT": 35.0,
        "TOMATO": 60.0,
        "STRAWBERRY": 120.0,
        "MELON": 250.0,
        "EGG": 50.0,
        "MILK": 160.0,
        "WOOL": 200.0,
        "FERTILIZER": 100.0,
    }
    market = _get(obs, "market", {}) or {}
    prices = _get(market, "prices", {}) or {}
    for k, v in prices.items():
        if k in start_prices and v is not None:
            start_prices[k] = float(v)

    return InitialTerminalConfiguration(
        number_of_days=total_days,
        amount_of_money=init_money,
        start_price_of_every_commodity=start_prices,
        terminal_money=100000.0,
        turns_per_day=turns_per_day,
        episode_steps=episode_steps,
        town_center_sell_interval=int(_get(configuration, "townCenterSellInterval", 24) or 24),
        town_shop_sell_interval=int(_get(configuration, "townShopSellInterval", 4) or 4),
    )


# ── Market Functions (Series of Future Price Functions) ─────────────────────────

def build_market_functions(
    obs: Dict[str, Any],
    config: Union[InitialTerminalConfiguration, Dict[str, Any]],
) -> Dict[str, Callable]:
    """Build a series of future price and market elasticity functions."""
    market = _get(obs, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}

    def predict_future_price(item: str, inventory_delta: int = 0, horizon_hours: int = 0) -> float:
        cur_inv = int(_get(inventory, item, 10000) or 10000)
        demand_rate = demand_per_day(obs, config, item) / float(_get(config, "turns_per_day", 24) or 24)
        projected_inv = max(0, cur_inv + inventory_delta - int(demand_rate * horizon_hours))
        return float(market_price(item, projected_inv))

    def predict_price_trajectory(item: str, horizon_days: int = 5) -> List[float]:
        traj = []
        for d in range(horizon_days):
            traj.append(predict_future_price(item, horizon_hours=d * 24))
        return traj

    def project_future_price_trajectory(item: str, horizon_days: int = 5) -> List[float]:
        return predict_price_trajectory(item, horizon_days=horizon_days)

    def get_price_elasticity(item: str, cur_inventory: int) -> float:
        p_base = float(market_price(item, cur_inventory))
        p_plus = float(market_price(item, cur_inventory + 50))
        return (p_plus - p_base) / 50.0

    def get_demand_absorption_rate(item: str) -> float:
        return float(demand_per_day(obs, config, item))

    def compute_optimal_sell_batch(item: str, cur_inventory: int) -> int:
        demand_day = max(1.0, demand_per_day(obs, config, item))
        return int(min(250, max(10, demand_day * 0.5)))

    def predict_spot_price(item: str) -> float:
        cur_inv = int(_get(inventory, item, 10000) or 10000)
        return float(market_price(item, cur_inv))

    return {
        "predict_future_price": predict_future_price,
        "predict_price_trajectory": predict_price_trajectory,
        "project_future_price_trajectory": project_future_price_trajectory,
        "get_price_elasticity": get_price_elasticity,
        "get_demand_absorption_rate": get_demand_absorption_rate,
        "compute_optimal_sell_batch": compute_optimal_sell_batch,
        "predict_spot_price": predict_spot_price,
    }


# ── Opponent Functions Suite ───────────────────────────────────────────────────

def build_opponent_functions(
    obs: Dict[str, Any],
    config: Union[InitialTerminalConfiguration, Dict[str, Any]],
) -> Dict[str, Callable]:
    """Build a suite of opponent trajectory prediction and adversarial modeling functions."""
    player = _seat(obs)
    opp = 1 - player
    farms = _get(obs, "farms", []) or []
    my_farm = farms[player] if len(farms) > player else {}
    opp_farm = farms[opp] if len(farms) > opp else {}

    def predict_opponent_money_trajectory(horizon_days: int = 5) -> float:
        opp_money = float(opp_farm.get("money", 0.0) or 0.0)
        opp_hires = len(opp_farm.get("hands", []) or [])
        growth_rate = 500.0 + (opp_hires * 250.0)
        return opp_money + (growth_rate * horizon_days)

    def predict_opponent_workforce_growth() -> int:
        return len(opp_farm.get("hands", []) or [])

    def estimate_opponent_market_impact(item: str) -> float:
        opp_quads = len(opp_farm.get("unlocked_quadrants", []) or [])
        return 0.1 * opp_quads

    def compute_adversarial_lead_gap() -> float:
        my_m = float(my_farm.get("money", 0.0) or 0.0)
        opp_m = float(opp_farm.get("money", 0.0) or 0.0)
        return my_m - opp_m

    def estimate_opponent_aggression() -> float:
        opp_quads = len(opp_farm.get("unlocked_quadrants", []) or [])
        opp_hires = len(opp_farm.get("hands", []) or [])
        return float(opp_quads * 1.5 + opp_hires * 0.5)

    return {
        "predict_opponent_money_trajectory": predict_opponent_money_trajectory,
        "predict_opponent_workforce_growth": predict_opponent_workforce_growth,
        "estimate_opponent_market_impact": estimate_opponent_market_impact,
        "compute_adversarial_lead_gap": compute_adversarial_lead_gap,
        "estimate_opponent_aggression": estimate_opponent_aggression,
    }
