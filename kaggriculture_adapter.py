"""Single source of truth for Kaggriculture observation/action contracts.

Maps between official Kaggle command-list actions and branched integer indices
used by DQN / Path B trainers. Shared by dataset_loader, env wrapper, train
export, eval_policy, and self-play.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Canonical constants (match dataset_loader + official episodes) ──────────

CROPS: List[str] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]

# Kinematic feedback season (optional via ``--turns-per-cycle 72``): 3×4×6.
# Default self-play uses competition ``turnsPerDay=24`` for ladder parity.
KINEMATIC_PHASE_A: int = 3
KINEMATIC_PHASE_B: int = 4
KINEMATIC_PHASE_C: int = 6
TURNS_PER_CYCLE: int = KINEMATIC_PHASE_A * KINEMATIC_PHASE_B * KINEMATIC_PHASE_C  # 72
CYCLES_PER_EPISODE: int = 10
EPISODE_STEPS: int = TURNS_PER_CYCLE * CYCLES_PER_EPISODE  # 720
# Competition / reference-ladder / historical episode tapes (Kaggle default).
COMPETITION_TURNS_PER_DAY: int = 24

FARMER_ACTIONS: Dict[str, int] = {
    "PASS": 0,
    "DIG": 1,
    "WATER": 2,
    "PLANT": 3,
    "HARVEST": 4,
    "NORTH": 5,
    "SOUTH": 6,
    "WEST": 7,
    "EAST": 8,
    "DROP": 9,
    "PICKUP": 10,
    "BUILD_COOP": 11,
    "BUILD_PASTURE": 12,
    "FERTILIZE": 13,
    "FEED": 14,
}

# Legacy encode aliases from episode JSON / prior BUY_ANIMAL/OTHER indices
_FARMER_ENCODE_ALIASES = {
    "MOVE_NORTH": 5,
    "MOVE_SOUTH": 6,
    "MOVE_WEST": 7,
    "MOVE_EAST": 8,
    "BUY_ANIMAL": 14,  # animals are market ops; map old index to FEED
    "OTHER": 0,
    "PLACE": 10,  # PLACE shares PICKUP slot; decode resolves on structure
    "CARE": 14,
    "COLLECT_FERTILIZER": 14,
}

FARMER_INDEX_TO_VERB: List[str] = [
    "PASS", "DIG", "WATER", "PLANT", "HARVEST",
    "NORTH", "SOUTH", "WEST", "EAST",
    "DROP", "PICKUP", "BUILD_COOP", "BUILD_PASTURE", "FERTILIZE", "FEED",
]

# Engine crop first_yield_day (must match kaggriculture.py CROPS table).
CROP_FIRST_YIELD_DAY: Dict[str, int] = {
    "WHEAT": 2,
    "CARROT": 2,
    "TOMATO": 8,
    "STRAWBERRY": 10,
    "MELON": 10,
}

MARKET_ACTIONS: Dict[str, int] = {
    "PASS": 0,
    "BUY_SEED": 1,
    "BUY_PRODUCT": 2,
    "BUY_ANIMAL": 3,
    "SELL": 4,
    "HIRE": 5,
    "BUY_LAND": 6,
}

MARKET_INDEX_TO_VERB: List[str] = [
    "PASS", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND",
    "OTHER", "OTHER", "OTHER",
]

NUM_FARMER_ACTIONS = 15
NUM_HAND_ACTIONS = 15
NUM_HANDS = 6
NUM_MARKET_ACTIONS = 10

TILE_CLASS: Dict[str, int] = {
    "EMPTY": 0,
    "LOCKED": 1,
    "WEED": 2,
    "WHEAT": 3,
    "CARROT": 4,
    "TOMATO": 5,
    "STRAWBERRY": 6,
    "MELON": 7,
    "OTHER": 8,
}

CROP_TO_TILE_CLASS = {crop: TILE_CLASS[crop] for crop in CROPS}

# WARNING: these MUST match the actual competition engine (repo root kaggriculture.py).
# They were previously CARROT=8 / TOMATO=5 / STRAWBERRY=3 / MELON=2 and land
# [200, 400, 800, 1600] — 2–10x cheaper than the real game. That made the market
# action masks claim actions were affordable when they were not, so the trained
# policy emitted silent no-ops (BuySeed/BuyLand with insufficient bank) on live
# servers and the BUY_LAND gate hid genuinely affordable expansions.
SEED_COSTS = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
LAND_PRICES: List[int] = [1000, 2000, 4000]

# Path B observation encoding.
SEASON_DAYS: int = 30  # competition season length (matches engine + reward shaper)
# Number of channels in encode_path_b_observation output. Order matters: keep
# PATH_B_TILE_CHANNELS and the channel indices below in sync with the encoder.
PATH_B_TILE_CHANNELS: int = 18


def hire_cost_today(hires_today: int) -> int:
    """Return Fibonacci hiring cost for the n-th hire of the day (unbounded)."""
    n = max(int(hires_today), 0)
    if n <= 1:
        return 1
    a, b = 1, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def daily_hire_orders_wanted(obs: Dict[str, Any]) -> int:
    """Count wanted daily hire orders (only in morning hour 0)."""
    if int(obs.get("hour", 0) or 0) > 0:
        return 0
    player = obs.get("player", 0)
    farms = obs.get("farms", [])
    farm = farms[player] if len(farms) > player else {}
    hires_today = int(farm.get("hires_today", 0) or 0)
    hands = farm.get("hands", []) or []
    money = float(farm.get("money", 0.0) or 0.0)
    if hires_today >= 4 or len(hands) >= 6 or money < 50.0:
        return 0
    return max(0, min(4 - hires_today, 6 - len(hands)))


def plant_is_harvestable(tile: Dict[str, Any], current_day: int) -> bool:
    """True if plant is on or past its crop-specific first_yield_day with yield > 0."""
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        return False
    planted_day = int(tile.get("planted_day", 0) or 0)
    yield_units = int(tile.get("yield_units", 0) or 0)
    crop = str(tile.get("crop", "WHEAT") or "WHEAT")
    first_yield = CROP_FIRST_YIELD_DAY.get(crop, 2)
    return (current_day - planted_day) >= first_yield and yield_units > 0


def plant_is_mature(tile: Dict[str, Any], current_day: int) -> bool:
    """True if plant reached maximum/good-enough yield units or mature age."""
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        return False
    planted_day = int(tile.get("planted_day", 0) or 0)
    yield_units = int(tile.get("yield_units", 0) or 0)
    return yield_units >= 3 or (current_day - planted_day) >= 4


def select_hand_farm_verbs(obs: Dict[str, Any]) -> List[int]:
    """Select appropriate farm action (WATER/HARVEST/DIG/PASS) for each hired hand."""
    player = obs.get("player", 0)
    farms = obs.get("farms", [])
    farm = farms[player] if len(farms) > player else {}
    tiles = farm.get("tiles", []) or []
    hands = farm.get("hands", []) or []
    day = int(obs.get("day", 1) or 1)
    verbs: List[int] = []
    for h in hands:
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            hx, hy = int(h[0]), int(h[1])
            if 0 <= hy < len(tiles) and 0 <= hx < len(tiles[hy]):
                tile = tiles[hy][hx]
                if isinstance(tile, dict):
                    if tile.get("kind") == "PLANT":
                        if not tile.get("watered_today", False):
                            verbs.append(FARMER_ACTIONS["WATER"])
                            continue
                        if plant_is_harvestable(tile, day):
                            verbs.append(FARMER_ACTIONS["HARVEST"])
                            continue
                    elif tile.get("kind") == "WEED":
                        verbs.append(FARMER_ACTIONS["DIG"])
                        continue
        verbs.append(FARMER_ACTIONS["PASS"])
    return verbs


def _normalize_action_op(raw_op: Any) -> str:
    if not isinstance(raw_op, str):
        return "PASS"
    if raw_op.startswith("PLANT"):
        return "PLANT"
    if raw_op.startswith("PICKUP"):
        return "PICKUP"
    if raw_op.startswith("DROP"):
        return "DROP"
    return raw_op


def _branch_action(raw_branch: Any, default: str = "PASS") -> int:
    if isinstance(raw_branch, list) and raw_branch:
        op = _normalize_action_op(raw_branch[0])
    elif isinstance(raw_branch, str):
        op = _normalize_action_op(raw_branch)
    else:
        op = default
    if op in FARMER_ACTIONS:
        return FARMER_ACTIONS[op]
    return _FARMER_ENCODE_ALIASES.get(op, FARMER_ACTIONS["PASS"])


def encode_tiles(raw_tiles: Any) -> np.ndarray:
    """Encode official tile grid to (10, 10) int64 class map."""
    grid = np.zeros((10, 10), dtype=np.int64)
    if not raw_tiles:
        return grid
    for y in range(min(10, len(raw_tiles))):
        row = raw_tiles[y]
        for x in range(min(10, len(row))):
            tile = row[x]
            if tile is None:
                grid[y, x] = TILE_CLASS["EMPTY"]
            elif tile == "LOCKED":
                grid[y, x] = TILE_CLASS["LOCKED"]
            elif isinstance(tile, dict):
                kind = tile.get("kind")
                if kind == "WEED":
                    grid[y, x] = TILE_CLASS["WEED"]
                elif kind == "PLANT":
                    crop = tile.get("crop", "")
                    grid[y, x] = CROP_TO_TILE_CLASS.get(crop, TILE_CLASS["OTHER"])
                else:
                    grid[y, x] = TILE_CLASS["OTHER"]
            else:
                grid[y, x] = TILE_CLASS["OTHER"]
    return grid


def observation_step_index(
    obs: Dict[str, Any],
    turns_per_cycle: int = COMPETITION_TURNS_PER_DAY,
) -> int:
    """Simulation step index (0–719 for a 720-step season).

    Defaults to competition ``turnsPerDay=24`` so historical episode tapes and
    reference-ladder agents stay aligned. Pass ``TURNS_PER_CYCLE`` (72) for the
    kinematic self-play profile.
    """
    if "step" in obs and obs["step"] is not None:
        return int(obs["step"])
    day = int(obs.get("day", 1) or 1)
    hour = int(obs.get("hour", 0) or 0)
    return max(0, (day - 1) * int(turns_per_cycle) + hour)


def parse_observation(agent_result: Dict[str, Any], player_id: Optional[int] = None) -> Dict[str, Any]:
    """Extract nested observation dict from a Kaggle agent state."""
    obs = agent_result.get("observation", agent_result)
    if player_id is None:
        pid = agent_result.get("id", obs.get("player", 0))
        if isinstance(pid, str) and pid.startswith("p"):
            try:
                player_id = int(pid[1:])
            except ValueError:
                player_id = 0
        else:
            player_id = int(pid) if pid is not None else 0
    parsed = {
        "player": player_id,
        "day": obs.get("day", 0),
        "hour": obs.get("hour", 0),
        "step": observation_step_index(obs),
        "farms": obs.get("farms", []),
        "market": obs.get("market", {}),
        "private": obs.get("private", {}),
        "town": obs.get("town", {}),
    }
    return parsed


def encode_observation(
    observation: Dict[str, Any],
    player_id: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Convert raw Kaggle observation JSON to tensor dict for main DQN."""
    farms = observation.get("farms", []) or []
    private = observation.get("private", {}) or {}
    market = observation.get("market", {}) or {}
    farm = farms[player_id] if len(farms) > player_id else {}

    tiles = encode_tiles(farm.get("tiles", []))

    prices = market.get("prices", {})
    if isinstance(prices, dict):
        market_prices = [float(prices.get(crop, 0.0)) for crop in CROPS]
    elif isinstance(prices, list):
        market_prices = (list(prices[:5]) + [0.0] * 5)[:5]
    else:
        market_prices = [0.0] * 5

    inventory = market.get("inventory", {})
    if isinstance(inventory, dict):
        market_inventory = [float(inventory.get(crop, 0.0)) for crop in CROPS]
    else:
        market_inventory = [0.0] * 5

    seeds = private.get("seeds", {}) or {}
    seed_values = [float(seeds.get(crop, 0.0)) for crop in CROPS]

    shed = private.get("shed", {}) or {}
    shed_values = [float(shed.get(crop, 0.0)) for crop in CROPS]

    inv_list: List[float] = []
    inventories = private.get("inventories", []) or []
    for hand in inventories[:NUM_HANDS]:
        if isinstance(hand, dict):
            inv_list.extend(float(hand.get(crop, 0.0)) for crop in CROPS)
        else:
            inv_list.extend([0.0] * len(CROPS))
    inv_list.extend([0.0] * (30 - len(inv_list)))

    opp_money = float(farms[1 - player_id].get("money", 0.0)) if len(farms) > 1 else 0.0

    return {
        "tiles": torch.tensor(tiles, device=device, dtype=torch.long),
        "day": torch.tensor([float(observation.get("day", 0))], device=device),
        "hour": torch.tensor([float(observation.get("hour", 0))], device=device),
        "player_id": torch.tensor([float(player_id)], device=device),
        "farms_p0_money": torch.tensor([float(farm.get("money", 0.0))], device=device),
        "farms_p1_money": torch.tensor([opp_money], device=device),
        "market_prices": torch.tensor(market_prices, device=device, dtype=torch.float32),
        "market_inventory": torch.tensor(market_inventory, device=device, dtype=torch.float32),
        "seeds": torch.tensor(seed_values, device=device, dtype=torch.float32),
        "shed": torch.tensor(shed_values, device=device, dtype=torch.float32),
        "inventories": torch.tensor(inv_list[:30], device=device, dtype=torch.float32),
    }


def encode_path_b_observation(
    observation: Dict[str, Any],
    player_id: int,
) -> Dict[str, np.ndarray]:
    """Convert official observation to Path B float tiles (18,10,10) + numeric(55).

    Tile channels (fixed order; PATH_B_TILE_CHANNELS must match this list):
      0  my farmer           9  WEED
      1  opponent farmer    10  LOCKED
      2  watered_today      11  COOP structure
      3  WHEAT              12  PASTURE structure
      4  CARROT             13  animal present
      5  TOMATO             14  fed_today
      6  STRAWBERRY         15  cared_today
      7  MELON              16  fertilizer_available
      8  yield_units (/6)   17  fertilized (plant)
    """
    farms = observation.get("farms", []) or []
    private = observation.get("private", {}) or {}
    market = observation.get("market", {}) or {}
    my_farm = farms[player_id] if len(farms) > player_id else {}
    opp_farm = farms[1 - player_id] if len(farms) > 1 - player_id else {}

    tiles_grid = np.zeros((PATH_B_TILE_CHANNELS, 10, 10), dtype=np.float32)
    my_farmer = my_farm.get("farmer", [0, 0])
    opp_farmer = opp_farm.get("farmer", [0, 0])

    if len(my_farmer) >= 2:
        fx, fy = int(my_farmer[0]), int(my_farmer[1])
        if 0 <= fy < 10 and 0 <= fx < 10:
            tiles_grid[0, fy, fx] = 1.0
    if len(opp_farmer) >= 2:
        fx, fy = int(opp_farmer[0]), int(opp_farmer[1])
        if 0 <= fy < 10 and 0 <= fx < 10:
            tiles_grid[1, fy, fx] = 1.0

    raw_tiles = my_farm.get("tiles", [])
    day = int(observation.get("day", 0) or 0)
    for y in range(min(10, len(raw_tiles))):
        row = raw_tiles[y]
        for x in range(min(10, len(row))):
            tile = row[x]
            if tile == "LOCKED":
                tiles_grid[10, y, x] = 1.0
            elif isinstance(tile, dict):
                if tile.get("watered_today"):
                    tiles_grid[2, y, x] = 1.0
                kind = tile.get("kind")
                if kind == "PLANT":
                    crop = tile.get("crop")
                    if crop in CROPS:
                        tiles_grid[3 + CROPS.index(crop), y, x] = 1.0
                    progress = tile.get("yield_units", 0)
                    if isinstance(progress, (int, float)):
                        tiles_grid[8, y, x] = min(1.0, float(progress) / 6.0)
                    if int(tile.get("fertilized_until_day", 0) or 0) >= day:
                        tiles_grid[17, y, x] = 1.0
                elif kind == "WEED":
                    tiles_grid[9, y, x] = 1.0
                elif kind in ("COOP", "PASTURE"):
                    tiles_grid[11 if kind == "COOP" else 12, y, x] = 1.0
                    if tile.get("animal"):
                        tiles_grid[13, y, x] = 1.0
                    if tile.get("fed_today"):
                        tiles_grid[14, y, x] = 1.0
                    if tile.get("cared_today"):
                        tiles_grid[15, y, x] = 1.0
                    if tile.get("fertilizer_available"):
                        tiles_grid[16, y, x] = 1.0
                    progress = tile.get("yield_units", 0)
                    if isinstance(progress, (int, float)):
                        tiles_grid[8, y, x] = min(1.0, float(progress) / 6.0)

    numeric: List[float] = []
    numeric.append(float(observation.get("day", 0)) / float(SEASON_DAYS))
    numeric.append(float(observation.get("hour", 0)) / float(COMPETITION_TURNS_PER_DAY))
    my_money = float(my_farm.get("money", 0.0))
    opp_money = float(opp_farm.get("money", 0.0))
    numeric.extend([
        np.tanh(my_money / 1000.0),
        np.tanh(opp_money / 1000.0),
        np.tanh((my_money - opp_money) / 1000.0),
    ])
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    for crop in CROPS:
        numeric.append(float(seeds.get(crop, 0)) / 100.0)
    for crop in CROPS:
        numeric.append(float(shed.get(crop, 0)) / 100.0)
    prices = market.get("prices", {})
    inv = market.get("inventory", {})
    for crop in CROPS:
        price = prices.get(crop, 10.0) if isinstance(prices, dict) else 10.0
        qty = inv.get(crop, 0.0) if isinstance(inv, dict) else 0.0
        numeric.append(float(price) / 100.0)
        numeric.append(float(qty) / 10000.0)
    shop_types = ["BAKERY", "PIZZA", "GROCERY", "BREWERY", "MILL", "JUICE_BAR", "SALAD_BAR"]
    unlocked = observation.get("town", {}).get("unlocked_shops", []) or []
    for shop in shop_types:
        numeric.append(1.0 if shop in unlocked else 0.0)
    numeric.append(len(my_farm.get("hands", [])) / 6.0)
    numeric.append(len(opp_farm.get("hands", [])) / 6.0)
    # Per-hand normalized positions (6 × 2) so the hand heads can coordinate
    # spatially instead of sharing one position-blind latent.
    my_hands = my_farm.get("hands", []) or []
    for i in range(6):
        if i < len(my_hands) and isinstance(my_hands[i], (list, tuple)) and len(my_hands[i]) >= 2:
            numeric.append(float(my_hands[i][0]) / 10.0)
            numeric.append(float(my_hands[i][1]) / 10.0)
        else:
            numeric.append(0.0)
            numeric.append(0.0)
    # Animal / fertilizer products held in the shed.
    for item in ("EGG", "MILK", "WOOL", "FERTILIZER"):
        numeric.append(float(shed.get(item, 0)) / 100.0)
    while len(numeric) < 55:
        numeric.append(0.0)

    return {"tiles": tiles_grid, "numeric": np.array(numeric[:55], dtype=np.float32)}


def encode_action(action_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map Kaggle command-list action to branched integer indices."""
    farmer_act = _branch_action(action_raw.get("farmer", ["PASS"]))

    hands_act: List[int] = []
    hands = action_raw.get("hands", []) or []
    for hand_idx in range(NUM_HANDS):
        if hand_idx < len(hands):
            hands_act.append(_branch_action(hands[hand_idx]))
        else:
            hands_act.append(FARMER_ACTIONS["PASS"])

    market_orders = action_raw.get("market", []) or []
    if market_orders and isinstance(market_orders[0], list) and market_orders[0]:
        market_op = market_orders[0][0]
    else:
        market_op = "PASS"
    market_act = MARKET_ACTIONS.get(market_op, 0)

    return {"farmer": farmer_act, "hands": hands_act, "market": market_act}


def encode_path_b_action(
    action_raw: Dict[str, Any],
    max_market_orders: int = 10,
) -> Dict[str, Any]:
    """Map Kaggle command-list action to Path B hierarchical indices."""
    farmer_raw = action_raw.get("farmer", ["PASS"]) or ["PASS"]
    verb_idx = _branch_action(farmer_raw)

    crop_idx = 0
    if isinstance(farmer_raw, list) and farmer_raw:
        op = _normalize_action_op(farmer_raw[0])
        if op == "PLANT" and len(farmer_raw) > 1:
            crop_name = farmer_raw[1]
            crop_idx = CROPS.index(crop_name) if crop_name in CROPS else 0

    hands_indices: List[int] = []
    hands = action_raw.get("hands", []) or []
    for hand_idx in range(NUM_HANDS):
        if hand_idx < len(hands):
            hands_indices.append(_branch_action(hands[hand_idx]))
        else:
            hands_indices.append(FARMER_ACTIONS["PASS"])

    market_indices = [MARKET_ACTIONS["PASS"]] * max_market_orders
    market_orders = action_raw.get("market", []) or []
    for i, order in enumerate(market_orders[:max_market_orders]):
        if isinstance(order, list) and order:
            op = order[0]
            market_indices[i] = MARKET_ACTIONS.get(op, MARKET_ACTIONS["PASS"])
        else:
            market_indices[i] = MARKET_ACTIONS["PASS"]

    return {
        "verb": verb_idx,
        "crop": crop_idx,
        "hands": np.array(hands_indices, dtype=np.int64),
        "market": np.array(market_indices, dtype=np.int64),
    }


def _best_plant_crop(observation: Dict[str, Any]) -> str:
    seeds = observation.get("private", {}).get("seeds", {}) or {}
    for crop in CROPS:
        if seeds.get(crop, 0) > 0:
            return crop
    return "WHEAT"


def _best_sell_crop(observation: Dict[str, Any]) -> Optional[str]:
    shed = observation.get("private", {}).get("shed", {}) or {}
    for crop in CROPS:
        if shed.get(crop, 0) > 0:
            return crop
    return None


def _best_buy_seed_crop(observation: Dict[str, Any]) -> Optional[str]:
    money = 0.0
    farms = observation.get("farms", []) or []
    player = int(observation.get("player", 0) or 0)
    if len(farms) > player:
        money = float(farms[player].get("money", 0.0) or 0.0)
    day = int(observation.get("day", 1) or 1)
    seeds = observation.get("private", {}).get("seeds", {}) or {}

    if day <= 4:
        pref = ["CARROT", "WHEAT"]
    elif day <= 18:
        pref = ["CARROT", "WHEAT", "TOMATO"]
    elif day <= 24:
        pref = ["CARROT", "WHEAT"]
    else:
        pref = ["WHEAT"]

    for c in pref:
        if seeds.get(c, 0) < 6 and money >= SEED_COSTS.get(c, 10):
            return c
    return pref[0] if money >= SEED_COSTS.get(pref[0], 10) else "WHEAT"


def _resolve_animal_or_place_verb(
    verb: str,
    observation: Dict[str, Any],
    pos: Optional[Tuple[int, int]] = None,
) -> List[Any]:
    """Map FEED/PICKUP indices onto FEED / CARE / COLLECT_FERTILIZER / PLACE when legal."""
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", []) or []
    farm = farms[player] if len(farms) > player else {}
    tiles = farm.get("tiles", []) or []
    if pos is None:
        farmer_pos = farm.get("farmer", [0, 0]) or [0, 0]
        pos = (int(farmer_pos[0]), int(farmer_pos[1])) if len(farmer_pos) >= 2 else (0, 0)
    fx, fy = pos
    tile = None
    if 0 <= fy < len(tiles) and 0 <= fx < len(tiles[fy]):
        tile = tiles[fy][fx]

    if verb in ("FEED", "CARE", "COLLECT_FERTILIZER") and isinstance(tile, dict):
        kind = tile.get("kind")
        if kind in ("COOP", "PASTURE") and tile.get("animal"):
            if not tile.get("fed_today", False):
                return ["FEED"]
            if int(tile.get("fertilizer_available", 0) or 0) > 0:
                return ["COLLECT_FERTILIZER"]
            if not tile.get("cared_today", False):
                return ["CARE"]
            return ["FEED"]

    if verb in ("PICKUP", "PLACE") and isinstance(tile, dict):
        kind = tile.get("kind")
        if kind in ("COOP", "PASTURE") and not tile.get("animal"):
            invs = observation.get("private", {}).get("inventories", []) or []
            main_inv = invs[0] if invs else {}
            if isinstance(main_inv, dict):
                for animal in ("GOOSE", "COW", "SHEEP"):
                    if int(main_inv.get(animal, 0) or 0) > 0:
                        return ["PLACE", animal]
    return [verb]


def decode_farmer_verb(verb_idx: int, crop_idx: int, observation: Dict[str, Any]) -> List[Any]:
    """Decode farmer verb index (+ optional crop) to Kaggle command list."""
    verb = FARMER_INDEX_TO_VERB[min(max(verb_idx, 0), NUM_FARMER_ACTIONS - 1)]
    if verb == "PLANT":
        crop = CROPS[min(max(crop_idx, 0), len(CROPS) - 1)]
        if crop not in CROPS or observation.get("private", {}).get("seeds", {}).get(crop, 0) <= 0:
            crop = _best_plant_crop(observation)
        return ["PLANT", crop]
    if verb in ("FEED", "PICKUP", "FERTILIZE"):
        if verb == "FERTILIZE":
            return ["FERTILIZE"]
        return _resolve_animal_or_place_verb(verb, observation)
    return [verb]


def decode_hand_verb(
    hand_idx: int,
    crop_idx: int = 0,
    observation: Optional[Dict[str, Any]] = None,
    hand_pos: Optional[Tuple[int, int]] = None,
    faithful: bool = False,
) -> List[Any]:
    verb = FARMER_INDEX_TO_VERB[min(max(hand_idx, 0), NUM_HAND_ACTIONS - 1)]
    if verb == "PLANT":
        crop = CROPS[min(max(crop_idx, 0), len(CROPS) - 1)]
        return ["PLANT", crop]
    # Faithful mode decodes the exact indexed verb so the transition the buffer
    # stores ("action_hands = idx") is the command the engine actually executed.
    # Previously, verbs like BUILD_COOP/DROP/PICKUP silently became PASS (or were
    # rewritten to heuristic WATER/HARVEST/DIG), mislabeling every such transition.
    if faithful or verb in (
        "PASS", "DIG", "WATER", "HARVEST", "FERTILIZE",
        "NORTH", "SOUTH", "WEST", "EAST",
        "DROP", "PICKUP", "BUILD_COOP", "BUILD_PASTURE", "FEED",
    ):
        if verb in ("FEED", "PICKUP") and observation is not None:
            return _resolve_animal_or_place_verb(verb, observation, hand_pos)
        return [verb]

    # Smart hand fallback for watering and harvest assistance.
    if observation is not None and hand_pos is not None:
        hx, hy = hand_pos
        player = int(observation.get("player", 0) or 0)
        farms = observation.get("farms", []) or []
        farm = farms[player] if len(farms) > player else {}
        tiles = farm.get("tiles", []) or []
        if 0 <= hy < len(tiles) and 0 <= hx < len(tiles[hy]):
            tile = tiles[hy][hx]
            if isinstance(tile, dict):
                if tile.get("kind") == "PLANT":
                    if not tile.get("watered_today", False):
                        return ["WATER"]
                    if tile.get("yield_units", 0) >= 3:
                        return ["HARVEST"]
                elif tile.get("kind") == "WEED":
                    return ["DIG"]

            # Step toward nearest unwatered plant
            best_target = None
            min_dist = 999
            for ty in range(min(5, len(tiles))):
                for tx in range(min(5, len(tiles[ty]))):
                    t = tiles[ty][tx]
                    if isinstance(t, dict) and t.get("kind") == "PLANT" and not t.get("watered_today", False):
                        d = abs(tx - hx) + abs(ty - hy)
                        if d < min_dist:
                            min_dist = d
                            best_target = (tx, ty)
            if best_target:
                tx, ty = best_target
                if tx < hx:
                    return ["WEST"]
                if tx > hx:
                    return ["EAST"]
                if ty < hy:
                    return ["NORTH"]
                if ty > hy:
                    return ["SOUTH"]

    return ["PASS"]


def decode_market_verb(market_idx: int, observation: Dict[str, Any]) -> List[Any]:
    """Decode single market index to one order or empty list."""
    idx = min(max(market_idx, 0), NUM_MARKET_ACTIONS - 1)
    verb = MARKET_INDEX_TO_VERB[idx]
    if verb == "PASS" or verb == "OTHER":
        return []
    if verb == "BUY_SEED":
        crop = _best_buy_seed_crop(observation)
        if crop:
            seeds = observation.get("private", {}).get("seeds", {}) or {}
            qty = 8 if seeds.get(crop, 0) == 0 else 4
            return [["BUY_SEED", crop, qty]]
        return []
    if verb == "SELL":
        crop = _best_sell_crop(observation)
        if crop:
            shed = observation.get("private", {}).get("shed", {}) or {}
            qty = min(40, int(shed.get(crop, 40) or 40))
            if qty > 0:
                return [["SELL", crop, qty]]
        return []
    if verb == "HIRE":
        return [["HIRE"]]
    if verb == "BUY_LAND":
        return [["BUY_LAND"]]
    if verb == "BUY_PRODUCT":
        money = 0.0
        farms = observation.get("farms", []) or []
        player = int(observation.get("player", 0) or 0)
        if len(farms) > player:
            money = float(farms[player].get("money", 0.0) or 0.0)
        prices = observation.get("market", {}).get("prices", {}) or {}
        # Only WHEAT and FERTILIZER are buyable via BUY_PRODUCT in the engine.
        for item in ("FERTILIZER", "WHEAT"):
            price = float(prices.get(item, 9999) or 9999)
            if money >= price:
                return [["BUY_PRODUCT", item, 1]]
        return []
    if verb == "BUY_ANIMAL":
        money = 0.0
        farms = observation.get("farms", []) or []
        player = int(observation.get("player", 0) or 0)
        if len(farms) > player:
            money = float(farms[player].get("money", 0.0) or 0.0)
        for animal, cost in (("GOOSE", 300), ("COW", 400), ("SHEEP", 500)):
            if money >= cost:
                return [["BUY_ANIMAL", animal, 1]]
        return []
    return []


def decode_action(
    action_indices: Dict[str, Any],
    observation: Dict[str, Any],
    *,
    faithful: bool = False,
) -> Dict[str, Any]:
    """Decode branched integer action to official Kaggle command dict.

    ``faithful=True`` (Path B / RL loop) decodes every hand to exactly the indexed
    verb so stored actions equal executed actions.
    """
    crop_idx = int(action_indices.get("crop", action_indices.get("action_crop", 0)))
    farmer = decode_farmer_verb(
        int(action_indices["farmer"]),
        crop_idx,
        observation,
    )

    hands_raw = action_indices.get("hands", [])
    active_hands = []
    if observation.get("farms"):
        player = observation.get("player", 0)
        farm = observation["farms"][player] if len(observation["farms"]) > player else {}
        active_hands = farm.get("hands", []) or []

    hands_out: List[List[Any]] = []
    for i in range(len(active_hands)):
        h_idx = int(hands_raw[i]) if i < len(hands_raw) else 0
        h_pos = None
        hand_obj = active_hands[i]
        if isinstance(hand_obj, dict) and "pos" in hand_obj:
            h_pos = (int(hand_obj["pos"][0]), int(hand_obj["pos"][1]))
        elif isinstance(hand_obj, (list, tuple)) and len(hand_obj) >= 2:
            h_pos = (int(hand_obj[0]), int(hand_obj[1]))
        hands_out.append(decode_hand_verb(
            h_idx,
            crop_idx=crop_idx,
            observation=observation,
            hand_pos=h_pos,
            faithful=faithful,
        ))

    market_indices = action_indices.get("market")
    market_orders: List[List[Any]] = []
    if isinstance(market_indices, (list, np.ndarray)):
        for m_idx in market_indices:
            orders = decode_market_verb(int(m_idx), observation)
            if not orders:
                break
            market_orders.extend(orders)
    else:
        market_orders = decode_market_verb(int(market_indices or 0), observation)

    return {"farmer": farmer, "hands": hands_out, "market": market_orders}


def decode_path_b_action(
    verb_idx: int,
    crop_idx: int,
    hands_indices: List[int],
    market_indices: List[int],
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """Decode Path B hierarchical indices via shared decode logic.

    Hands decode faithfully (faithful=True) so the integer stored in the replay
    buffer is exactly the command the engine executed — no heuristic rewrites.
    """
    return decode_action(
        {
            "farmer": verb_idx,
            "crop": crop_idx,
            "hands": hands_indices,
            "market": market_indices,
        },
        observation,
        faithful=True,
    )


def get_action_masks(observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Build action masks from nested observation using canonical indices."""
    player = observation.get("player", 0)
    farms = observation.get("farms", []) or []
    my_farm = farms[player] if len(farms) > player else {}
    private = observation.get("private", {}) or {}
    money = float(my_farm.get("money", 0.0) or 0.0)
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}

    farmer_mask = np.zeros(NUM_FARMER_ACTIONS, dtype=bool)
    farmer_mask[0] = True  # PASS

    tiles = my_farm.get("tiles", [])
    farmer_pos = my_farm.get("farmer", [0, 0]) or [0, 0]
    if len(farmer_pos) >= 2:
        fx, fy = int(farmer_pos[0]), int(farmer_pos[1])
    else:
        fx, fy = 0, 0

    day = int(observation.get("day", 1) or 1)

    has_seeds = any(seeds.get(c, 0) > 0 for c in CROPS)
    if 0 <= fy < len(tiles) and 0 <= fx < len(tiles[fy]):
        tile = tiles[fy][fx]
        if isinstance(tile, dict):
            kind = tile.get("kind")
            if kind == "WEED":
                farmer_mask[1] = True  # DIG
            elif kind == "PLANT":
                if not tile.get("watered_today", False):
                    farmer_mask[2] = True  # WATER
                if plant_is_harvestable(tile, day):
                    farmer_mask[4] = True  # HARVEST
                fert = private.get("shed", {}).get("FERTILIZER", 0) or 0
                invs = private.get("inventories", []) or []
                main_inv = invs[0] if invs else {}
                if isinstance(main_inv, dict):
                    fert = max(int(fert), int(main_inv.get("FERTILIZER", 0) or 0))
                if fert > 0:
                    farmer_mask[FARMER_ACTIONS["FERTILIZE"]] = True
            elif kind in ("COOP", "PASTURE") and tile.get("animal"):
                farmer_mask[FARMER_ACTIONS["FEED"]] = True
            elif kind not in ("LOCKED",) and has_seeds and day <= 26:
                farmer_mask[3] = True  # PLANT
        elif tile in ("EMPTY", "", None) and has_seeds and day <= 26:
            farmer_mask[3] = True  # PLANT

    for dx, dy, idx in [(0, -1, 5), (0, 1, 6), (-1, 0, 7), (1, 0, 8)]:
        nx, ny = fx + dx, fy + dy
        if 0 <= nx < 10 and 0 <= ny < 10:
            if ny < len(tiles) and nx < len(tiles[ny]) and tiles[ny][nx] != "LOCKED":
                farmer_mask[idx] = True

    crop_mask = np.zeros(len(CROPS), dtype=bool)
    for i, crop in enumerate(CROPS):
        if seeds.get(crop, 0) > 0:
            crop_mask[i] = True

    total_seeds = sum(int(seeds.get(c, 0) or 0) for c in CROPS)
    market_mask = np.zeros(NUM_MARKET_ACTIONS, dtype=bool)
    market_mask[0] = True
    for crop in CROPS:
        if money >= SEED_COSTS.get(crop, 999) and day <= 24 and total_seeds < 10:
            market_mask[1] = True  # BUY_SEED
            break
    if any(shed.get(c, 0) > 0 for c in CROPS):
        market_mask[4] = True  # SELL

    fert_price = observation.get("market", {}).get("prices", {}).get("FERTILIZER", 100.0)
    if isinstance(fert_price, (int, float)) and money >= fert_price and day <= 20 and money >= 500:
        market_mask[MARKET_ACTIONS["BUY_PRODUCT"]] = True

    unlocked = my_farm.get("unlocked_quadrants", [])
    if isinstance(unlocked, list) and len(unlocked) < 4:
        land_idx = max(0, len(unlocked) - 1)
        if land_idx < len(LAND_PRICES) and money >= LAND_PRICES[land_idx] and day <= 20:
            market_mask[MARKET_ACTIONS["BUY_LAND"]] = True

    hires_today = int(my_farm.get("hires_today", 0) or 0)
    hire_cost = hire_cost_today(hires_today)
    hands = my_farm.get("hands", []) or []
    if money >= hire_cost and len(hands) < 4 and day <= 24 and money >= 10:
        market_mask[MARKET_ACTIONS["HIRE"]] = True

    if money >= 800 and day <= 20:
        market_mask[MARKET_ACTIONS["BUY_ANIMAL"]] = True

    # Per-hand verb masks: hands act at their own tile position. Hands beyond the
    # currently-hired count get a PASS-only mask (they never produce an action at
    # decode time). This lets the network and exploration stay legal per hand.
    hand_masks = np.zeros((NUM_HANDS, NUM_FARMER_ACTIONS), dtype=bool)
    hands = my_farm.get("hands", []) or []
    for h_idx in range(NUM_HANDS):
        hand_masks[h_idx, FARMER_ACTIONS["PASS"]] = True
        if h_idx >= len(hands):
            continue
        h_pos = hands[h_idx]
        if not isinstance(h_pos, (list, tuple)) or len(h_pos) < 2:
            continue
        hx, hy = int(h_pos[0]), int(h_pos[1])
        if 0 <= hy < len(tiles) and 0 <= hx < len(tiles[hy]):
            htile = tiles[hy][hx]
            if isinstance(htile, dict):
                hkind = htile.get("kind")
                if hkind == "WEED":
                    hand_masks[h_idx, FARMER_ACTIONS["DIG"]] = True
                elif hkind == "PLANT":
                    if not htile.get("watered_today", False):
                        hand_masks[h_idx, FARMER_ACTIONS["WATER"]] = True
                    if plant_is_harvestable(htile, day):
                        hand_masks[h_idx, FARMER_ACTIONS["HARVEST"]] = True
                    hand_masks[h_idx, FARMER_ACTIONS["FERTILIZE"]] = True
                elif hkind in ("COOP", "PASTURE") and htile.get("animal"):
                    hand_masks[h_idx, FARMER_ACTIONS["FEED"]] = True
                elif hkind not in ("LOCKED",) and has_seeds and day <= 26:
                    hand_masks[h_idx, FARMER_ACTIONS["PLANT"]] = True
            elif htile in ("EMPTY", "", None) and has_seeds and day <= 26:
                hand_masks[h_idx, FARMER_ACTIONS["PLANT"]] = True
        for dx, dy, idx in [(0, -1, FARMER_ACTIONS["NORTH"]), (0, 1, FARMER_ACTIONS["SOUTH"]),
                            (-1, 0, FARMER_ACTIONS["WEST"]), (1, 0, FARMER_ACTIONS["EAST"])]:
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < 10 and 0 <= ny < 10 and ny < len(tiles) and nx < len(tiles[ny]) and tiles[ny][nx] != "LOCKED":
                hand_masks[h_idx, idx] = True

    return {
        "farmer_verb": farmer_mask,
        "crop_parameter": crop_mask,
        "market": market_mask,
        "hands": hand_masks,
    }


def final_bank_coins(observation: Dict[str, Any], player_id: int) -> float:
    """Return final bank coins for rubric win/loss comparison."""
    farms = observation.get("farms", []) or []
    if len(farms) > player_id:
        return float(farms[player_id].get("money", 0.0))
    return 0.0


def compare_episode_outcome(
    obs_p0: Dict[str, Any],
    obs_p1: Dict[str, Any],
) -> Tuple[int, int, int]:
    """Return (p0_wins, p1_wins, ties) for rubric-aligned outcome."""
    m0 = final_bank_coins(obs_p0, 0)
    m1 = final_bank_coins(obs_p1, 1)
    if m0 > m1:
        return 1, 0, 0
    if m1 > m0:
        return 0, 1, 0
    return 0, 0, 1


def mlx_is_available() -> bool:
    """True when Apple MLX Metal backend is importable and available."""
    if importlib.util.find_spec("mlx.core") is None:
        return False
    try:
        mx = importlib.import_module("mlx.core")
        metal = getattr(mx, "metal", None)
        return bool(metal and metal.is_available())
    except Exception:
        return False


def resolve_training_device(device_name: str = "auto") -> torch.device:
    """Resolve PyTorch device: cuda → mps → cpu. ``mlx`` maps to mps on Apple Silicon."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "mlx":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def gpu_backend_diagnostics() -> Dict[str, Any]:
    """Summarize CUDA, MLX, and MPS availability for notebooks and logging."""
    diag: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "mlx_available": mlx_is_available(),
        "resolved_device": str(resolve_training_device("auto")),
    }
    if diag["cuda_available"]:
        diag["cuda_device"] = torch.cuda.get_device_name(0)
    if diag["mlx_available"]:
        try:
            mx = importlib.import_module("mlx.core")
            diag["mlx_default_device"] = str(mx.default_device())
        except Exception as exc:
            diag["mlx_error"] = str(exc)
    return diag


def any_accelerator_available() -> bool:
    """True when CUDA, PyTorch MPS, or MLX Metal is available."""
    return (
        torch.cuda.is_available()
        or torch.backends.mps.is_available()
        or mlx_is_available()
    )
