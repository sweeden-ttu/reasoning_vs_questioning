"""Agricultural scheduler for Kaggriculture.

Coordinates the farmer and hired hands to execute field operations (planting,
watering, bonus window harvesting, weeding, building, feeding) and market orders.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

BOARD = 10
HALF = BOARD // 2
TURNS_PER_DAY = 24
SHED_CAP = 100

SHED_TILES = [(HALF - 1, HALF - 1), (HALF, HALF - 1), (HALF - 1, HALF), (HALF, HALF)]

CROPS = {
    "WHEAT":      {"seed": 10,  "first": 2,  "max_day": 4,  "interval": 0, "cap": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first": 2,  "max_day": 3,  "interval": 0, "cap": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first": 8,  "max_day": 8,  "interval": 1, "cap": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "max_day": 10, "interval": 2, "cap": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first": 10, "max_day": 12, "interval": 0, "cap": 6, "ongoing": False},
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]

MOVE_OP = {
    (0, -1): "NORTH",
    (0, 1):  "SOUTH",
    (-1, 0): "WEST",
    (1, 0):  "EAST",
}


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(pos: Tuple[int, int], target: Tuple[int, int]) -> List[str]:
    px, py = pos
    tx, ty = target
    if px != tx:
        return [MOVE_OP[(1, 0)] if tx > px else MOVE_OP[(-1, 0)]]
    if py != ty:
        return [MOVE_OP[(0, 1)] if ty > py else MOVE_OP[(0, -1)]]
    return ["PASS"]


def _shed_tile(tiles: List[List[Any]]) -> Tuple[int, int]:
    for (x, y) in SHED_TILES:
        if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) and tiles[y][x] != "LOCKED":
            return (x, y)
    return SHED_TILES[0]


def _open_tiles(tiles: List[List[Any]]) -> List[Tuple[int, int]]:
    sx, sy = _shed_tile(tiles)
    out: List[Tuple[int, int]] = []
    for y in range(min(BOARD, len(tiles))):
        for x in range(min(BOARD, len(tiles[y]))):
            if tiles[y][x] is None:
                out.append((x, y))
    out.sort(key=lambda p: (abs(p[0] - sx) + abs(p[1] - sy), p[1], p[0]))
    return out


def _attainable_yield(crop: str, planted_day: int, day: int) -> int:
    info = CROPS[crop]
    window_start = (info["max_day"] + 1) // 2
    days = max(0, min(info["max_day"], day) - window_start + 1)
    return min(info["cap"], 1 + days)


def _ready_to_harvest(tile: Dict[str, Any], day: int) -> bool:
    crop = tile.get("crop", "WHEAT")
    if crop not in CROPS:
        return False
    info = CROPS[crop]
    age = day - int(tile.get("planted_day", 0) or 0)
    if age < info["first"]:
        return False
    if int(tile.get("yield_units", 0) or 0) <= 0:
        return False
    if info["ongoing"]:
        return True
    if age >= info["max_day"]:
        return True
    return int(tile.get("yield_units", 0) or 0) >= min(
        info["cap"], _attainable_yield(crop, int(tile.get("planted_day", 0) or 0), info["max_day"])
    )


class AgriculturalScheduler:
    """Multi-unit agricultural scheduler for Kaggriculture."""

    def __init__(
        self,
        target_hands: int = 4,
        crops: Optional[List[str]] = None,
        crop_share: Optional[Dict[str, float]] = None,
        target_quadrants: int = 1,
    ) -> None:
        self.target_hands = target_hands
        self.crops = crops or ["WHEAT", "CARROT", "TOMATO"]
        self.crop_share = crop_share or {"WHEAT": 0.5, "CARROT": 0.3, "TOMATO": 0.2}
        self.target_quadrants = target_quadrants
        self.policy: Dict[str, Any] = {
            "hands": target_hands,
            "crops": self.crops,
            "crop_share": self.crop_share,
            "seed_batch": 6,
            "seed_stock": 8,
            "seed_buffer": 80,
            "sell_order": ["TOMATO", "CARROT", "WHEAT"],
            "sell_chunk": 40,
            "shed_pressure": 70,
            "plant_until_day": 26,
            "liquidate_from_day": 28,
        }

    def compute_field_and_market_actions(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)
        player = int(obs.get("player", 0) or 0)
        farms = obs.get("farms", []) or []
        me = farms[player] if len(farms) > player else {}
        priv = obs.get("private", {}) or {}
        tiles = me.get("tiles", []) or []
        money = float(me.get("money", 0.0) or 0.0)
        seeds = priv.get("seeds", {}) or {}
        shed = priv.get("shed", {}) or {}
        prices = (obs.get("market", {}) or {}).get("prices", {}) or {}

        # 1. Collect jobs
        water_jobs, harvest_jobs, dig_jobs, plant_jobs = [], [], [], []
        for y in range(min(BOARD, len(tiles))):
            for x in range(min(BOARD, len(tiles[y]))):
                tile = tiles[y][x]
                if not isinstance(tile, dict):
                    continue
                pos = (x, y)
                kind = tile.get("kind")
                if kind == "WEED":
                    dig_jobs.append({"pos": pos, "op": ["DIG"]})
                elif kind == "PLANT":
                    if not tile.get("watered_today", False):
                        water_jobs.append({"pos": pos, "op": ["WATER"]})
                    if _ready_to_harvest(tile, day):
                        harvest_jobs.append({"pos": pos, "op": ["HARVEST"]})

        # Planting jobs
        empties = _open_tiles(tiles)
        if day <= self.policy["plant_until_day"]:
            reserved = 0
            for crop in self.policy["crops"]:
                have = int(seeds.get(crop, 0) or 0)
                if have <= 0:
                    continue
                for pos in empties[reserved : reserved + have]:
                    plant_jobs.append({"pos": pos, "op": ["PLANT", crop]})
                reserved += have

        # Priority: Water > Harvest > Plant > Dig
        jobs = water_jobs + harvest_jobs + plant_jobs + dig_jobs

        # 2. Multi-unit assignment
        positions = [tuple(me.get("farmer", [0, 0]))] + [tuple(p) for p in (me.get("hands", []) or [])]
        n_units = len(positions)
        ops: List[List[str]] = [["PASS"] for _ in range(n_units)]
        assigned_jobs: Set[Tuple[int, int]] = set()

        for i, u_pos in enumerate(positions):
            best_job = None
            best_dist = 999
            for j in jobs:
                j_pos = j["pos"]
                if j_pos in assigned_jobs:
                    continue
                d = _dist(u_pos, j_pos)
                if d < best_dist:
                    best_dist = d
                    best_job = j

            if best_job is not None:
                assigned_jobs.add(best_job["pos"])
                if u_pos == best_job["pos"]:
                    ops[i] = list(best_job["op"])
                else:
                    ops[i] = _step_toward(u_pos, best_job["pos"])

        farmer_act = ops[0] if ops else ["PASS"]
        hands_act = ops[1:] if len(ops) > 1 else []

        # 3. Market orders
        market_orders: List[List[Any]] = []

        # Sells
        used_shed = sum(int(v or 0) for k, v in shed.items() if k in PRODUCTS)
        pressure = used_shed >= self.policy["shed_pressure"] or day >= self.policy["liquidate_from_day"]
        for item in self.policy["sell_order"]:
            have = int(shed.get(item, 0) or 0)
            if have <= 0:
                continue
            n = min(have, self.policy["sell_chunk"])
            if n > 0:
                market_orders.append(["SELL", item, n])
                money += 0.8 * float(prices.get(item, 10)) * n

        # Hires in morning
        current_hands = len(me.get("hands", []) or [])
        hires_today = int(me.get("hires_today", 0) or 0)
        target_h = self.policy["hands"]
        if day >= 4:
            target_h = max(target_h, 6)
        if day >= 10:
            target_h = max(target_h, 8)

        if hour <= 2 and day <= 27:
            todo = min(target_h - hires_today, 4)
            for _ in range(max(0, todo)):
                hire_cost = hires_today + 1
                if money >= hire_cost:
                    market_orders.append(["HIRE"])
                    hires_today += 1
                    money -= hire_cost

        # Quadrant expansion
        unlocked = me.get("unlocked_quadrants", [0]) or [0]
        if len(unlocked) < (self.target_quadrants + 1) and day <= 16 and money >= 4200:
            market_orders.append(["BUY_LAND"])
            money -= 3000

        # Buy seeds
        if day <= self.policy["plant_until_day"]:
            free_land = len(empties)
            stock_mult = 2 if len(unlocked) > 1 else 1
            for crop in self.policy["crops"]:
                info = CROPS.get(crop, CROPS["WHEAT"])
                have = int(seeds.get(crop, 0) or 0)
                share = self.policy["crop_share"].get(crop, 0.33)
                want = min(int(free_land * share), self.policy["seed_stock"] * stock_mult)
                gap = min(want - have, self.policy["seed_batch"] * stock_mult)
                seed_cost = info["seed"]
                if gap > 0 and money >= seed_cost * gap + self.policy["seed_buffer"]:
                    market_orders.append(["BUY_SEED", crop, gap])
                    money -= seed_cost * gap

        return {
            "farmer": farmer_act,
            "hands": hands_act,
            "market": market_orders,
        }
