"""10x10x10 Spatial-Depth Voxel State Extractor for Kaggriculture.

Constructs a 3D Voxel Tensor of shape (10, 10, 10) = 1,000 spatial-depth features:
  - Dim 0: Spatial X in [0..9]
  - Dim 1: Spatial Y in [0..9]
  - Dim 2: Depth Channel Z in [0..9]:
      Z=0: Tile type (0=Locked, 1=Empty soil, 2=Planted crop, 3=Weed, 4=Pasture/Coop)
      Z=1: Crop/Entity ID (0=None, 1=Wheat, 2=Carrot, 3=Tomato, 4=Strawberry,
                           5=Melon, 6=Egg/Chicken, 7=Milk/Cow, 8=Wool/Sheep, 9=Fertilizer)
      Z=2: Crop maturity / growth stage ratio in [0.0, 1.0]
      Z=3: Soil Moisture level in [0.0, 1.0]
      Z=4: Weed infestation risk flag (0.0 or 1.0)
      Z=5: Worker Proximity field (exp(-0.25*d_farmer) + exp(-0.25*d_hands))
      Z=6: Farm quadrant expansion stage (0..3)
      Z=7: Commodity price forecast gradient from market_functions
      Z=8: Town shop demand elasticity & absorption rate for tile entity
      Z=9: Opponent pressure & contestation field from opponent_functions

Vector Feature Concatenation:
  - 1,000 Voxel elements (flattened 10x10x10 tensor)
  - 3 Canonical QKD probes (Q_DAYS_REMAINING, K_OPP_WALLET_BALANCE, D_SUBAGENTS_WALLET_BALANCE)
  - 32 Dense global economic & game features
  Total Feature Vector Dimension: D = 1,035.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from market_config_suite import (
        InitialTerminalConfiguration,
        build_initial_terminal_configuration,
        build_market_functions,
        build_opponent_functions,
        market_price,
    )
    from qkd_statistical_questions import CANONICAL_QKD_QUESTIONS, QKDStatisticalQuestionBank
except ImportError:
    from reasoning_vs_questioning.market_config_suite import (
        InitialTerminalConfiguration,
        build_initial_terminal_configuration,
        build_market_functions,
        build_opponent_functions,
        market_price,
    )
    from reasoning_vs_questioning.qkd_statistical_questions import (
        CANONICAL_QKD_QUESTIONS,
        QKDStatisticalQuestionBank,
    )

# Commodity mapping
COMMODITY_TO_ID = {
    "NONE": 0,
    "WHEAT": 1,
    "CARROT": 2,
    "TOMATO": 3,
    "STRAWBERRY": 4,
    "MELON": 5,
    "EGG": 6,
    "MILK": 7,
    "WOOL": 8,
    "FERTILIZER": 9,
}

ID_TO_COMMODITY = {v: k for k, v in COMMODITY_TO_ID.items()}

COMMODITIES_LIST = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
]


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


class VoxelStateExtractor:
    """Extracts high-dimensional 10x10x10 voxel tensors and 1,035-dim state vectors."""

    def __init__(
        self,
        grid_height: int = 10,
        grid_width: int = 10,
        depth_channels: int = 10,
    ) -> None:
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.depth_channels = depth_channels
        self.voxel_count = grid_height * grid_width * depth_channels  # 1,000
        self.qkd_probes_count = len(CANONICAL_QKD_QUESTIONS)  # 3
        self.global_features_count = 32
        self.total_feature_dim = self.voxel_count + self.qkd_probes_count + self.global_features_count  # 1,035
        self.question_bank = QKDStatisticalQuestionBank()

    def extract_voxel_tensor(
        self,
        obs: Dict[str, Any],
        configuration: Optional[Union[InitialTerminalConfiguration, Dict[str, Any]]] = None,
        market_functions: Optional[Dict[str, Callable[[int], float]]] = None,
        opponent_functions: Optional[Dict[str, Callable[[Dict[str, Any], int], float]]] = None,
    ) -> np.ndarray:
        """Extract the (10, 10, 10) 3D Voxel Tensor from game observation.
        
        Returns:
            np.ndarray of shape (10, 10, 10) and dtype float32.
        """
        config = (
            configuration
            if isinstance(configuration, InitialTerminalConfiguration)
            else build_initial_terminal_configuration(obs, configuration)
        )
        if market_functions is None:
            market_functions = build_market_functions(obs, config)
        if opponent_functions is None:
            opponent_functions = build_opponent_functions(obs, config)

        tensor = np.zeros(
            (self.grid_height, self.grid_width, self.depth_channels),
            dtype=np.float32,
        )

        player_idx = int(_get(obs, "player", 0) or 0)
        farms = list(_get(obs, "farms", []) or [])
        my_farm = farms[player_idx] if player_idx < len(farms) else {}
        opp_farm = farms[1 - player_idx] if (1 - player_idx) < len(farms) else {}

        # 1. Grid / Plot Information
        grid = _get(my_farm, "grid", None) or _get(my_farm, "plots", None) or []
        farmer_pos = _get(my_farm, "farmer_pos", [0, 0]) or [0, 0]
        hands = list(_get(my_farm, "hands", []) or [])
        worker_positions = [farmer_pos] + [
            _get(h, "pos", [0, 0]) or [0, 0] for h in hands if isinstance(h, dict)
        ]

        # 2. Market and Prices
        market = _get(obs, "market", {}) or {}
        inventory = _get(market, "inventory", {}) or {}
        prices = _get(market, "prices", {}) or {}

        # Compute price trends from market_functions
        price_trends: Dict[str, float] = {}
        for item in COMMODITIES_LIST:
            fn = market_functions.get(item)
            curr_inv = int(_get(inventory, item, 10000) or 10000)
            curr_p = float(_get(prices, item, market_price(item, curr_inv)) or 10.0)
            if callable(fn):
                future_p = float(fn(curr_inv))
                price_trends[item] = (future_p - curr_p) / max(1.0, curr_p)
            else:
                price_trends[item] = 0.0

        # Opponent threat field
        opp_threat_fn = opponent_functions.get("threat_level")
        opp_threat = float(opp_threat_fn(obs, 0)) if callable(opp_threat_fn) else 0.5

        # Populate the 10x10 grid voxels
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                tile_data = None
                if isinstance(grid, list):
                    idx = y * self.grid_width + x
                    if idx < len(grid):
                        tile_data = grid[idx]
                elif isinstance(grid, dict):
                    tile_data = grid.get(f"{x},{y}") or grid.get((x, y))

                # Defaults
                tile_type = 1.0  # Open Soil
                crop_id = 0.0
                maturity = 0.0
                moisture = 0.5
                weed_flag = 0.0
                quadrant = min(3.0, (x // 5) + 2 * (y // 5))

                if isinstance(tile_data, dict):
                    raw_type = str(_get(tile_data, "type", "SOIL")).upper()
                    if "LOCK" in raw_type or "BLOCKED" in raw_type:
                        tile_type = 0.0
                    elif "WEED" in raw_type:
                        tile_type = 3.0
                        weed_flag = 1.0
                    elif "PASTURE" in raw_type or "COOP" in raw_type or "BARN" in raw_type:
                        tile_type = 4.0
                    elif "CROP" in raw_type or "PLANT" in raw_type or _get(tile_data, "crop"):
                        tile_type = 2.0

                    crop_name = str(_get(tile_data, "crop", "NONE")).upper()
                    crop_id = float(COMMODITY_TO_ID.get(crop_name, 0))
                    maturity = float(_get(tile_data, "growth", _get(tile_data, "maturity", 0.0)) or 0.0)
                    moisture = float(_get(tile_data, "moisture", _get(tile_data, "water", 0.5)) or 0.5)
                    weed_flag = 1.0 if bool(_get(tile_data, "has_weed", False)) or tile_type == 3.0 else 0.0

                # Worker proximity potential
                worker_prox = 0.0
                for wx, wy in worker_positions:
                    dist = math.sqrt((x - wx) ** 2 + (y - wy) ** 2)
                    worker_prox += math.exp(-0.25 * dist)

                # Commodity price forecast gradient
                crop_key = ID_TO_COMMODITY.get(int(crop_id), "WHEAT")
                price_gradient = float(price_trends.get(crop_key, 0.0))

                # Town demand elasticity
                town_demand_fn = market_functions.get(f"{crop_key}_demand")
                if callable(town_demand_fn):
                    demand_score = float(town_demand_fn(int(_get(inventory, crop_key, 10000) or 10000)))
                else:
                    demand_score = 1.0

                # Opponent contestation pressure
                opp_pressure = opp_threat * math.exp(-0.1 * (x + y))

                # Fill the 10 depth channels
                tensor[y, x, 0] = tile_type / 4.0  # Normalize to [0, 1]
                tensor[y, x, 1] = crop_id / 9.0  # Normalize to [0, 1]
                tensor[y, x, 2] = max(0.0, min(1.0, maturity))
                tensor[y, x, 3] = max(0.0, min(1.0, moisture))
                tensor[y, x, 4] = weed_flag
                tensor[y, x, 5] = min(2.0, worker_prox) / 2.0
                tensor[y, x, 6] = quadrant / 3.0
                tensor[y, x, 7] = math.tanh(price_gradient)
                tensor[y, x, 8] = math.tanh(demand_score / 10.0)
                tensor[y, x, 9] = max(0.0, min(1.0, opp_pressure))

        return tensor

    def extract_qkd_probes(
        self,
        obs: Dict[str, Any],
        configuration: Optional[Union[InitialTerminalConfiguration, Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """Compute the canonical 3-probe QKD vector:
        1. Q_DAYS_REMAINING
        2. K_OPP_WALLET_BALANCE
        3. D_SUBAGENTS_WALLET_BALANCE
        """
        try:
            from qkd_statistical_questions import map_observation_to_qkd
        except ImportError:
            from reasoning_vs_questioning.qkd_statistical_questions import map_observation_to_qkd

        qkd_map = map_observation_to_qkd(obs)
        return qkd_map.vector.astype(np.float32)

    def extract_global_economic_features(
        self,
        obs: Dict[str, Any],
        configuration: Optional[Union[InitialTerminalConfiguration, Dict[str, Any]]] = None,
        market_functions: Optional[Dict[str, Callable[[int], float]]] = None,
    ) -> np.ndarray:
        """Extract 32 dense global economic and game state features."""
        config = (
            configuration
            if isinstance(configuration, InitialTerminalConfiguration)
            else build_initial_terminal_configuration(obs, configuration)
        )
        features: List[float] = []

        player_idx = int(_get(obs, "player", 0) or 0)
        farms = list(_get(obs, "farms", []) or [])
        my_farm = farms[player_idx] if player_idx < len(farms) else {}
        opp_farm = farms[1 - player_idx] if (1 - player_idx) < len(farms) else {}

        # 1. Wallets
        my_money = float(_get(my_farm, "money", _get(my_farm, "cash", 3000.0)) or 3000.0)
        opp_money = float(_get(opp_farm, "money", _get(opp_farm, "cash", 3000.0)) or 3000.0)
        features.append(math.log1p(max(0.0, my_money)) / 12.0)
        features.append(math.log1p(max(0.0, opp_money)) / 12.0)

        # 2. Time, Days, and Season
        step = int(_get(obs, "step", 0) or 0)
        turns_per_day = int(_get(config, "turns_per_day", 24) or 24)
        total_days = int(_get(config, "number_of_days", 120) or 120)
        current_day = step // turns_per_day
        season_idx = (current_day // 30) % 4
        day_of_season = current_day % 30

        features.append(float(step) / max(1.0, float(total_days * turns_per_day)))
        features.append(float(current_day) / max(1.0, float(total_days)))
        features.append(float(season_idx) / 3.0)
        features.append(float(day_of_season) / 29.0)

        # 3. Market Inventories & Prices for all 9 Commodities
        market = _get(obs, "market", {}) or {}
        inventory = _get(market, "inventory", {}) or {}
        prices = _get(market, "prices", {}) or {}

        for item in COMMODITIES_LIST:
            inv = float(_get(inventory, item, 10000) or 10000)
            prc = float(_get(prices, item, market_price(item, int(inv))) or 25.0)
            features.append(inv / 20000.0)
            features.append(math.log1p(max(1.0, prc)) / 6.0)

        # 4. Warehouse / Stock of my farm
        my_warehouse = _get(my_farm, "warehouse", {}) or _get(my_farm, "inventory", {}) or {}
        total_stock = sum(
            float(_get(my_warehouse, item, 0) or 0) for item in COMMODITIES_LIST
        )
        features.append(math.log1p(total_stock) / 10.0)

        # 5. Worker Hands count
        hands = list(_get(my_farm, "hands", []) or [])
        features.append(float(len(hands)) / 10.0)

        # Pad or trim strictly to 32 features
        while len(features) < self.global_features_count:
            features.append(0.0)
        return np.array(features[: self.global_features_count], dtype=np.float32)

    def extract_full_state_vector(
        self,
        obs: Dict[str, Any],
        configuration: Optional[Union[InitialTerminalConfiguration, Dict[str, Any]]] = None,
        market_functions: Optional[Dict[str, Callable[[int], float]]] = None,
        opponent_functions: Optional[Dict[str, Callable[[Dict[str, Any], int], float]]] = None,
    ) -> np.ndarray:
        """Construct the complete 1,035-dimensional state feature vector:
        - 1,000 Voxel elements (10x10x10 tensor flattened)
        - 3 Canonical QKD probes
        - 32 Dense global economic features
        
        Returns:
            np.ndarray of shape (1035,) and dtype float32.
        """
        voxel_tensor = self.extract_voxel_tensor(
            obs, configuration, market_functions, opponent_functions
        )
        flat_voxels = voxel_tensor.reshape(-1)  # (1000,)
        qkd_probes = self.extract_qkd_probes(obs, configuration)  # (3,)
        global_features = self.extract_global_economic_features(
            obs, configuration, market_functions
        )  # (32,)

        full_vector = np.concatenate([flat_voxels, qkd_probes, global_features])
        return full_vector.astype(np.float32)
