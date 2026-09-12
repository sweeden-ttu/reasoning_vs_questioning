"""Multi-Phase Imitation Opponent Agent.

Implements a dynamic multi-phase opponent policy that transitions across:
  - Iteration 1 (or 1st half of replay buffer): Imitates Player 0 from Replay Buffer
  - Iteration 2: Imitates Player 1 from Replay Buffer
  - Iteration 3: Imitates Subagent 2 (Scott Weeden Referee / Auditor)
  - Iteration 4: Transitional Interim Policy (Referee & Land Expansion Blend)
  - Iteration 5: Imitates Subagent 7 (Land Expansion & Weed Suppression)
  - Iteration 6: Imitates Opponent 4 (Melon Mateo / Reference Ladder Tier 4)
  - Iteration > 6: Configurable cycle or adaptive hold

Supports both:
  1. Multi-episode iteration scheduling (iteration-based phase transitions)
  2. Single-episode intra-match phasing (step-based phase transitions)
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RVQ_DIR = Path(__file__).resolve().parent.parent
if str(RVQ_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from agents.ten_agents import (
        LandExpansionSubagent,
        ScottWeedenSubagent,
        build_all_10_subagents,
    )
    from market_config_suite import (
        InitialTerminalConfiguration,
        build_initial_terminal_configuration,
        build_market_functions,
        build_opponent_functions,
    )
    from replay_qkd_extractor import ReplayQKDExtractor
except ImportError:
    from reasoning_vs_questioning.agents.ten_agents import (
        LandExpansionSubagent,
        ScottWeedenSubagent,
        build_all_10_subagents,
    )
    from reasoning_vs_questioning.market_config_suite import (
        InitialTerminalConfiguration,
        build_initial_terminal_configuration,
        build_market_functions,
        build_opponent_functions,
    )
    from reasoning_vs_questioning.replay_qkd_extractor import ReplayQKDExtractor

logger = logging.getLogger("multi_phase_imitation_opponent")

# Default fallback action
PASS_ACTION: Dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}

# Quadrant expansion order and costs
QUADRANT_COSTS = {"NE": 1000.0, "SW": 2000.0, "SE": 4000.0}
QUADRANT_OFFSETS = {
    "NW": (0, 0),
    "NE": (0, 5),
    "SW": (5, 0),
    "SE": (5, 5),
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return default
        return float(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if not s or s in ("empty", "null", "none", "nan", "undefined"):
            return default
        try:
            f = float(val)
            return default if (np.isnan(f) or np.isinf(f)) else f
        except (ValueError, TypeError):
            return default
    return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    f = _safe_float(val, float(default))
    return int(f)


def _safe_dict(val: Any) -> Dict[str, Any]:
    """Ensure value is a dict."""
    return val if isinstance(val, dict) else {}


def _safe_list(val: Any) -> List[Any]:
    """Ensure value is a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, (tuple, set)):
        return list(val)
    return []


class MultiPhaseImitationOpponent:
    """Multi-Phase Imitation Opponent covering Player 0, Player 1, Subagents 2 & 7, and Opponent 4."""

    def __init__(
        self,
        replay_buffer_p0: Optional[Sequence[Dict[str, Any]]] = None,
        replay_buffer_p1: Optional[Sequence[Dict[str, Any]]] = None,
        replays_dir: Optional[Union[Path, str]] = None,
        start_iteration: int = 1,
        mode: str = "iteration",  # "iteration" | "step_phased"
        total_match_steps: int = 720,
    ) -> None:
        self.current_iteration = max(1, start_iteration)
        self.mode = mode
        self.total_match_steps = total_match_steps
        self.current_step = 0
        self.step_history: List[Dict[str, Any]] = []

        # 1. Initialize Replay Buffers for Player 0 and Player 1
        self.replay_p0 = list(replay_buffer_p0) if replay_buffer_p0 is not None else []
        self.replay_p1 = list(replay_buffer_p1) if replay_buffer_p1 is not None else []

        if not self.replay_p0 or not self.replay_p1:
            self._load_default_replay_buffers(replays_dir)

        # 2. Initialize Subagent 2 (Scott Weeden Referee/Auditor) & Subagent 7 (Land Expansion)
        self.subagent_2 = ScottWeedenSubagent()
        self.subagent_7 = LandExpansionSubagent()

        # 3. Initialize Opponent 4 (Melon Mateo)
        self.opponent_4_fn = self._load_opponent_4_callable()

    def _load_default_replay_buffers(self, replays_dir: Optional[Union[Path, str]] = None) -> None:
        """Load Player 0 and Player 1 action sequences from replays/ or champion route."""
        r_dir = Path(replays_dir) if replays_dir else ROOT_DIR / "replays"
        extractor = ReplayQKDExtractor(r_dir)

        # Try loading from high-scoring match replay JSONs
        loaded_p0 = False
        loaded_p1 = False

        if r_dir.exists():
            for json_file in sorted(r_dir.glob("*.json")):
                try:
                    with open(json_file, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    steps = data.get("steps", [])
                    if len(steps) >= 100:
                        p0_actions = [
                            s[0].get("action") for s in steps[1:] if len(s) > 0 and s[0].get("action")
                        ]
                        p1_actions = [
                            s[1].get("action") for s in steps[1:] if len(s) > 1 and s[1].get("action")
                        ]
                        if len(p0_actions) > 50 and not loaded_p0 and not self.replay_p0:
                            self.replay_p0 = p0_actions
                            loaded_p0 = True
                        if len(p1_actions) > 50 and not loaded_p1 and not self.replay_p1:
                            self.replay_p1 = p1_actions
                            loaded_p1 = True
                        if loaded_p0 and loaded_p1:
                            break
                except Exception as e:
                    logger.debug("Failed parsing replay %s: %e", json_file, e)

        # Fallback to decompressed champion route if buffers are empty
        if not self.replay_p0:
            try:
                from agents.qkd_replay_rl_agent import _CHAMPION_REPLAY_ROUTE
                decompressed = extractor.decompress_actions(_CHAMPION_REPLAY_ROUTE)
                self.replay_p0 = decompressed
            except Exception:
                self.replay_p0 = [dict(PASS_ACTION) for _ in range(720)]

        if not self.replay_p1:
            # Shifted / alternate variation for Player 1
            if self.replay_p0:
                self.replay_p1 = [dict(act) for act in reversed(self.replay_p0)]
            else:
                self.replay_p1 = [dict(PASS_ACTION) for _ in range(720)]

    def _load_opponent_4_callable(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Dynamically load Tier 4 reference opponent (Melon Mateo or Homestead Hana)."""
        try:
            from opponents.melon_mateo import agent as melon_agent
            return melon_agent
        except Exception:
            try:
                from opponents.homestead_hana import agent as hana_agent
                return hana_agent
            except Exception:
                return self._fallback_melon_mateo_policy

    def _fallback_melon_mateo_policy(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Self-contained melon metering and fertilizer policy fallback."""
        obs_dict = _safe_dict(obs)
        player = _safe_int(obs_dict.get("player", 0), 0)
        farms = _safe_list(obs_dict.get("farms", []))
        me = _safe_dict(farms[player]) if len(farms) > player else {}
        priv = _safe_dict(obs_dict.get("private", {}))
        shed = _safe_dict(priv.get("shed", {}))
        day = _safe_int(obs_dict.get("day", 0), 0)

        market: List[List[Any]] = []
        # Buy melon seeds early
        if day <= 16 and float(me.get("money", 0) or 0) >= 300:
            market.append(["BUY_SEED", "MELON", 6])
            market.append(["BUY_SEED", "WHEAT", 4])

        # Sell melons above floor
        melon_count = _safe_int(shed.get("MELON", 0), 0)
        if melon_count >= 12:
            market.append(["SELL", "MELON", 12])

        wheat_count = _safe_int(shed.get("WHEAT", 0), 0)
        if wheat_count >= 20:
            market.append(["SELL", "WHEAT", 15])

        # Default movement/harvest
        return {"farmer": ["PASS"], "hands": [], "market": market}

    def determine_active_phase(self, step_idx: int, day: int) -> Tuple[str, int]:
        """Determine current active imitation phase name and target index.
        
        Phases:
          1. 'player_0_replay_half' (Iteration 1: Player 0 for 1st half of replay buffer)
          2. 'player_1_replay'      (Iteration 2: Player 1 imitation)
          3. 'subagent_2_referee'   (Iteration 3: Subagent 2 - Scott Weeden)
          4. 'interim_transition'   (Iteration 4: Transition blend)
          5. 'subagent_7_expansion' (Iteration 5: Subagent 7 - Land Expansion)
          6. 'opponent_4_melon'     (Iteration 6: Opponent 4 - Melon Mateo)
        """
        if self.mode == "step_phased":
            # Map turns [0, total_match_steps) to the 6 phases
            p1_cutoff = int(self.total_match_steps * 0.25)
            p2_cutoff = int(self.total_match_steps * 0.50)
            p3_cutoff = int(self.total_match_steps * 0.70)
            p4_cutoff = int(self.total_match_steps * 0.80)
            p5_cutoff = int(self.total_match_steps * 0.90)

            if step_idx < p1_cutoff:
                return "player_0_replay_half", 1
            elif step_idx < p2_cutoff:
                return "player_1_replay", 2
            elif step_idx < p3_cutoff:
                return "subagent_2_referee", 3
            elif step_idx < p4_cutoff:
                return "interim_transition", 4
            elif step_idx < p5_cutoff:
                return "subagent_7_expansion", 5
            else:
                return "opponent_4_melon", 6

        # Standard Iteration-Phased Mode
        iter_num = self.current_iteration
        if iter_num == 1:
            return "player_0_replay_half", 1
        elif iter_num == 2:
            return "player_1_replay", 2
        elif iter_num == 3:
            return "subagent_2_referee", 3
        elif iter_num == 4:
            return "interim_transition", 4
        elif iter_num == 5:
            return "subagent_7_expansion", 5
        elif iter_num == 6:
            return "opponent_4_melon", 6
        else:
            # Cycle through phases 1..6
            cycle_idx = ((iter_num - 1) % 6) + 1
            return self._phase_name_by_index(cycle_idx), cycle_idx

    @staticmethod
    def _phase_name_by_index(idx: int) -> str:
        mapping = {
            1: "player_0_replay_half",
            2: "player_1_replay",
            3: "subagent_2_referee",
            4: "interim_transition",
            5: "subagent_7_expansion",
            6: "opponent_4_melon",
        }
        return mapping.get(idx, "player_0_replay_half")

    def _act_player_0_replay(self, obs: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
        """Imitate Player 0 from Replay Buffer (First half of buffer)."""
        if not self.replay_p0:
            return dict(PASS_ACTION)
        half_len = max(1, len(self.replay_p0) // 2)
        # Select action from the first half of the replay buffer
        target_idx = step_idx % half_len
        action = self.replay_p0[target_idx]
        return self._sanitize_action(action)

    def _act_player_1_replay(self, obs: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
        """Imitate Player 1 on Second Iteration."""
        if not self.replay_p1:
            return self._act_player_0_replay(obs, step_idx)
        target_idx = step_idx % len(self.replay_p1)
        action = self.replay_p1[target_idx]
        return self._sanitize_action(action)

    def _act_subagent_2_referee(self, obs: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
        """Imitate Subagent 2 (Scott Weeden Referee/Auditor) on Third Iteration.
        
        Audits weed status, checks tile integrity, and executes weed removal or maintenance.
        """
        advice = self.subagent_2.advise(obs, question="Audit weed count and farm health")
        obs_dict = _safe_dict(obs)
        player = _safe_int(obs_dict.get("player", 0), 0)
        farms = _safe_list(obs_dict.get("farms", []))
        my_farm = _safe_dict(farms[player]) if len(farms) > player else {}
        tiles = _safe_list(my_farm.get("tiles", []))

        # Check for weeds and target them for digging
        dig_action: Optional[List[Any]] = None
        for r_idx, row in enumerate(tiles):
            for c_idx, tile in enumerate(_safe_list(row)):
                t_dict = _safe_dict(tile)
                if str(t_dict.get("kind", "")).upper() == "WEED":
                    dig_action = ["DIG", r_idx, c_idx]
                    break
            if dig_action:
                break

        market_ops: List[List[Any]] = []
        priv = _safe_dict(obs_dict.get("private", {}))
        shed = _safe_dict(priv.get("shed", {}))
        wheat = _safe_int(shed.get("WHEAT", 0), 0)
        if wheat > 10:
            market_ops.append(["SELL", "WHEAT", min(20, wheat)])

        return {
            "farmer": dig_action or ["PASS"],
            "hands": [],
            "market": market_ops,
            "advice": advice,
        }

    def _act_subagent_7_expansion(self, obs: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
        """Imitate Subagent 7 (Land Expansion & Weed Suppression) on Fifth Iteration.
        
        Unlocks quadrants (NE=1000, SW=2000, SE=4000) and clears weeds in new zones.
        """
        advice = self.subagent_7.advise(obs, question="Check affordable quadrant expansions")
        obs_dict = _safe_dict(obs)
        player = _safe_int(obs_dict.get("player", 0), 0)
        farms = _safe_list(obs_dict.get("farms", []))
        my_farm = _safe_dict(farms[player]) if len(farms) > player else {}
        money = _safe_float(my_farm.get("money", 0.0), 0.0)
        unlocked = _safe_list(my_farm.get("unlocked_quadrants", [0]))

        market_ops: List[List[Any]] = []
        farmer_op: List[Any] = ["PASS"]

        # Check if we can unlock next quadrant in order: NE (1), SW (2), SE (3)
        if 1 not in unlocked and money >= QUADRANT_COSTS["NE"]:
            market_ops.append(["UNLOCK", "NE"])
        elif 2 not in unlocked and money >= QUADRANT_COSTS["SW"]:
            market_ops.append(["UNLOCK", "SW"])
        elif 3 not in unlocked and money >= QUADRANT_COSTS["SE"]:
            market_ops.append(["UNLOCK", "SE"])

        # Hire workforce if land expanded
        if len(unlocked) > 1 and money >= 500:
            market_ops.append(["HIRE"])

        # Dig any weeds in unlocked quadrants
        tiles = _safe_list(my_farm.get("tiles", []))
        for r_idx, row in enumerate(tiles):
            for c_idx, tile in enumerate(_safe_list(row)):
                t_dict = _safe_dict(tile)
                if str(t_dict.get("kind", "")).upper() == "WEED":
                    farmer_op = ["DIG", r_idx, c_idx]
                    break
            if farmer_op[0] == "DIG":
                break

        return {
            "farmer": farmer_op,
            "hands": [],
            "market": market_ops,
            "advice": advice,
        }

    def _act_interim_transition(self, obs: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
        """Intermediate transition policy for Iteration 4."""
        # Blend Subagent 2 weed defense with Subagent 7 expansion check
        if step_idx % 2 == 0:
            return self._act_subagent_2_referee(obs, step_idx)
        return self._act_subagent_7_expansion(obs, step_idx)

    def _act_opponent_4_melon(self, obs: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
        """Imitate Opponent 4 (Melon Mateo / Tier 4 Reference Ladder) on Sixth Iteration."""
        try:
            act_res = self.opponent_4_fn(obs)
            return self._sanitize_action(act_res)
        except Exception as e:
            logger.debug("Opponent 4 execution error: %s, falling back", e)
            return self._fallback_melon_mateo_policy(obs)

    def _sanitize_action(self, action: Any) -> Dict[str, Any]:
        """Defensively sanitize and format action dictionary."""
        if not isinstance(action, dict):
            return dict(PASS_ACTION)
        farmer = action.get("farmer", ["PASS"])
        hands = action.get("hands", [])
        market = action.get("market", [])
        return {
            "farmer": farmer if isinstance(farmer, list) else ["PASS"],
            "hands": hands if isinstance(hands, list) else [],
            "market": market if isinstance(market, list) else [],
        }

    def act(self, obs: Dict[str, Any], configuration: Any = None) -> Dict[str, Any]:
        """Select action according to the active iteration / step phase."""
        obs_dict = _safe_dict(obs)
        day = _safe_int(obs_dict.get("day", 0), 0)
        hour = _safe_int(obs_dict.get("hour", 0), 0)
        step_idx = (day - 1) * 24 + hour if day >= 1 else self.current_step

        phase_name, phase_idx = self.determine_active_phase(step_idx, day)

        if phase_name == "player_0_replay_half":
            action = self._act_player_0_replay(obs_dict, step_idx)
        elif phase_name == "player_1_replay":
            action = self._act_player_1_replay(obs_dict, step_idx)
        elif phase_name == "subagent_2_referee":
            action = self._act_subagent_2_referee(obs_dict, step_idx)
        elif phase_name == "interim_transition":
            action = self._act_interim_transition(obs_dict, step_idx)
        elif phase_name == "subagent_7_expansion":
            action = self._act_subagent_7_expansion(obs_dict, step_idx)
        elif phase_name == "opponent_4_melon":
            action = self._act_opponent_4_melon(obs_dict, step_idx)
        else:
            action = self._act_player_0_replay(obs_dict, step_idx)

        self.current_step += 1
        self.step_history.append({
            "step": step_idx,
            "day": day,
            "hour": hour,
            "iteration": self.current_iteration,
            "phase": phase_name,
            "phase_index": phase_idx,
        })
        return action

    def __call__(self, obs: Dict[str, Any], configuration: Any = None) -> Dict[str, Any]:
        """Kaggle environment direct callable."""
        return self.act(obs, configuration)

    def set_iteration(self, iteration_num: int) -> None:
        """Set the active iteration number (1 to 6+)."""
        self.current_iteration = max(1, int(iteration_num))
        self.current_step = 0

    def next_iteration(self) -> int:
        """Advance to the next iteration and reset match step counter."""
        self.current_iteration += 1
        self.current_step = 0
        return self.current_iteration

    def get_current_phase(self) -> Dict[str, Any]:
        """Return status information for current iteration and active phase."""
        phase_name, phase_idx = self.determine_active_phase(self.current_step, self.current_step // 24)
        return {
            "current_iteration": self.current_iteration,
            "current_step": self.current_step,
            "mode": self.mode,
            "active_phase": phase_name,
            "phase_index": phase_idx,
            "replay_p0_size": len(self.replay_p0),
            "replay_p1_size": len(self.replay_p1),
        }


# Global singleton instance for Kaggle runner / eval
_global_multi_phase_opponent: Optional[MultiPhaseImitationOpponent] = None


def get_multi_phase_opponent(
    start_iteration: int = 1,
    mode: str = "iteration",
) -> MultiPhaseImitationOpponent:
    """Get or create singleton MultiPhaseImitationOpponent instance."""
    global _global_multi_phase_opponent
    if _global_multi_phase_opponent is None:
        _global_multi_phase_opponent = MultiPhaseImitationOpponent(
            start_iteration=start_iteration,
            mode=mode,
        )
    return _global_multi_phase_opponent


def multi_phase_imitation_agent(obs: Dict[str, Any], configuration: Any = None) -> Dict[str, Any]:
    """Kaggle-compatible callable entrypoint for the Multi-Phase Imitation Opponent."""
    opponent = get_multi_phase_opponent()
    return opponent.act(obs, configuration)


# Alias for naming consistency
MultiPhaseImitationOpponentAgent = MultiPhaseImitationOpponent
