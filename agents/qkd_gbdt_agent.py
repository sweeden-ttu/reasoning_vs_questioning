"""QKD GBDT Reinforcement Learning Agent for 120-Day Scenarios.

Features:
  - 10x10x10 Voxel Spatial-Depth Tensor Feature Extraction (1,000 spatial voxels)
  - 3 Canonical QKD Statistical Probes (Q_DAYS_REMAINING, K_OPP_WALLET_BALANCE, D_SUBAGENTS_WALLET_BALANCE)
  - 32 Dense Global Economic & Market Features (Total Dim = 1,035)
  - Multi-Head Gradient Boosted Decision Tree (GBDT) Decision Engine (~100 MB capacity)
  - Strict Two-Stage Attention Interface:
      Stage 1: Att(obs, config, market_functions)
      Stage 2: Att(obs, config, market_functions, opponent_functions)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RVQ_DIR = Path(__file__).resolve().parent.parent
if str(RVQ_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from market_config_suite import (
        InitialTerminalConfiguration,
        build_initial_terminal_configuration,
        build_market_functions,
        build_opponent_functions,
    )
    from qkd_gbdt_vector_policy import QKDGBDTPolicy
    from voxel_state_extractor import VoxelStateExtractor
except ImportError:
    from reasoning_vs_questioning.market_config_suite import (
        InitialTerminalConfiguration,
        build_initial_terminal_configuration,
        build_market_functions,
        build_opponent_functions,
    )
    from reasoning_vs_questioning.qkd_gbdt_vector_policy import QKDGBDTPolicy
    from reasoning_vs_questioning.voxel_state_extractor import VoxelStateExtractor

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "qkd_gbdt_120day_100mb.pkl",
)


class QKDGBDTAgent:
    """Standardized QKD GBDT RL Agent for 120-Day Macro Scenarios."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        auto_load: bool = True,
    ) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.extractor = VoxelStateExtractor()
        
        # Load or initialize policy
        if auto_load and os.path.exists(self.model_path):
            try:
                self.policy = QKDGBDTPolicy.load(self.model_path)
            except Exception:
                self.policy = QKDGBDTPolicy(n_estimators=100)
        else:
            self.policy = QKDGBDTPolicy(n_estimators=100)

        self._last_stage1_att: Optional[Any] = None
        self._last_stage2_att: Optional[Any] = None

    def Att(
        self,
        obs: Dict[str, Any],
        configuration: Optional[Union[InitialTerminalConfiguration, Dict[str, Any]]] = None,
        market_functions: Optional[Dict[str, Callable[[int], float]]] = None,
        opponent_functions: Optional[Dict[str, Callable[[Dict[str, Any], int], float]]] = None,
    ) -> Dict[str, Any]:
        """Two-Stage Attention Pipeline:
        Stage 1: Att(obs, config, market_functions) -> Initial Market Equilibrium Focus
        Stage 2: Att(obs, config, market_functions, opponent_functions) -> Opponent Adjusted Focus
        """
        config = (
            configuration
            if isinstance(configuration, InitialTerminalConfiguration)
            else build_initial_terminal_configuration(obs, configuration)
        )
        if market_functions is None:
            market_functions = build_market_functions(obs, config)

        if opponent_functions is None:
            # Stage 1: Market functions only
            self._last_stage1_att = self.policy.select_action(
                obs, config, market_functions, None
            )
            return self._last_stage1_att
        else:
            # Stage 2: Market + Opponent functions
            self._last_stage2_att = self.policy.select_action(
                obs, config, market_functions, opponent_functions
            )
            return self._last_stage2_att

    def act(
        self,
        obs: Dict[str, Any],
        configuration: Optional[Union[InitialTerminalConfiguration, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Standardized decision entrypoint executing Stage 1 then Stage 2."""
        config = (
            configuration
            if isinstance(configuration, InitialTerminalConfiguration)
            else build_initial_terminal_configuration(obs, configuration)
        )
        market_functions = build_market_functions(obs, config)
        opponent_functions = build_opponent_functions(obs, config)

        # Stage 1
        self.Att(obs, config, market_functions)
        # Stage 2
        final_action = self.Att(obs, config, market_functions, opponent_functions)
        return final_action


# Singleton Agent instance for competition runner
_global_agent_instance: Optional[QKDGBDTAgent] = None


def get_agent() -> QKDGBDTAgent:
    global _global_agent_instance
    if _global_agent_instance is None:
        _global_agent_instance = QKDGBDTAgent()
    return _global_agent_instance


def agent(obs: Dict[str, Any], configuration: Any = None) -> Dict[str, Any]:
    """Kaggle Environment standard entrypoint callable."""
    inst = get_agent()
    return inst.act(obs, configuration)
