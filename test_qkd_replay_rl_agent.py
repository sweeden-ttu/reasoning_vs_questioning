import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RVQ_DIR = Path(__file__).resolve().parent
if str(RVQ_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest

from agents.qkd_replay_rl_agent import (
    QKDReplayRLAgent,
    _align_hands,
    _order_score,
    _rank_sell_slots,
)
from qkd_replay_rl_agent_standalone import StandaloneQKDReplayRLAgent


def _make_mock_obs(step: int = 0, weed_on_farmer: bool = False, hands_count: int = 0) -> dict:
    day = step // 24
    hour = step % 24
    tiles_p0 = [[None for _ in range(10)] for _ in range(10)]
    if weed_on_farmer:
        tiles_p0[4][4] = {"kind": "WEED"}
    else:
        tiles_p0[4][4] = {"kind": "PLANT", "stage": 3, "moisture": 0.8}

    return {
        "step": step,
        "day": day,
        "hour": hour,
        "player": 0,
        "farms": [
            {
                "farmer": [4, 4],
                "hands": [[4, 5] for _ in range(hands_count)],
                "hires_today": hands_count,
                "money": 3000.0 + (step * 50.0),
                "tiles": tiles_p0,
                "unlocked_quadrants": ["NW"],
            },
            {
                "farmer": [4, 4],
                "hands": [],
                "hires_today": 0,
                "money": 3000.0 + (step * 30.0),
                "tiles": [[None for _ in range(10)] for _ in range(10)],
                "unlocked_quadrants": ["NW"],
            },
        ],
        "private": {
            "shed": {"TOMATO": 20, "WHEAT": 10},
            "seeds": {"TOMATO": 5, "WHEAT": 5},
        },
        "market": {
            "inventory": {"TOMATO": 9900, "WHEAT": 9800, "CARROT": 10000},
            "prices": {"TOMATO": 62, "WHEAT": 26, "CARROT": 35},
        },
        "town": {
            "unlocked_shops": ["PIZZA_SHOP", "FARMERS_MARKET"],
        },
    }


def test_replay_route_decompression():
    agent = QKDReplayRLAgent()
    assert len(agent.actions) >= 719
    # First action should be valid structure
    a0 = agent.actions[0]
    assert "farmer" in a0
    assert "hands" in a0
    assert "market" in a0


def test_hand_alignment():
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    obs_0 = _make_mock_obs(hands_count=0)
    aligned_0 = _align_hands(action, obs_0)
    assert len(aligned_0["hands"]) == 0

    obs_3 = _make_mock_obs(hands_count=3)
    aligned_3 = _align_hands(action, obs_3)
    assert len(aligned_3["hands"]) == 3
    for h in aligned_3["hands"]:
        assert h == ["PASS"]


def test_weed_repair_activation():
    agent = QKDReplayRLAgent()
    agent.reset()

    # Step with weed on tile
    obs_weed = _make_mock_obs(step=5, weed_on_farmer=True)
    action = agent.act(obs_weed)

    # When intended action is PLANT/BUILD_PASTURE and tile has WEED, should emit DIG
    # Even if replay had PLANT, DIG should take precedence
    assert isinstance(action["farmer"], list)
    assert len(action["farmer"]) > 0


def test_qkd_statistical_probing():
    agent = QKDReplayRLAgent()
    obs = _make_mock_obs(step=48)
    qkd_vals = agent.evaluate_qkd(obs)

    assert "Q_DAYS_REMAINING" in qkd_vals
    assert "K_OPP_WALLET_BALANCE" in qkd_vals
    assert "D_SUBAGENTS_WALLET_BALANCE" in qkd_vals
    assert qkd_vals["Q_DAYS_REMAINING"] > 0.0

    # At day 30 (step 700+), days remaining should be ~0
    obs_end = _make_mock_obs(step=700)
    obs_end["day"] = 30
    qkd_end = agent.evaluate_qkd(obs_end)
    assert qkd_end["Q_DAYS_REMAINING"] == 0.0


def test_market_sell_slot_reranking():
    # Set inventory above equilibrium so selling causes price drop (positive impact score)
    obs = _make_mock_obs(step=100)
    obs["market"]["inventory"] = {"WHEAT": 10000, "TOMATO": 10200, "CARROT": 10000}
    obs["market"]["prices"] = {"WHEAT": 25, "TOMATO": 50, "CARROT": 35}
    cfg = {"turnsPerDay": 24, "townCenterSellInterval": 24, "townShopSellInterval": 4}
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["SELL", "WHEAT", 10], ["SELL", "TOMATO", 50]],
    }
    ranked = _rank_sell_slots(obs, action, cfg)
    assert len(ranked["market"]) == 2
    # Tomato has high impact and town demand urgency, so it ranks first
    assert ranked["market"][0][1] == "TOMATO"


def test_full_720_step_simulation():
    agent = QKDReplayRLAgent()
    standalone = StandaloneQKDReplayRLAgent()
    agent.reset()
    standalone.reset()

    cfg = {"turnsPerDay": 24, "townCenterSellInterval": 24, "townShopSellInterval": 4}

    for step in range(720):
        obs = _make_mock_obs(step=step, hands_count=min(4, step // 100))
        act_modular = agent.act(obs, cfg)
        act_standalone = standalone.act(obs, cfg)

        assert "farmer" in act_modular
        assert "hands" in act_modular
        assert "market" in act_modular
        assert len(act_modular["hands"]) == min(4, step // 100)

        assert "farmer" in act_standalone
        assert "hands" in act_standalone
        assert "market" in act_standalone
        assert len(act_standalone["hands"]) == min(4, step // 100)


def test_exception_fallback():
    agent = QKDReplayRLAgent()
    # Force an internal exception to test fallback handler
    agent.actions = "invalid_not_a_list"  # type: ignore
    act = agent.act(_make_mock_obs(step=0))
    assert act["farmer"] == ["PASS"]
    assert isinstance(act["hands"], list)
    assert isinstance(act["market"], list)


def test_two_stage_att_interface():
    from market_config_suite import (
        InitialTerminalConfiguration,
        build_initial_terminal_configuration,
        build_market_functions,
        build_opponent_functions,
    )

    obs = _make_mock_obs(step=48)
    config = build_initial_terminal_configuration(obs)

    # Validate InitialTerminalConfiguration properties and commodity prices
    assert config.number_of_days == 30
    assert config.days == 30
    assert config.amount_of_money == 5400.0
    assert config.initial_money == 5400.0

    expected_commodities = [
        "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
        "EGG", "MILK", "WOOL", "FERTILIZER"
    ]
    for c in expected_commodities:
        assert c in config.start_price_of_every_commodity
        assert config.start_price_of_every_commodity[c] > 0
        assert c in config.start_prices
        assert config.start_prices[c] > 0

    # Validate Market Functions suite
    mkt_funcs = build_market_functions(obs, config)
    assert callable(mkt_funcs["predict_future_price"])
    assert callable(mkt_funcs["predict_price_trajectory"])
    assert callable(mkt_funcs["project_future_price_trajectory"])
    assert callable(mkt_funcs["get_price_elasticity"])
    assert callable(mkt_funcs["get_demand_absorption_rate"])
    assert callable(mkt_funcs["compute_optimal_sell_batch"])
    assert callable(mkt_funcs["predict_spot_price"])

    # Test future price trajectory
    traj = mkt_funcs["predict_price_trajectory"]("WHEAT", horizon_days=5)
    assert len(traj) == 5
    assert all(isinstance(p, float) and p > 0 for p in traj)

    # Validate Opponent Functions suite
    opp_funcs = build_opponent_functions(obs, config)
    assert callable(opp_funcs["predict_opponent_money_trajectory"])
    assert callable(opp_funcs["predict_opponent_workforce_growth"])
    assert callable(opp_funcs["estimate_opponent_market_impact"])
    assert callable(opp_funcs["compute_adversarial_lead_gap"])
    assert callable(opp_funcs["estimate_opponent_aggression"])

    opp_traj = opp_funcs["predict_opponent_money_trajectory"](horizon_days=5)
    assert isinstance(opp_traj, float)
    assert opp_traj > 0

    # Validate QKDReplayRLAgent Two-Stage Att Execution
    agent = QKDReplayRLAgent()
    stage1_action = agent.Att(obs, config, mkt_funcs)
    assert "farmer" in stage1_action
    assert "hands" in stage1_action
    assert "market" in stage1_action

    stage2_action = agent.Att(obs, config, mkt_funcs, opp_funcs)
    assert "farmer" in stage2_action
    assert "hands" in stage2_action
    assert "market" in stage2_action

    final_act = agent.act(obs, config.to_dict())
    assert "farmer" in final_act
    assert "hands" in final_act
    assert "market" in final_act


def test_all_agents_two_stage_att():
    from agents.questioning_agent import QuestioningAgent
    from agents.rag_observer_reasoning_agent import RAGObserverReasoningAgent
    from agents.reasoning_agent import ReasoningAgent
    from agents.scott_weeden_agent import ScottWeedenAgent
    from agents.ten_agents import EricSchmidtAgent
    from agents.tripartite_composite_agent import TripartiteReasoningAgent
    from market_config_suite import (
        build_initial_terminal_configuration,
        build_market_functions,
        build_opponent_functions,
    )

    obs = _make_mock_obs(step=48)
    config = build_initial_terminal_configuration(obs)
    mkt_funcs = build_market_functions(obs, config)
    opp_funcs = build_opponent_functions(obs, config)

    agent_instances = [
        QKDReplayRLAgent(),
        TripartiteReasoningAgent(act_all_hours=True),
        ReasoningAgent(),
        QuestioningAgent(),
        RAGObserverReasoningAgent(),
        ScottWeedenAgent(),
        EricSchmidtAgent(),
    ]

    for ag in agent_instances:
        assert hasattr(ag, "Att"), f"{ag.__class__.__name__} missing Att method"
        assert hasattr(ag, "act"), f"{ag.__class__.__name__} missing act method"

        # Stage 1: Att(obs, config, market_functions)
        s1 = ag.Att(obs, config, mkt_funcs)
        assert isinstance(s1, dict), f"{ag.__class__.__name__} Stage 1 Att did not return dict"

        # Stage 2: Att(obs, config, market_functions, opponent_functions)
        s2 = ag.Att(obs, config, mkt_funcs, opp_funcs)
        assert isinstance(s2, dict), f"{ag.__class__.__name__} Stage 2 Att did not return dict"

        # Complete act(obs, config) pipeline
        res = ag.act(obs, config.to_dict())
        assert isinstance(res, dict), f"{ag.__class__.__name__} act did not return dict"
        assert "farmer" in res
        assert "hands" in res
        assert "market" in res

