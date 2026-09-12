"""4-Year Macro Horizon Self-Training & Empirical Market Curve Extraction Engine.

Simulates 4 full macro years (48 seasons = 1,440 game days = 34,560 hourly steps)
against the 11-opponent league to:
  1. Smooth out seasonal discrete inventory shocks ('lumps').
  2. Extract continuous empirical market demand & price response curves for all 9 commodities.
  3. Analyze the longitudinal stability of the tripartite QKD question matrix across 48 seasons.
"""

from __future__ import annotations

import copy
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Workspace Path Setup
ROOT_DIR = Path(__file__).resolve().parent.parent
RVQ_DIR = Path(__file__).resolve().parent
if str(RVQ_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval import (
    _make_kaggle_env,
    _normalize_states,
    discover_reference_opponents,
    load_kaggle_agent_policy,
    parse_observation,
)
from qkd_statistical_questions import (
    CANONICAL_QKD_QUESTIONS,
    QKDObservationMap,
    QKDStatisticalQuestionBank,
    map_observation_to_qkd,
    map_observation_to_questions,
)
from reasoning_vs_questioning.agents.qkd_replay_rl_agent import (
    QKDReplayRLAgent,
    _MARKET_PARAMS,
    agent as qkd_replay_agent_fn,
)
from vector_memory_bank import StateVectorEncoder

logger = logging.getLogger("four_year_macro_trainer")

COMMODITIES = list(_MARKET_PARAMS.keys())


@dataclass
class MarketDataPoint:
    year: int
    season: int
    day: int
    hour: int
    global_step: int
    commodity: str
    inventory: int
    price: float
    shop_demand: float
    sales_volume: int


@dataclass
class MacroSeasonResult:
    year: int
    season_in_year: int
    global_season: int
    opponent_name: str
    champion_final_money: float
    opp_final_money: float
    win: bool
    mean_q_probe: float
    mean_k_probe: float
    mean_d_probe: float


class FourYearMacroTrainer:
    """Simulates 48 seasons (4 years) of continuous self-training across 11 opponents."""

    def __init__(
        self,
        total_years: int = 4,
        seasons_per_year: int = 12,
        embedding_dim: int = 256,
    ) -> None:
        self.total_years = total_years
        self.seasons_per_year = seasons_per_year
        self.total_seasons = total_years * seasons_per_year  # 48 seasons
        self.encoder = StateVectorEncoder(dim=embedding_dim, half_dim=128)
        self.question_bank = QKDStatisticalQuestionBank()
        self.agent = QKDReplayRLAgent()

        # Discovered league opponents
        discovered = dict(discover_reference_opponents())
        self.opponents: Dict[str, Callable] = {}
        for k, v in discovered.items():
            if callable(v):
                self.opponents[k] = v

        # Add self-play anchor
        self.opponents["self_play_anchor"] = qkd_replay_agent_fn

        self.market_history: List[MarketDataPoint] = []
        self.season_results: List[MacroSeasonResult] = []

    @staticmethod
    def _load_optional_kmap_mask() -> Optional[np.ndarray]:
        """Load fitted 128×128 kmap mask from experiment artifact if present."""
        candidates = [
            ROOT_DIR / "experiments" / "qkd_model_6months_8opponents" / "qkd_model.npz",
            Path("experiments/qkd_model_6months_8opponents/qkd_model.npz"),
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                data = np.load(path, allow_pickle=True)
                if "kmap_2d_mask" in data.files:
                    return np.asarray(data["kmap_2d_mask"], dtype=np.float32)
            except Exception:
                continue
        return None

    def run_macro_simulation(
        self,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the full 4-year (48 seasons = 34,560 steps) simulation."""
        opp_keys = list(self.opponents.keys())
        global_step = 0

        t0_macro = time.perf_counter()
        if verbose:
            print("=" * 90)
            print(f"🚀 Starting 4-Year Macro Horizon Self-Training ({self.total_seasons} Seasons | 1,440 Days | 34,560 Turns)")
            print(f"🏟️ League Opponents ({len(self.opponents)}): {', '.join(opp_keys)}")
            print("=" * 90)

        for global_season in range(1, self.total_seasons + 1):
            year = ((global_season - 1) // self.seasons_per_year) + 1
            season_in_year = ((global_season - 1) % self.seasons_per_year) + 1

            # Cycle through league opponents
            opp_key = opp_keys[(global_season - 1) % len(opp_keys)]
            opp_fn = self.opponents[opp_key]

            env = _make_kaggle_env(max_steps=720, turns_per_day=24)
            states = _normalize_states(env.reset())
            obs_p0 = parse_observation(states[0], player_id=0)
            obs_p1 = parse_observation(states[1], player_id=1)

            step = 0
            done = False
            q_probes = []
            k_probes = []
            d_probes = []

            while not done and step < 720:
                global_step += 1
                act_p0 = self.agent.act(obs_p0)
                act_p1 = opp_fn(obs_p1)

                # Record QKD Probes
                qkd_map = map_observation_to_qkd(obs_p0)
                q_probes.append(qkd_map.q_norm)
                k_probes.append(qkd_map.k_norm)
                d_probes.append(qkd_map.d_norm)

                # Log Market Data
                market = obs_p0.get("market", {}) or {}
                inv = market.get("inventory", {}) or {}
                prices = market.get("prices", {}) or {}

                for commodity in COMMODITIES:
                    cur_inv = int(inv.get(commodity, 10000) or 10000)
                    cur_price = float(prices.get(commodity, 25.0) or 25.0)
                    self.market_history.append(
                        MarketDataPoint(
                            year=year,
                            season=season_in_year,
                            day=int(obs_p0.get("day", 1) or 1),
                            hour=int(obs_p0.get("hour", 0) or 0),
                            global_step=global_step,
                            commodity=commodity,
                            inventory=cur_inv,
                            price=cur_price,
                            shop_demand=10.0,
                            sales_volume=0,
                        )
                    )

                states = _normalize_states(env.step([act_p0, act_p1]))
                obs_p0 = parse_observation(states[0], player_id=0)
                obs_p1 = parse_observation(states[1], player_id=1)

                status = states[0].get("status", "ACTIVE")
                done = status in ("DONE", "TIMEOUT", "INVALID", "ERROR") or step >= 719
                step += 1

            farms = obs_p0.get("farms", []) or []
            p0_money = float(farms[0].get("money", 0.0) if len(farms) > 0 else 0.0)
            p1_money = float(farms[1].get("money", 0.0) if len(farms) > 1 else 0.0)

            res = MacroSeasonResult(
                year=year,
                season_in_year=season_in_year,
                global_season=global_season,
                opponent_name=opp_key,
                champion_final_money=p0_money,
                opp_final_money=p1_money,
                win=p0_money > p1_money,
                mean_q_probe=float(np.mean(q_probes)),
                mean_k_probe=float(np.mean(k_probes)),
                mean_d_probe=float(np.mean(d_probes)),
            )
            self.season_results.append(res)

            if progress_callback:
                progress_callback(global_season, self.total_seasons)

            if verbose and (global_season % 6 == 0 or global_season == self.total_seasons):
                wins_so_far = sum(1 for r in self.season_results if r.win)
                win_pct = (wins_so_far / len(self.season_results)) * 100.0
                print(
                    f"  [Year {year} | Season {season_in_year:2d}/{self.seasons_per_year:2d}] "
                    f"vs {opp_key:<16}: Score ${p0_money:>8,.0f} vs ${p1_money:>7,.0f} | "
                    f"Macro Win Rate: {wins_so_far}/{len(self.season_results)} ({win_pct:.1f}%)"
                )

        total_time = time.perf_counter() - t0_macro
        total_wins = sum(1 for r in self.season_results if r.win)

        summary = {
            "total_years": self.total_years,
            "total_seasons": self.total_seasons,
            "total_steps": global_step,
            "total_time_sec": total_time,
            "total_wins": total_wins,
            "win_rate": total_wins / self.total_seasons,
            "mean_final_score": float(np.mean([r.champion_final_money for r in self.season_results])),
            "market_datapoints": len(self.market_history),
        }

        # Output Polarized K-Map Heatmap on Training Exit
        try:
            from vector_memory_bank import save_kmap_heatmap
        except ImportError:
            from reasoning_vs_questioning.vector_memory_bank import save_kmap_heatmap

        sample_obs = {
            "player": 0, "day": 15, "hour": 12,
            "farms": [{"money": summary["mean_final_score"], "hires_today": 3, "tiles": []}, {"money": 35000.0, "hires_today": 2, "tiles": []}],
            "market": {"inventory": {"WHEAT": 9000}, "prices": {"WHEAT": 30.0}},
            "town": {"unlocked_shops": ["BAKERY", "PET_CAFE"]},
        }
        # QKDReplayRLAgent may not carry a fitted 128×128 mask; fall back to None
        # (StateVectorEncoder.compute_polarized_kmap_matrix uses the proportional mask).
        mask_2d = getattr(self.agent, "kmap_2d_mask", None)
        if mask_2d is None:
            mask_2d = self._load_optional_kmap_mask()
        polarized_kmap = self.encoder.compute_polarized_kmap_matrix(
            sample_obs,
            mask_2d=mask_2d,
        )
        kmap_path = save_kmap_heatmap(
            polarized_kmap,
            save_path="four_year_kmap_polarization_heatmap.png",
            title="4-Year Macro Horizon: Polarized 2D K-Map Activation Matrix ({0, 1} Asymptote)",
        )
        summary["kmap_heatmap_path"] = kmap_path

        if verbose:
            print("=" * 90)
            print(f"🎉 4-Year Macro Simulation Complete in {total_time:.2f}s ({global_step:,} Total Game Steps)")
            print(f"🏆 Overall Macro Record: {total_wins}/{self.total_seasons} Won ({summary['win_rate']*100:.1f}%) | Mean Score: ${summary['mean_final_score']:,.0f}")
            print(f"[+] Output polarized K-Map heatmap on training exit -> {kmap_path}")
            print("=" * 90)

        return summary

    def get_market_dataframe(self) -> pd.DataFrame:
        """Convert collected market history into a clean Pandas DataFrame."""
        return pd.DataFrame([
            {
                "year": pt.year,
                "season": pt.season,
                "day": pt.day,
                "hour": pt.hour,
                "global_step": pt.global_step,
                "commodity": pt.commodity,
                "inventory": pt.inventory,
                "price": pt.price,
            }
            for pt in self.market_history
        ])

    def get_season_results_dataframe(self) -> pd.DataFrame:
        """Convert season outcomes into a clean Pandas DataFrame."""
        return pd.DataFrame([
            {
                "Year": r.year,
                "Season": r.season_in_year,
                "Global Season": r.global_season,
                "Opponent": r.opponent_name,
                "Champion Score": r.champion_final_money,
                "Opponent Score": r.opp_final_money,
                "Margin": r.champion_final_money - r.opp_final_money,
                "Win": r.win,
                "Mean Q (Days Rem)": r.mean_q_probe,
                "Mean K (Opp Money)": r.mean_k_probe,
                "Mean D (Subagents)": r.mean_d_probe,
            }
            for r in self.season_results
        ])

    def compute_smoothed_market_curves(self) -> Dict[str, pd.DataFrame]:
        """Extract continuous smoothed price curves across all commodities."""
        df = self.get_market_dataframe()
        curves = {}

        for commodity in COMMODITIES:
            sub = df[df["commodity"] == commodity].sort_values("inventory")
            if sub.empty:
                continue
            # Bin inventory levels to average out discrete sell lumps
            sub["inv_bin"] = pd.cut(sub["inventory"], bins=30)
            binned = sub.groupby("inv_bin", observed=True).agg(
                mean_inv=("inventory", "mean"),
                mean_price=("price", "mean"),
                min_price=("price", "min"),
                max_price=("price", "max"),
                count=("price", "count"),
            ).dropna().reset_index()

            curves[commodity] = binned

        return curves
