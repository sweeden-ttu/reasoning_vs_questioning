"""Environment wrappers for Kaggriculture self-play training.

Contains:
- KaggleCompetitiveEnv: Two-player wrapper around official kaggle-environments.
- create_competitive_env: Factory function.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from kaggriculture_adapter import parse_observation


def _normalize_env_states(result: Any) -> List[Dict[str, Any]]:
    """Normalize environment step/reset result to a list of dicts."""
    if isinstance(result, list):
        return [s if isinstance(s, dict) else {
            "observation": getattr(s, "observation", {}),
            "status": getattr(s, "status", "ACTIVE"),
            "reward": getattr(s, "reward", 0),
        } for s in result]
    return [result if isinstance(result, dict) else {
        "observation": getattr(result, "observation", {}),
        "status": getattr(result, "status", "ACTIVE"),
        "reward": getattr(result, "reward", 0),
    }]


class KaggleCompetitiveEnv:
    """Two-player wrapper around official kaggle-environments.

    Parameters
    ----------
    max_steps : int
        Maximum steps per episode (default 720 = competition standard).
    seed : int
        Random seed for the environment.
    turns_per_cycle : int
        Engine turnsPerDay (default 24 = competition parity; use 72 for
        kinematic self-play profile).
    """

    def __init__(
        self,
        max_steps: int = 720,
        seed: int = 42,
        turns_per_cycle: int = 24,
    ):
        import kaggle_environments
        self.max_steps = max_steps
        self.turns_per_cycle = int(turns_per_cycle)
        self.env = kaggle_environments.make(
            "kaggriculture",
            configuration={
                "episodeSteps": max_steps,
                "turnsPerDay": self.turns_per_cycle,
                "seed": seed,
            },
            debug=False,
        )
        self._obs: List[Dict[str, Any]] = [{}, {}]
        self._prev_money: List[float] = [0.0, 0.0]

    def transfer_bank(self, from_player: int, to_player: int, amount: float) -> bool:
        """Move ``amount`` coins from one farm bank to the other (experiment charity).

        Kaggriculture has no native donate op; this mutates the live engine farms
        and refreshes parsed observations so the transfer is publicly visible.
        """
        amount = float(amount)
        if amount <= 0 or from_player == to_player:
            return False
        if from_player not in (0, 1) or to_player not in (0, 1):
            return False

        # Prefer mutating the official engine state so later steps keep the transfer.
        engine_farms = None
        try:
            states = getattr(self.env, "state", None)
            if states and len(states) > 0:
                obs0 = states[0].observation if hasattr(states[0], "observation") else states[0].get("observation")
                if obs0 is not None:
                    engine_farms = obs0["farms"] if isinstance(obs0, dict) else getattr(obs0, "farms", None)
        except (TypeError, KeyError, AttributeError):
            engine_farms = None

        if engine_farms is not None and len(engine_farms) > max(from_player, to_player):
            src = float(engine_farms[from_player].get("money", 0.0) or 0.0)
            if src < amount:
                return False
            engine_farms[from_player]["money"] = src - amount
            engine_farms[to_player]["money"] = float(
                engine_farms[to_player].get("money", 0.0) or 0.0
            ) + amount
            # Mirror into every seated observation's farms list when present.
            for st in states:
                o = st.observation if hasattr(st, "observation") else st.get("observation", {})
                farms = o["farms"] if isinstance(o, dict) else getattr(o, "farms", None)
                if farms is None:
                    continue
                farms[from_player]["money"] = engine_farms[from_player]["money"]
                farms[to_player]["money"] = engine_farms[to_player]["money"]

        # Always update our cached parsed observations (public money on both seats).
        for pid in (0, 1):
            farms = self._obs[pid].get("farms", []) or []
            if len(farms) <= max(from_player, to_player):
                continue
            src = float(farms[from_player].get("money", 0.0) or 0.0)
            if src < amount and engine_farms is None:
                return False
            if engine_farms is None:
                farms[from_player]["money"] = src - amount
                farms[to_player]["money"] = float(farms[to_player].get("money", 0.0) or 0.0) + amount
            else:
                farms[from_player]["money"] = float(engine_farms[from_player]["money"])
                farms[to_player]["money"] = float(engine_farms[to_player]["money"])

        self._prev_money = [
            float((self._obs[0].get("farms") or [{}])[0].get("money", 0.0) or 0.0)
            if len(self._obs[0].get("farms") or []) > 0
            else 0.0,
            float((self._obs[1].get("farms") or [{}, {}])[1].get("money", 0.0) or 0.0)
            if len(self._obs[1].get("farms") or []) > 1
            else 0.0,
        ]
        return True

    def reset(self) -> Dict[str, Any]:
        """Reset environment and return player 0's observation."""
        states = _normalize_env_states(self.env.reset())
        self._obs = [
            parse_observation(states[0], player_id=0),
            parse_observation(states[1], player_id=1),
        ]
        farms_p0 = self._obs[0].get("farms", [])
        farms_p1 = self._obs[1].get("farms", [])
        self._prev_money = [
            float(farms_p0[0].get("money", 0.0)) if len(farms_p0) > 0 else 0.0,
            float(farms_p1[1].get("money", 0.0)) if len(farms_p1) > 1 else 0.0,
        ]
        return self._obs[0]

    def _get_obs(self, player: int) -> Dict[str, Any]:
        """Return parsed observation for a specific player."""
        return self._obs[player]

    def step(
        self, actions: List[Dict[str, Any]]
    ) -> Tuple[Tuple[Dict[str, Any], Dict[str, Any]], List[float], bool, Dict]:
        """Execute a pair of actions and return transitions.

        Returns
        -------
        ((obs_p0, obs_p1), rewards, done, info)
        """
        states = _normalize_env_states(self.env.step(actions))
        self._obs = [
            parse_observation(states[0], player_id=0),
            parse_observation(states[1], player_id=1),
        ]
        rewards: List[float] = []
        for p in range(2):
            farms = self._obs[p].get("farms", [])
            money = float(farms[p].get("money", 0.0)) if len(farms) > p else 0.0
            rewards.append((money - self._prev_money[p]) / 100.0)
            self._prev_money[p] = money
        status = states[0].get("status", "ACTIVE")
        done = status in ("DONE", "TIMEOUT", "INVALID")
        return (self._obs[0], self._obs[1]), rewards, done, {}


def create_competitive_env(
    use_kaggle: bool = True,
    max_steps: int = 720,
    seed: int = 42,
    turns_per_cycle: int = 24,
):
    """Factory for creating competitive environments.

    Parameters
    ----------
    use_kaggle : bool
        If False, raises RuntimeError (offline training requires the
        official Kaggle simulator).
    """
    if not use_kaggle:
        raise RuntimeError(
            "Offline training requires the official Kaggle simulator (use_kaggle_env=True). "
            "Install kaggle-environments and attach the kaggriculture environment."
        )
    try:
        return KaggleCompetitiveEnv(
            max_steps=max_steps,
            seed=seed,
            turns_per_cycle=turns_per_cycle,
        )
    except ImportError as exc:
        raise ImportError(
            "kaggle-environments is required for self-play training. "
            "Install with: pip install kaggle-environments"
        ) from exc
