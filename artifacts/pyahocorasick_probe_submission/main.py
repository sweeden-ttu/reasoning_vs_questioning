"""Kaggle Kaggriculture agent entry — probes pyahocorasick then PASSes.

On first observation, runs the full public-API probe and writes
``pyahocorasick_probe_report.json`` under ``/kaggle/working`` (or cwd).
Always returns a legal PASS action so the episode can complete.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pyahocorasick_probe import probe_environment  # noqa: E402

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


class Agent:
    def __init__(self) -> None:
        self._probed = False
        self._report: Dict[str, Any] = {}

    def _run_probe(self) -> None:
        if self._probed:
            return
        self._probed = True
        self._report = probe_environment()
        out_dir = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else HERE
        path = out_dir / "pyahocorasick_probe_report.json"
        path.write_text(json.dumps(self._report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Also echo a one-line verdict for competition logs.
        print(
            "[pyahocorasick_probe]",
            json.dumps(
                {
                    "verdict": self._report.get("verdict"),
                    "package_present": self._report.get("package_present"),
                    "passed": self._report.get("passed"),
                    "failed": self._report.get("failed"),
                    "machine": (self._report.get("environment") or {}).get("machine"),
                    "python": (self._report.get("environment") or {}).get("python_version_info"),
                    "ahocorasick_file": (self._report.get("environment") or {}).get("ahocorasick_file"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def __call__(self, obs: Dict[str, Any], config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self._run_probe()
        # Keep hands length legal if observation exposes hired hands.
        farms = obs.get("farms") or []
        me = {}
        try:
            me = farms[int(obs.get("player", 0) or 0)] or {}
        except Exception:  # noqa: BLE001
            me = {}
        n_hands = len(me.get("hands") or [])
        hands: List[List[str]] = [["PASS"] for _ in range(n_hands)]
        return {"farmer": ["PASS"], "hands": hands, "market": []}


# Kaggle environments often look for a top-level callable named `agent`.
_AGENT = Agent()


def agent(obs, config=None):  # noqa: ANN001
    return _AGENT(obs, config)


def main() -> None:
    # Offline / notebook smoke without the full env.
    _AGENT._run_probe()


if __name__ == "__main__":
    main()
