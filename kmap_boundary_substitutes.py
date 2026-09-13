"""K-map asymptotic boundary substitutes (∞ ↔ imaginary).

When one K-map axis approaches real infinity and the other approaches the
imaginary axis, concrete farm actions replace the asymptotic sentinels:

  - Infinity side  (None)          → ``expanded land farming`` (trimmed ≤ 40×40)
  - Imaginary side ("" / \"empty\") → ``labor`` (trimmed ≤ 4 days × 4 subagents)

Either assignment of which axis is ∞ vs imaginary still yields the same
two-option substitute set; only which side owns which option flips.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

# Asymptotic limits (shared with Aho-Corasick path-trust boundary language).
LIMIT_APPROACHES_INFINITY: float = float("inf")
LIMIT_APPROACHES_IMAGINARY: complex = complex(0.0, float("inf"))

# Concrete Kaggriculture substitutes under the dual-limit regime.
SUBSTITUTE_INFINITY: str = "expanded land farming"
SUBSTITUTE_IMAGINARY: str = "labor"
DUAL_LIMIT_OPTIONS: Tuple[str, str] = (SUBSTITUTE_IMAGINARY, SUBSTITUTE_INFINITY)

# Trimmed decision-tree caps (replace unbounded ∞ / imaginary growth).
LAND_EXPANSION_MAX_ROWS: int = 40
LAND_EXPANSION_MAX_COLS: int = 40
LAND_EXPANSION_MAX_CELLS: int = LAND_EXPANSION_MAX_ROWS * LAND_EXPANSION_MAX_COLS  # 1600
LABOR_MAX_DAYS: int = 4
LABOR_MAX_SUBAGENTS: int = 4
LABOR_MAX_HIRE_SLOTS: int = LABOR_MAX_DAYS * LABOR_MAX_SUBAGENTS  # 16


def classify_kmap_axis_boundary(
    value: Any,
) -> Optional[Tuple[str, Union[float, complex], str]]:
    """Classify an axis sentinel as ∞ or imaginary, else ``None`` (ordinary)."""
    if value is None:
        return ("none", LIMIT_APPROACHES_INFINITY, "approaches_real_infinity")
    if isinstance(value, complex):
        if abs(value.real) < 1e-15 and value.imag == float("inf"):
            return ("empty", LIMIT_APPROACHES_IMAGINARY, "approaches_imaginary_infinity")
    if isinstance(value, float) and value == float("inf"):
        return ("none", LIMIT_APPROACHES_INFINITY, "approaches_real_infinity")
    if isinstance(value, str):
        if value == "" or value.lower() == "empty":
            return ("empty", LIMIT_APPROACHES_IMAGINARY, "approaches_imaginary_infinity")
    return None


def substitute_for_boundary(kind: str) -> str:
    """Map a single boundary kind to its farm-action substitute."""
    if kind == "none":
        return SUBSTITUTE_INFINITY
    if kind == "empty":
        return SUBSTITUTE_IMAGINARY
    raise ValueError(f"unknown boundary kind: {kind!r}")


def clamp_land_expansion_cell(row: int, col: int) -> Tuple[int, int]:
    """Clamp a land-expansion target into the trimmed 40×40 decision grid."""
    r = max(0, min(LAND_EXPANSION_MAX_ROWS - 1, int(row)))
    c = max(0, min(LAND_EXPANSION_MAX_COLS - 1, int(col)))
    return r, c


def land_expansion_allowed(unlocked_cells: int) -> bool:
    """True while land-expansion decision tree has remaining 40×40 capacity."""
    return int(unlocked_cells) < LAND_EXPANSION_MAX_CELLS


def labor_hire_allowed(
    *,
    day_in_window: int,
    subagents_active: int,
    hires_in_window: int,
) -> bool:
    """True while labor tree is inside 4 days × 4 subagents (16 hire slots)."""
    if int(day_in_window) < 0 or int(day_in_window) >= LABOR_MAX_DAYS:
        return False
    if int(subagents_active) >= LABOR_MAX_SUBAGENTS:
        return False
    return int(hires_in_window) < LABOR_MAX_HIRE_SLOTS


def trim_decision_tree_menu(
    *,
    unlocked_land_cells: int = 0,
    day_in_labor_window: int = 0,
    subagents_active: int = 0,
    hires_in_window: int = 0,
) -> Dict[str, Any]:
    """Trim dual-limit options to land≤40×40 and labor≤4×4 capacity."""
    land_ok = land_expansion_allowed(unlocked_land_cells)
    labor_ok = labor_hire_allowed(
        day_in_window=day_in_labor_window,
        subagents_active=subagents_active,
        hires_in_window=hires_in_window,
    )
    options: List[str] = []
    if labor_ok:
        options.append(SUBSTITUTE_IMAGINARY)
    if land_ok:
        options.append(SUBSTITUTE_INFINITY)
    return {
        "options": options,
        "land": {
            "max_shape": [LAND_EXPANSION_MAX_ROWS, LAND_EXPANSION_MAX_COLS],
            "max_cells": LAND_EXPANSION_MAX_CELLS,
            "unlocked_cells": int(unlocked_land_cells),
            "remaining_cells": max(0, LAND_EXPANSION_MAX_CELLS - int(unlocked_land_cells)),
            "allowed": land_ok,
            "substitute": SUBSTITUTE_INFINITY,
        },
        "labor": {
            "max_days": LABOR_MAX_DAYS,
            "max_subagents": LABOR_MAX_SUBAGENTS,
            "max_hire_slots": LABOR_MAX_HIRE_SLOTS,
            "day_in_window": int(day_in_labor_window),
            "subagents_active": int(subagents_active),
            "hires_in_window": int(hires_in_window),
            "remaining_hires": max(0, LABOR_MAX_HIRE_SLOTS - int(hires_in_window)),
            "allowed": labor_ok,
            "substitute": SUBSTITUTE_IMAGINARY,
        },
        "decision_tree_trimmed": True,
    }


def resolve_dual_limit_substitutes(
    self_axis: Any,
    opp_axis: Any,
    *,
    unlocked_land_cells: int = 0,
    day_in_labor_window: int = 0,
    subagents_active: int = 0,
    hires_in_window: int = 0,
) -> Optional[Dict[str, Any]]:
    """If one axis → ∞ and the other → imaginary, return trimmed labor / land options.

    Returns ``None`` unless the dual-limit condition holds.
    """
    self_b = classify_kmap_axis_boundary(self_axis)
    opp_b = classify_kmap_axis_boundary(opp_axis)
    if self_b is None or opp_b is None:
        return None
    self_kind, self_limit, self_label = self_b
    opp_kind, opp_limit, opp_label = opp_b
    # Require complementary asymptotes (∞ × imaginary), not ∞×∞ or empty×empty.
    if self_kind == opp_kind:
        return None
    self_opt = substitute_for_boundary(self_kind)
    opp_opt = substitute_for_boundary(opp_kind)
    trimmed = trim_decision_tree_menu(
        unlocked_land_cells=unlocked_land_cells,
        day_in_labor_window=day_in_labor_window,
        subagents_active=subagents_active,
        hires_in_window=hires_in_window,
    )
    ordered = tuple(opt for opt in (self_opt, opp_opt) if opt in trimmed["options"])
    return {
        "dual_limit": True,
        "self_axis": {
            "kind": self_kind,
            "limit": self_limit,
            "label": self_label,
            "substitute": self_opt,
        },
        "opp_axis": {
            "kind": opp_kind,
            "limit": opp_limit,
            "label": opp_label,
            "substitute": opp_opt,
        },
        "options": list(trimmed["options"]),
        "ordered_by_axis": ordered,
        "decision_menu": {
            SUBSTITUTE_IMAGINARY: "hire_or_assign_labor",
            SUBSTITUTE_INFINITY: "unlock_or_expand_land",
        },
        "decision_tree_caps": trimmed,
    }


def apply_dual_limit_to_obs(
    obs: Optional[Dict[str, Any]],
    *,
    self_key: str = "self_axis",
    opp_key: str = "opp_axis",
) -> Dict[str, Any]:
    """Annotate ``obs`` with dual-limit substitutes when both sentinels appear."""
    out: Dict[str, Any] = dict(obs or {})
    unlocked = int(out.get("unlocked_land_cells", out.get("land_cells", 0)) or 0)
    day = int(out.get("day", 0) or 0)
    day_in_window = day % LABOR_MAX_DAYS
    hands = out.get("hands") or []
    if not isinstance(hands, list):
        farms = out.get("farms") or []
        player = int(out.get("player", 0) or 0)
        if isinstance(farms, list) and player < len(farms) and isinstance(farms[player], dict):
            hands = farms[player].get("hands") or []
            unlocked = unlocked or int(farms[player].get("unlocked_land_cells", 0) or 0)
    subagents = len(hands) if isinstance(hands, list) else int(out.get("subagents_active", 0) or 0)
    hires = int(out.get("hires_in_window", out.get("hires_today", 0)) or 0)

    resolved = resolve_dual_limit_substitutes(
        out.get(self_key),
        out.get(opp_key),
        unlocked_land_cells=unlocked,
        day_in_labor_window=day_in_window,
        subagents_active=subagents,
        hires_in_window=hires,
    )
    if resolved is None:
        # Also accept None/'empty' on money/hands style probes as axis proxies.
        resolved = resolve_dual_limit_substitutes(
            out.get("my_money", out.get("money")),
            out.get("opp_money"),
            unlocked_land_cells=unlocked,
            day_in_labor_window=day_in_window,
            subagents_active=subagents,
            hires_in_window=hires,
        )
    if resolved is not None:
        out["kmap_dual_limit"] = resolved
        out["kmap_substitute_options"] = list(resolved["options"])
        out["preferred_actions"] = list(resolved["ordered_by_axis"])
        out["decision_tree_caps"] = resolved.get("decision_tree_caps")
    else:
        # Always expose trimmed caps even outside dual-limit, for policy gates.
        out["decision_tree_caps"] = trim_decision_tree_menu(
            unlocked_land_cells=unlocked,
            day_in_labor_window=day_in_window,
            subagents_active=subagents,
            hires_in_window=hires,
        )
    return out
