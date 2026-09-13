"""RoL / LoR dual-agent QKD attention matrices.

Questioning agent operates on **DKR** (Decisions × K-map × Reasoning-facing):
  apply **RoL** (Rotate-Left = ``np.rot90(..., k=1)``).

Reasoning agent operates on **DKQ** (Decisions × K-map × Questions):
  apply **LoR** (Left-over-Right = ``np.rot90(..., k=-1)``, inverse orientation).

Decision-tree caps (locked defaults):
  - Land expansion ≤ 40×40
  - Labor hire ≤ 4 days × 4 subagents
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from kmap_boundary_substitutes import (
    DUAL_LIMIT_OPTIONS,
    LABOR_MAX_DAYS,
    LABOR_MAX_HIRE_SLOTS,
    LABOR_MAX_SUBAGENTS,
    LAND_EXPANSION_MAX_CELLS,
    LAND_EXPANSION_MAX_COLS,
    LAND_EXPANSION_MAX_ROWS,
)

try:
    from kmap_2d_mask import build_2d_kmap_mask
except ImportError:  # pragma: no cover
    from reasoning_vs_questioning.kmap_2d_mask import build_2d_kmap_mask  # type: ignore


# Canonical scalar operands (32-bit half-rotate).
ROL_OPERAND_HEX = 0x5009FF3  # Questioning seat — Rotate-Left of this word
LOR_OPERAND_HEX = 0x1005FF9  # Reasoning seat — Left-over-Right of this word
SCALAR_OPERAND_WIDTH = 32


def RoL(matrix: np.ndarray) -> np.ndarray:
    """Rotate-Left attention transform (CCW 90°)."""
    return np.ascontiguousarray(np.rot90(np.asarray(matrix, dtype=np.float32), k=1))


def LoR(matrix: np.ndarray) -> np.ndarray:
    """Left-over-Right attention transform (CW 90°; inverse orientation of RoL)."""
    return np.ascontiguousarray(np.rot90(np.asarray(matrix, dtype=np.float32), k=-1))


def _even_width_mask(width: int) -> Tuple[int, int]:
    if width <= 0 or width % 2 != 0:
        raise ValueError(f"scalar rotate width must be positive even, got {width}")
    return (1 << width) - 1, width // 2


def RoL_u(value: int, width: int = SCALAR_OPERAND_WIDTH) -> int:
    """Scalar Rotate-Left: ≡ rotate-left by width//2 (inverse of LoR_u).

    Example: RoL_u(0x5009FF3, 32) → 0x9FF30500
    """
    mask, half = _even_width_mask(width)
    v = int(value) & mask
    return ((v << half) | (v >> half)) & mask


def LoR_u(value: int, width: int = SCALAR_OPERAND_WIDTH) -> int:
    """Scalar Left-over-Right: swap halves (≡ rotate-right by width//2).

    Example: LoR_u(0x1005FF9, 32) → 0x5FF90100
    """
    mask, half = _even_width_mask(width)
    v = int(value) & mask
    return ((v >> half) | (v << half)) & mask


def rol_operand_pair(
    value: int = ROL_OPERAND_HEX, width: int = SCALAR_OPERAND_WIDTH
) -> Dict[str, Any]:
    """Return in/out hex for the locked Questioning RoL scalar operand."""
    out = RoL_u(value, width)
    return {
        "rol_operand_in": int(value),
        "rol_operand_in_hex": hex(int(value) & ((1 << width) - 1)),
        "rol_operand_out": out,
        "rol_operand_out_hex": hex(out),
        "rol_operand_width": width,
        "rol_scalar_def": "half-rotate-left ≡ RoL(width//2)",
    }


def lor_operand_pair(
    value: int = LOR_OPERAND_HEX, width: int = SCALAR_OPERAND_WIDTH
) -> Dict[str, Any]:
    """Return in/out hex for the locked Reasoning LoR scalar operand."""
    out = LoR_u(value, width)
    return {
        "lor_operand_in": int(value),
        "lor_operand_in_hex": hex(int(value) & ((1 << width) - 1)),
        "lor_operand_out": out,
        "lor_operand_out_hex": hex(out),
        "lor_operand_width": width,
        "lor_scalar_def": "half-swap ≡ RoR(width//2)",
    }


def modulate_with_scalar_operand(matrix: np.ndarray, word: int) -> np.ndarray:
    """Phase-modulate an attention matrix with bits of a scalar rotate result."""
    m = np.asarray(matrix, dtype=np.float32)
    h, w = m.shape
    bits = [(int(word) >> i) & 1 for i in range(32)]
    row_phase = np.array(
        [1.0 if bits[i % 32] else -1.0 for i in range(h)], dtype=np.float32
    )
    col_phase = np.array(
        [1.0 if bits[(i * 7) % 32] else -1.0 for i in range(w)], dtype=np.float32
    )
    out = m * np.outer(row_phase, col_phase)
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = out / peak
    return out.astype(np.float32)


def modulate_with_lor_operand(matrix: np.ndarray, lor_word: int) -> np.ndarray:
    """Backward-compatible alias for Reasoning LoR modulation."""
    return modulate_with_scalar_operand(matrix, lor_word)


def modulate_with_rol_operand(matrix: np.ndarray, rol_word: int) -> np.ndarray:
    """Phase-modulate RoL(DKR) with bits of the scalar RoL result."""
    return modulate_with_scalar_operand(matrix, rol_word)


def _dequant_mean(vectors: np.ndarray, scales: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    s = np.asarray(scales, dtype=np.float32).reshape(-1, 1)
    return np.mean(v * s, axis=0)


def _unit(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < 1e-8:
        return x.astype(np.float32)
    return (x / n).astype(np.float32)


def build_channel_prototypes(npz_path: str | Path) -> Dict[str, np.ndarray]:
    """Load Q/K/D INT8 stores and return L2-normalized 256-D prototypes."""
    data = np.load(npz_path, allow_pickle=True)
    q = _unit(_dequant_mean(data["q_vectors"], data["q_scales"]))
    k = _unit(_dequant_mean(data["k_vectors"], data["k_scales"]))
    d = _unit(_dequant_mean(data["d_vectors"], data["d_scales"]))
    mask = np.asarray(data["kmap_2d_mask"], dtype=np.float32)
    return {"q": q, "k": k, "d": d, "kmap_2d_mask": mask}


def _paint_prototype_on_mask(
    proto: np.ndarray,
    mask: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Redistribute prototype energy onto mask-active rows (axis=0) or cols (axis=1)."""
    dim = mask.shape[0] if axis == 0 else mask.shape[1]
    weights = np.zeros(dim, dtype=np.float32)
    active = np.where(mask.sum(axis=1 - axis) > 0)[0]
    if active.size == 0:
        return np.ones(dim, dtype=np.float32)
    energy = np.sort(np.abs(np.asarray(proto[:dim], dtype=np.float32)))[::-1]
    take = min(active.size, energy.size)
    weights[active[:take]] = energy[:take]
    if float(weights[active].sum()) < 1e-12:
        weights[active] = 1.0
    return weights


def build_dkr_matrix(
    d: np.ndarray,
    k: np.ndarray,
    mask: np.ndarray,
    *,
    dim: int = 128,
) -> np.ndarray:
    """DKR = painted(D ⊗ K) ⊙ M — Decisions × K-map, Reasoning-facing mask."""
    m = np.asarray(mask, dtype=np.float32)
    if m.shape != (dim, dim):
        m = build_2d_kmap_mask(dim, dim)
    row_w = _paint_prototype_on_mask(d, m, axis=0)
    col_w = _paint_prototype_on_mask(k, m, axis=1)
    outer = np.outer(row_w, col_w).astype(np.float32)
    dkr = outer * m
    peak = float(np.max(dkr))
    if peak > 0:
        dkr /= peak
    return dkr


def build_dkq_matrix(
    d: np.ndarray,
    k: np.ndarray,
    q: np.ndarray,
    mask: np.ndarray,
    *,
    dim: int = 128,
) -> np.ndarray:
    """DKQ = painted(D ⊗ mix(K,Q)) ⊙ M — Decisions × K-map × Questions."""
    m = np.asarray(mask, dtype=np.float32)
    if m.shape != (dim, dim):
        m = build_2d_kmap_mask(dim, dim)
    mix = 0.5 * np.asarray(k[:dim], dtype=np.float32) + 0.5 * np.asarray(q[:dim], dtype=np.float32)
    row_w = _paint_prototype_on_mask(d, m, axis=0)
    col_w = _paint_prototype_on_mask(mix, m, axis=1)
    outer = np.outer(row_w, col_w).astype(np.float32)
    dkq = outer * m
    peak = float(np.max(dkq))
    if peak > 0:
        dkq /= peak
    return dkq


@dataclass
class DualRoLLoRAttentionModel:
    """Joint attention payload for Questioning (RoL∘DKR) and Reasoning (LoR∘DKQ)."""

    questioning_dkr: np.ndarray
    questioning_dkr_rol: np.ndarray
    reasoning_dkq: np.ndarray
    reasoning_dkq_lor: np.ndarray
    decision_tree_caps: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build_from_qkd_npz(
        cls,
        questioning_npz: str | Path,
        reasoning_npz: str | Path,
    ) -> "DualRoLLoRAttentionModel":
        q_proto = build_channel_prototypes(questioning_npz)
        r_proto = build_channel_prototypes(reasoning_npz)

        # Questioning: RoL(DKR), then modulate with RoL_u(0x5009FF3).
        dkr = build_dkr_matrix(q_proto["d"], q_proto["k"], q_proto["kmap_2d_mask"])
        rol_bits = rol_operand_pair()
        dkr_rol = modulate_with_rol_operand(RoL(dkr), int(rol_bits["rol_operand_out"]))

        # Reasoning: LoR(DKQ), then modulate with LoR_u(0x1005FF9).
        dkq = build_dkq_matrix(
            r_proto["d"], r_proto["k"], r_proto["q"], r_proto["kmap_2d_mask"]
        )
        lor_bits = lor_operand_pair()
        dkq_lor = modulate_with_lor_operand(LoR(dkq), int(lor_bits["lor_operand_out"]))

        caps = {
            "land_expansion_max": [LAND_EXPANSION_MAX_ROWS, LAND_EXPANSION_MAX_COLS],
            "land_max_cells": LAND_EXPANSION_MAX_CELLS,
            "labor_max_days": LABOR_MAX_DAYS,
            "labor_max_subagents": LABOR_MAX_SUBAGENTS,
            "labor_max_hire_slots": LABOR_MAX_HIRE_SLOTS,
            "dual_limit_options": list(DUAL_LIMIT_OPTIONS),
            "locked_default": True,
        }
        meta = {
            "questioning_transform": "RoL(DKR) ⊙ phase(RoL_u(0x5009FF3))",
            "reasoning_transform": "LoR(DKQ) ⊙ phase(LoR_u(0x1005FF9))",
            "rol_def": "np.rot90(M, k=1)  # CCW / rotate-left",
            "lor_def": "np.rot90(M, k=-1) # CW / left-over-right",
            "dkr_shape": list(dkr.shape),
            "dkq_shape": list(dkq.shape),
            "questioning_npz": str(questioning_npz),
            "reasoning_npz": str(reasoning_npz),
            **rol_bits,
            **lor_bits,
        }
        return cls(
            questioning_dkr=dkr,
            questioning_dkr_rol=dkr_rol,
            reasoning_dkq=dkq,
            reasoning_dkq_lor=dkq_lor,
            decision_tree_caps=caps,
            metadata=meta,
        )

    def attention_for(self, role: str) -> np.ndarray:
        role_n = str(role).strip().lower()
        if role_n in {"questioning", "q", "agent2", "scott"}:
            return self.questioning_dkr_rol
        if role_n in {"reasoning", "r", "agent1", "jonathan"}:
            return self.reasoning_dkq_lor
        raise ValueError(f"unknown role: {role!r}")

    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            questioning_dkr=self.questioning_dkr,
            questioning_dkr_rol=self.questioning_dkr_rol,
            reasoning_dkq=self.reasoning_dkq,
            reasoning_dkq_lor=self.reasoning_dkq_lor,
            decision_tree_caps_json=np.asarray(
                json.dumps(self.decision_tree_caps), dtype=object
            ),
            metadata_json=np.asarray(json.dumps(self.metadata), dtype=object),
        )
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> "DualRoLLoRAttentionModel":
        data = np.load(path, allow_pickle=True)
        caps_raw = data["decision_tree_caps_json"]
        meta_raw = data["metadata_json"]
        caps = json.loads(str(caps_raw.item() if hasattr(caps_raw, "item") else caps_raw))
        meta = json.loads(str(meta_raw.item() if hasattr(meta_raw, "item") else meta_raw))
        return cls(
            questioning_dkr=np.asarray(data["questioning_dkr"], dtype=np.float32),
            questioning_dkr_rol=np.asarray(data["questioning_dkr_rol"], dtype=np.float32),
            reasoning_dkq=np.asarray(data["reasoning_dkq"], dtype=np.float32),
            reasoning_dkq_lor=np.asarray(data["reasoning_dkq_lor"], dtype=np.float32),
            decision_tree_caps=caps,
            metadata=meta,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "questioning_dkr_rol_shape": list(self.questioning_dkr_rol.shape),
            "reasoning_dkq_lor_shape": list(self.reasoning_dkq_lor.shape),
            "questioning_rol_sum": float(np.sum(self.questioning_dkr_rol)),
            "reasoning_lor_sum": float(np.sum(self.reasoning_dkq_lor)),
            "rol_neq_base": not np.allclose(self.questioning_dkr_rol, self.questioning_dkr),
            "lor_neq_base": not np.allclose(self.reasoning_dkq_lor, self.reasoning_dkq),
            "decision_tree_caps": self.decision_tree_caps,
            "metadata": self.metadata,
        }


def render_dual_attention_heatmap(
    model: DualRoLLoRAttentionModel,
    save_path: str | Path = "dual_rol_lor_attention.png",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
    for ax, mat, title in [
        (axes[0], model.questioning_dkr_rol, "Questioning RoL(DKR)"),
        (axes[1], model.reasoning_dkq_lor, "Reasoning LoR(DKQ)"),
    ]:
        im = ax.imshow(mat, cmap="magma", aspect="auto")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Dual-agent attention · land≤40×40 · labor≤4×4", fontsize=12)
    fig.tight_layout()
    out = Path(save_path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out.resolve())


def build_and_save_dual_model(
    questioning_npz: str | Path,
    reasoning_npz: str | Path,
    out_npz: str | Path,
    heatmap_png: Optional[str | Path] = None,
) -> Dict[str, Any]:
    model = DualRoLLoRAttentionModel.build_from_qkd_npz(
        questioning_npz, reasoning_npz
    )
    path = model.save_npz(out_npz)
    result = {"npz": str(path.resolve()), **model.summary()}
    if heatmap_png is not None:
        result["heatmap"] = render_dual_attention_heatmap(model, heatmap_png)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    q_npz = root / "kagg_defense/for_scott_questioning_clone/models/qkd_model.npz"
    if not q_npz.exists():
        q_npz = Path("/Users/sweeden/kagg_defense/for_scott_questioning_clone/models/qkd_model.npz")
    r_npz = Path("/Users/sweeden/kagg_defense/for_jonathan_reasoning_clone/models/qkd_model.npz")
    out = root / "models" / "qkd_rol_lor_dual_agent.npz"
    if not out.parent.exists():
        out = Path("/Users/sweeden/kagg/models/qkd_rol_lor_dual_agent.npz")
    info = build_and_save_dual_model(
        q_npz,
        r_npz,
        out,
        heatmap_png=Path("/Users/sweeden/kagg/dual_rol_lor_attention.png"),
    )
    print(json.dumps(info, indent=2, default=str))
