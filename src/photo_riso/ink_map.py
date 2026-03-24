"""Map cluster centroids to user ink palettes in CIELAB."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from photo_riso.lab_colors import delta_e_lab, ink_entry_to_rgb, rgb_uint8_to_lab

InkMapMode = Literal["nearest", "assign", "family"]


@dataclass
class InkMapping:
    cluster_index: int
    centroid_rgb: tuple[int, int, int]
    centroid_lab: tuple[float, float, float]
    assigned_ink_name: str
    assigned_ink_rgb: tuple[int, int, int]
    delta_e: float
    one_to_one: bool


def load_palette(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("palette JSON must be a list of ink objects")
    return list(data)


def _inks_lab(palette: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    rgbs: list[tuple[int, int, int]] = []
    for i, entry in enumerate(palette):
        name = str(entry.get("name", f"ink_{i}"))
        rgb = ink_entry_to_rgb(entry)
        names.append(name)
        rgbs.append(rgb)
    arr = np.array(rgbs, dtype=np.uint8).reshape(-1, 1, 1, 3)
    lab = rgb_uint8_to_lab(arr).reshape(-1, 3)
    return names, lab.astype(np.float64)


def _centroid_rgb_lab(centroids_lab: np.ndarray) -> np.ndarray:
    """kx3 lab -> kx1x1x3 for rgb conversion path if needed — use lab_colors.lab_vector_to_rgb_uint8"""
    from photo_riso.lab_colors import lab_vector_to_rgb_uint8

    rgb = lab_vector_to_rgb_uint8(centroids_lab)
    return rgb


def map_inks(
    centroids_lab: np.ndarray,
    palette: list[dict[str, Any]],
    mode: InkMapMode,
    family_alpha: float = 0.5,
    max_combo_enum: int = 64,
) -> list[InkMapping]:
    k = centroids_lab.shape[0]
    names, inks_lab = _inks_lab(palette)
    n_inks = len(names)
    if n_inks == 0:
        raise ValueError("palette is empty")

    centroid_rgb = _centroid_rgb_lab(centroids_lab)

    cost = delta_e_lab(centroids_lab[:, np.newaxis, :], inks_lab[np.newaxis, :, :])

    if mode == "nearest":
        out: list[InkMapping] = []
        for i in range(k):
            jj = int(np.argmin(cost[i]))
            r, g, b = ink_entry_to_rgb(palette[jj])
            out.append(
                InkMapping(
                    cluster_index=i,
                    centroid_rgb=tuple(int(x) for x in centroid_rgb[i]),
                    centroid_lab=tuple(float(x) for x in centroids_lab[i]),
                    assigned_ink_name=names[jj],
                    assigned_ink_rgb=(r, g, b),
                    delta_e=float(cost[i, jj]),
                    one_to_one=False,
                )
            )
        return out

    if mode in ("assign", "family"):
        if n_inks < k:
            raise ValueError(f"palette has {n_inks} inks but k={k}; need at least k for {mode}")

    if mode == "assign":
        row_ind, col_ind = linear_sum_assignment(cost)
        ink_for_row = np.empty(k, dtype=np.int64)
        ink_for_row[row_ind] = col_ind
        out = []
        for i in range(k):
            jj = int(ink_for_row[i])
            r, g, b = ink_entry_to_rgb(palette[jj])
            out.append(
                InkMapping(
                    cluster_index=i,
                    centroid_rgb=tuple(int(x) for x in centroid_rgb[i]),
                    centroid_lab=tuple(float(x) for x in centroids_lab[i]),
                    assigned_ink_name=names[jj],
                    assigned_ink_rgb=(r, g, b),
                    delta_e=float(cost[i, jj]),
                    one_to_one=True,
                )
            )
        return out

    # family
    d_cent = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            d_cent[i, j] = d_cent[j, i] = float(delta_e_lab(centroids_lab[i], centroids_lab[j]))

    def pair_mismatch(perm: tuple[int, ...]) -> float:
        s = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                ii, jj = perm[i], perm[j]
                d_ink = float(delta_e_lab(inks_lab[ii], inks_lab[jj]))
                s += abs(d_cent[i, j] - d_ink)
        return s

    def total_delta(perm: tuple[int, ...]) -> float:
        return float(sum(cost[i, perm[i]] for i in range(k)))

    def family_cost(perm: tuple[int, ...]) -> float:
        a = family_alpha
        return a * total_delta(perm) + (1.0 - a) * pair_mismatch(perm)

    best_perm: tuple[int, ...] | None = None
    best_c = np.inf

    def consider_perm(perm: tuple[int, ...]) -> None:
        nonlocal best_perm, best_c
        c = family_cost(perm)
        if c < best_c:
            best_c = c
            best_perm = perm

    max_perm_evals = max(1, max_combo_enum) * max(1, max_combo_enum) * 100

    if n_inks == k:
        for perm in itertools.permutations(range(k)):
            consider_perm(perm)
    else:
        n_eval = 0
        for combo in itertools.combinations(range(n_inks), k):
            for perm in itertools.permutations(combo):
                consider_perm(perm)
                n_eval += 1
                if n_eval >= max_perm_evals:
                    break
            if n_eval >= max_perm_evals:
                break
        if best_perm is None:
            row_ind, col_ind = linear_sum_assignment(cost)
            chosen = tuple(int(c) for c in col_ind)
            for perm in itertools.permutations(chosen):
                consider_perm(perm)

    assert best_perm is not None
    out = []
    for i in range(k):
        jj = int(best_perm[i])
        r, g, b = ink_entry_to_rgb(palette[jj])
        out.append(
            InkMapping(
                cluster_index=i,
                centroid_rgb=tuple(int(x) for x in centroid_rgb[i]),
                centroid_lab=tuple(float(x) for x in centroids_lab[i]),
                assigned_ink_name=names[jj],
                assigned_ink_rgb=(r, g, b),
                delta_e=float(cost[i, jj]),
                one_to_one=True,
            )
        )
    return out


def mappings_to_json(mappings: list[InkMapping]) -> list[dict[str, Any]]:
    return [
        {
            "cluster_index": m.cluster_index,
            "centroid_rgb": list(m.centroid_rgb),
            "centroid_lab": list(m.centroid_lab),
            "assigned_ink_name": m.assigned_ink_name,
            "assigned_ink_rgb": list(m.assigned_ink_rgb),
            "delta_e": m.delta_e,
            "one_to_one": m.one_to_one,
        }
        for m in mappings
    ]
