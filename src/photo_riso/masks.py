"""Build separation masks (binary / tonal) and optional dither."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

MaskMode = Literal["binary", "tonal-softmax", "tonal-assigned"]
DitherMode = Literal["none", "ordered", "floyd"]


def _softmax_neg_beta(dists: np.ndarray, beta: float) -> np.ndarray:
    """dists HxWxk -> weights HxWxk summing to 1."""
    x = -beta * dists
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(np.clip(x, -50, 50))
    return ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-12)


def _d_own(lab_dists: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Distance from each pixel to its assigned centroid."""
    h, w = labels.shape
    yi = np.arange(h, dtype=np.intp)[:, None]
    xi = np.arange(w, dtype=np.intp)[None, :]
    return lab_dists[yi, xi, labels.astype(np.intp)]


def build_masks(
    labels: np.ndarray,
    lab_dists: np.ndarray,
    k: int,
    mode: MaskMode,
    tonal_beta: float = 1.0,
    tonal_sigma: float = 15.0,
    tonal_gamma: float = 1.0,
) -> list[np.ndarray]:
    """
    Return k uint8 HxW masks.
    lab_dists: HxWxk Euclidean distances in CIELAB.
    """
    h, w = labels.shape
    if mode == "binary":
        return [
            np.where(labels == i, 255, 0).astype(np.uint8) for i in range(k)
        ]

    if mode == "tonal-softmax":
        wts = _softmax_neg_beta(lab_dists, tonal_beta)
        out = []
        for i in range(k):
            plane = (wts[..., i] * 255.0).astype(np.float64)
            if tonal_gamma != 1.0:
                plane = np.clip((plane / 255.0) ** tonal_gamma * 255.0, 0, 255)
            out.append(plane.astype(np.uint8))
        return out

    if mode == "tonal-assigned":
        d = _d_own(lab_dists, labels)
        sig = max(tonal_sigma, 1e-6)
        base = np.exp(-(d**2) / (2.0 * sig**2))
        out = []
        for i in range(k):
            m = labels == i
            plane = np.zeros((h, w), dtype=np.float64)
            plane[m] = base[m] * 255.0
            if tonal_gamma != 1.0:
                nz = plane > 0
                plane[nz] = np.clip((plane[nz] / 255.0) ** tonal_gamma * 255.0, 0, 255)
            out.append(plane.astype(np.uint8))
        return out

    raise ValueError(f"unknown mask mode: {mode}")


_BAYER_4 = (
    np.array(
        [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
        dtype=np.float64,
    )
    / 16.0
)


def _dither_ordered(gray: np.ndarray) -> np.ndarray:
    """8-bit gray -> 1-bit expanded to 0/255 uint8, tiled Bayer 4x4."""
    h, w = gray.shape
    th, tw = (h + 3) // 4 * 4, (w + 3) // 4 * 4
    g = np.zeros((th, tw), dtype=np.float64)
    g[:h, :w] = gray.astype(np.float64)
    by, bx = np.meshgrid(np.arange(th), np.arange(tw), indexing="ij")
    thr = _BAYER_4[by % 4, bx % 4] * 255.0
    out = (g >= thr).astype(np.uint8) * 255
    return out[:h, :w]


def _dither_floyd(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float64).copy()
    h, w = g.shape
    for y in range(h):
        for x in range(w):
            old = g[y, x]
            new = 255.0 if old >= 128.0 else 0.0
            err = old - new
            g[y, x] = new
            if x + 1 < w:
                g[y, x + 1] += err * (7.0 / 16.0)
            if y + 1 < h:
                if x > 0:
                    g[y + 1, x - 1] += err * (3.0 / 16.0)
                g[y + 1, x] += err * (5.0 / 16.0)
                if x + 1 < w:
                    g[y + 1, x + 1] += err * (1.0 / 16.0)
    return (g > 127).astype(np.uint8) * 255


def apply_dither(masks: list[np.ndarray], dither: DitherMode) -> list[np.ndarray]:
    if dither == "none":
        return masks
    if dither == "ordered":
        return [_dither_ordered(m) for m in masks]
    if dither == "floyd":
        return [_dither_floyd(m) for m in masks]
    raise ValueError(f"unknown dither: {dither}")


def save_mask_png(path: str, mask: np.ndarray) -> None:
    Image.fromarray(mask, mode="L").save(path)
