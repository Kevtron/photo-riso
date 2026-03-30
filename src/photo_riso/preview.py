"""Composite masks tinted by layer RGB (additive screen preview)."""

from __future__ import annotations

import numpy as np


def composite_masks_rgb(
    masks: list[np.ndarray],
    layer_rgb: np.ndarray,
) -> np.ndarray:
    """
    Blend separation masks with per-layer sRGB.

    Each mask contributes (mask/255) * color channel-wise; results are clipped
    to [0, 255]. This is a screen-style additive preview, not physical ink
    overprint simulation.

    masks: k entries, each HxW uint8
    layer_rgb: kx3 uint8
    """
    if not masks:
        raise ValueError("masks must be non-empty")
    h, w = masks[0].shape
    layer_rgb = np.asarray(layer_rgb, dtype=np.float64)
    if layer_rgb.shape != (len(masks), 3):
        raise ValueError("layer_rgb must be kx3")

    acc = np.zeros((h, w, 3), dtype=np.float64)
    for i, m in enumerate(masks):
        if m.shape != (h, w):
            raise ValueError("all masks must share shape")
        wgt = m.astype(np.float64) / 255.0
        acc += wgt[..., np.newaxis] * layer_rgb[i]
    return np.clip(np.round(acc), 0, 255).astype(np.uint8)
