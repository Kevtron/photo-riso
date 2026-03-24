"""Load images as sRGB uint8; alpha is composited on white (documented)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_uint8(path: Path) -> np.ndarray:
    """
    Return HxWx3 uint8 sRGB. If the source has an alpha channel, composite
    over opaque white so CIELAB k-means sees premultiplied-correct sRGB.
    """
    im = Image.open(path)
    im = im.convert("RGBA")
    rgba = np.array(im, dtype=np.uint8)
    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    comp = np.clip(rgb * a + 255.0 * (1.0 - a), 0, 255).astype(np.uint8)
    return comp
