"""sRGB (D65) to CIELAB and back via skimage (linear RGB internally)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from skimage.color import lab2rgb, rgb2lab


def srgb_uint8_to_linear_float(rgb: np.ndarray) -> np.ndarray:
    """rgb uint8 HxWx3 -> linear float HxWx3 in [0, 1]."""
    c = rgb.astype(np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_float_to_srgb_uint8(rgb: np.ndarray) -> np.ndarray:
    """Linear float HxWx3 in [0,1] -> sRGB uint8."""
    c = np.clip(rgb, 0.0, 1.0)
    srgb = np.where(c <= 0.0031308, 12.92 * c, 1.055 * (c ** (1.0 / 2.4)) - 0.055)
    return np.round(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def rgb_uint8_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 HxWx3 -> CIELAB (D65) HxWx3 (L*, a*, b* skimage convention)."""
    linear = srgb_uint8_to_linear_float(rgb)
    return rgb2lab(linear, channel_axis=-1)


def lab_to_rgb_uint8(lab: np.ndarray) -> np.ndarray:
    """CIELAB HxWx3 -> sRGB uint8."""
    linear = lab2rgb(lab, channel_axis=-1)
    return linear_float_to_srgb_uint8(linear)


def lab_vector_to_rgb_uint8(lab: np.ndarray) -> np.ndarray:
    """lab Nx3 -> RGB uint8 Nx3."""
    lab = np.asarray(lab, dtype=np.float64).reshape(-1, 1, 1, 3)
    rgb = lab_to_rgb_uint8(lab)
    return rgb.reshape(-1, 3)


def delta_e_lab(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean ΔE in CIELAB. a, b shape (..., 3)."""
    return np.linalg.norm(a - b, axis=-1)


_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


def parse_hex(s: str) -> tuple[int, int, int]:
    m = _HEX.match(s.strip())
    if not m:
        raise ValueError(f"invalid hex color: {s!r}")
    h = m.group(1)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def ink_entry_to_rgb(entry: dict[str, Any]) -> tuple[int, int, int]:
    if "hex" in entry:
        return parse_hex(str(entry["hex"]))
    if all(k in entry for k in ("r", "g", "b")):
        return int(entry["r"]), int(entry["g"]), int(entry["b"])
    raise ValueError(f"ink entry needs hex or r,g,b: {entry!r}")


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"
