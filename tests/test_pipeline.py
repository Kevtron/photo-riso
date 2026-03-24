"""Pipeline tests on synthetic images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from photo_riso.cluster import pixel_centroid_distances, segment_image
from photo_riso.ink_map import map_inks
from photo_riso.lab_colors import rgb_uint8_to_lab
from photo_riso.masks import build_masks


def _five_color_image(h: int = 50, w: int = 50) -> np.ndarray:
    """HxW horizontal stripes of 5 saturated colors."""
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
    ]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    band = h // 5
    for i, c in enumerate(colors):
        y0, y1 = i * band, (i + 1) * band if i < 4 else h
        img[y0:y1, :, :] = c
    return img


def test_segment_binary_masks_mutually_exclusive():
    rgb = _five_color_image()
    seg = segment_image(rgb, k=5, sample_size=5000, random_state=42)
    lab = rgb_uint8_to_lab(seg.rgb)
    dists = pixel_centroid_distances(lab, seg.centroids_lab)
    masks = build_masks(seg.labels, dists, 5, mode="binary")
    stack = np.stack([m.astype(np.int32) for m in masks], axis=-1)
    assert np.all(np.sum(stack == 255, axis=-1) == 1)
    assert np.sum(stack == 255) == rgb.shape[0] * rgb.shape[1]


def test_tonal_softmax_sums_to_255():
    rgb = _five_color_image(20, 20)
    seg = segment_image(rgb, k=5, sample_size=400, random_state=0)
    lab = rgb_uint8_to_lab(seg.rgb)
    dists = pixel_centroid_distances(lab, seg.centroids_lab)
    masks = build_masks(
        seg.labels, dists, 5, mode="tonal-softmax", tonal_beta=1.0, tonal_gamma=1.0
    )
    got = np.stack([m.astype(np.float64) for m in masks], axis=-1)
    s = got.sum(axis=-1)
    assert np.all(s >= 252.0) and np.all(s <= 255.0)


def test_tonal_assigned_nonzero_only_on_label():
    rgb = _five_color_image(30, 30)
    seg = segment_image(rgb, k=5, sample_size=800, random_state=1)
    lab = rgb_uint8_to_lab(seg.rgb)
    dists = pixel_centroid_distances(lab, seg.centroids_lab)
    masks = build_masks(
        seg.labels, dists, 5, mode="tonal-assigned", tonal_sigma=20.0
    )
    for i, m in enumerate(masks):
        assert np.all(m[seg.labels != i] == 0)
        assert np.any(m[seg.labels == i] > 0)


def test_cli_writes_outputs(tmp_path: Path):
    rgb = _five_color_image(32, 32)
    p = tmp_path / "in.png"
    Image.fromarray(rgb, mode="RGB").save(p)
    import subprocess
    import sys

    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "photo_riso",
            "--input",
            str(p),
            "--output-dir",
            str(tmp_path / "out"),
            "--k",
            "5",
            "--seed",
            "0",
            "--sample-size",
            "2000",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    out = tmp_path / "out"
    for i in range(5):
        assert (out / f"mask_{i + 1:02d}_binary.png").is_file()
    data = json.loads((out / "colors.json").read_text(encoding="utf-8"))
    assert len(data) == 5
    assert all("lab" in row for row in data)


def test_ink_map_nearest():
    centroids = np.array(
        [[50.0, 20.0, -30.0], [80.0, -10.0, 90.0]],
        dtype=np.float64,
    )
    palette = [
        {"name": "A", "hex": "#808080"},
        {"name": "B", "hex": "#ffff00"},
    ]
    m = map_inks(centroids, palette, mode="nearest")
    assert len(m) == 2
    assert all(x.delta_e >= 0 for x in m)


def test_ink_map_assign_one_to_one():
    centroids = np.array(
        [
            [30.0, 40.0, 20.0],
            [60.0, -20.0, -20.0],
        ],
        dtype=np.float64,
    )
    palette = [
        {"name": "ink0", "r": 100, "g": 50, "b": 50},
        {"name": "ink1", "r": 200, "g": 200, "b": 50},
    ]
    m = map_inks(centroids, palette, mode="assign")
    assert len(m) == 2
    assert m[0].one_to_one and m[1].one_to_one
    inks = {m[0].assigned_ink_name, m[1].assigned_ink_name}
    assert inks == {"ink0", "ink1"}
