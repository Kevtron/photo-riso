"""CLI for photo-riso."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from photo_riso.cluster import pixel_centroid_distances, segment_image
from photo_riso.image_io import load_rgb_uint8
from photo_riso.ink_map import load_palette, map_inks, mappings_to_json
from photo_riso.lab_colors import lab_vector_to_rgb_uint8, rgb_to_hex, rgb_uint8_to_lab
from photo_riso.masks import apply_dither, build_masks, save_mask_png
from photo_riso.preview import composite_masks_rgb


def _colors_payload(centroids_lab: np.ndarray) -> list[dict]:
    rgb = lab_vector_to_rgb_uint8(centroids_lab)
    out = []
    for i, row in enumerate(centroids_lab):
        r, g, b = int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])
        out.append(
            {
                "cluster_index": i,
                "rgb": [r, g, b],
                "hex": rgb_to_hex(r, g, b),
                "lab": [float(row[0]), float(row[1]), float(row[2])],
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(
        description="K-means color separation in CIELAB and risograph-style masks.",
    )
    p.add_argument("--input", "-i", type=Path, required=True, help="Input image path")
    p.add_argument("--output-dir", "-o", type=Path, required=True, help="Output directory")
    p.add_argument("--k", type=int, default=5, help="Number of colors / masks")
    p.add_argument("--sample-size", type=int, default=100_000, help="Pixels for k-means fit")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument(
        "--mask-mode",
        choices=("binary", "tonal-softmax", "tonal-assigned"),
        default="binary",
    )
    p.add_argument("--tonal-beta", type=float, default=1.0)
    p.add_argument("--tonal-sigma", type=float, default=15.0)
    p.add_argument("--tonal-gamma", type=float, default=1.0)
    p.add_argument("--dither", choices=("none", "ordered", "floyd"), default="none")
    p.add_argument("--ink-palette", type=Path, default=None)
    p.add_argument(
        "--ink-map-mode",
        choices=("nearest", "assign", "family"),
        default="nearest",
    )
    p.add_argument("--ink-family-alpha", type=float, default=0.5)
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip writing preview.png (masks tinted and summed in sRGB).",
    )
    args = p.parse_args(argv)

    rgb = load_rgb_uint8(args.input)
    seg = segment_image(
        rgb,
        k=args.k,
        sample_size=args.sample_size,
        random_state=args.seed,
    )
    lab = rgb_uint8_to_lab(seg.rgb)
    dists = pixel_centroid_distances(lab, seg.centroids_lab)

    masks = build_masks(
        seg.labels,
        dists,
        args.k,
        mode=args.mask_mode,  # type: ignore[arg-type]
        tonal_beta=args.tonal_beta,
        tonal_sigma=args.tonal_sigma,
        tonal_gamma=args.tonal_gamma,
    )
    if args.mask_mode != "binary" and args.dither != "none":
        masks = apply_dither(masks, args.dither)  # type: ignore[arg-type]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.mask_mode.replace("tonal-", "t-")
    for i, m in enumerate(masks):
        save_mask_png(
            str(args.output_dir / f"mask_{i + 1:02d}_{suffix}.png"),
            m,
        )

    colors = _colors_payload(seg.centroids_lab)
    (args.output_dir / "colors.json").write_text(
        json.dumps(colors, indent=2),
        encoding="utf-8",
    )

    layer_rgb = lab_vector_to_rgb_uint8(seg.centroids_lab)
    if args.ink_palette is not None:
        palette = load_palette(args.ink_palette)
        mappings = map_inks(
            seg.centroids_lab,
            palette,
            mode=args.ink_map_mode,  # type: ignore[arg-type]
            family_alpha=args.ink_family_alpha,
        )
        (args.output_dir / "mapping.json").write_text(
            json.dumps(mappings_to_json(mappings), indent=2),
            encoding="utf-8",
        )
        layer_rgb = np.array([m.assigned_ink_rgb for m in mappings], dtype=np.uint8)

    if not args.no_preview:
        preview = composite_masks_rgb(masks, layer_rgb)
        Image.fromarray(preview, mode="RGB").save(
            args.output_dir / "preview.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
