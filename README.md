# photo-riso

Separate a photograph into **k** dominant colors using **k-means in CIELAB**, then export **grayscale separation masks** (one PNG per color) plus a JSON summary of centroid colors. A **`preview.png`** stacks those masks as an **additive RGB composite** tinted by each layer’s color (cluster sRGB by default, or **mapped ink RGB** when `--ink-palette` is set). Optional **ink palette mapping** suggests which named spot inks best match each cluster for risograph-style workflows.

Clustering is perceptual (LAB space, D65, sRGB with proper gamma). Transparent pixels are **composited on white** before analysis. Mask dimensions match the source image for registration across plates.

## Requirements

- Python 3.10+

## Install

```bash
pip install -e .
```

For tests:

```bash
pip install -e ".[dev]"
```

## Usage

Run as a module or via the console script (after install):

```bash
python -m photo_riso --input path/to/image.png --output-dir path/to/out
photo-riso -i path/to/image.png -o path/to/out
```

### Outputs

| File | Description |
|------|-------------|
| `mask_01_<mode>.png` … `mask_k_<mode>.png` | Single-channel masks; cluster `01` is the **most prevalent** color after sorting |
| `preview.png` | RGB image: each mask is scaled by its layer color and **summed** (screen-style preview; not physical ink overprint). Omitted if `--no-preview` |
| `colors.json` | Per cluster: `rgb`, `hex`, `lab` (CIELAB centroid) |
| `mapping.json` | Only if `--ink-palette` is set: suggested ink per cluster, ΔE, flags |

### Common options

| Option | Default | Description |
|--------|---------|-------------|
| `-i`, `--input` | (required) | Source image path |
| `-o`, `--output-dir` | (required) | Directory for masks and JSON (created if missing) |
| `--k` | `5` | Number of colors / masks |
| `--seed` | `0` | Random seed for subsampling and k-means |
| `--sample-size` | `100000` | Pixels used to fit k-means (full image is still labeled) |
| `--mask-mode` | `binary` | `binary`, `tonal-softmax`, or `tonal-assigned` |
| `--tonal-beta` | `1.0` | Softmax sharpness (tonal-softmax) |
| `--tonal-sigma` | `15.0` | Gaussian width in LAB units (tonal-assigned) |
| `--tonal-gamma` | `1.0` | Gamma on tonal values before save |
| `--dither` | `none` | `ordered` (Bayer) or `floyd` (Floyd–Steinberg); **only applied for tonal modes** |
| `--ink-palette` | — | JSON file listing inks (see below) |
| `--ink-map-mode` | `nearest` | `nearest`, `assign` (one-to-one Hungarian), or `family` (relationship-aware) |
| `--ink-family-alpha` | `0.5` | Blend ΔE vs pairwise geometry for `family` |
| `--no-preview` | off | Do not write `preview.png` |

Run `python -m photo_riso --help` for the full list.

### Mask modes (short)

- **binary** — 255 on pixels assigned to that cluster, 0 elsewhere (non-overlapping plates).
- **tonal-softmax** — Per-pixel weights from softmax of distances to all centroids (can overlap tone across plates).
- **tonal-assigned** — Each pixel contributes only on its winning cluster; strength falls off with distance to that centroid (Gaussian by `--tonal-sigma`).

### Ink palette JSON

A JSON **array** of objects. Each object needs a `name` and either `hex` (`"#rrggbb"`) or `r`, `g`, `b` (0–255).

Example using the bundled six-color subset (from [mattdesl/riso-colors](https://github.com/mattdesl/riso-colors)):

```bash
python -m photo_riso -i photo.png -o out --k 6 ^
  --ink-palette data/riso-palette-six.json --ink-map-mode assign
```

On Unix shells, use `\` for line continuation instead of `^`.

For `--ink-map-mode assign` or `family`, the palette must contain **at least `--k` inks**. Use `nearest` if you allow the same ink for multiple clusters.

### Batch sweep (k = 3, 4, 5)

`scripts/batch_all_mask_combos.py` scans an **input directory** (default: `examples/`) for images (PNG, JPEG, WebP, TIFF, BMP, GIF). For **each** file it runs **63** jobs: every mix of `--mask-mode`, `--dither`, and `data/riso-palette-six.json` with each `--ink-map-mode` (`nearest` / `assign` / `family`), for `k` 3–5. (No centroid-only / no-ink runs.) Outputs go to **`batch_outputs/<sanitized_filename_stem>/<run_id>/`** (gitignored). Each run folder has **`command.txt`**; per source image, **`batch_outputs/<stem>/INDEX.txt`** lists all commands. Override the folder with **`--input-dir path/to/folder`**.

### Development

```bash
python -m pytest tests/ -v
```
