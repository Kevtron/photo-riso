#!/usr/bin/env python3
"""
Run photo-riso across k in {3,4,5} and all combinations of:
- mask-mode: binary, tonal-softmax, tonal-assigned
- dither: none, ordered, floyd (tonal modes only; binary uses none)
- ink palette data/riso-palette-six.json with map-mode nearest / assign / family

(No centroid-only / no-ink runs.)

Processes every image file in an input directory (default: examples/).
Each source file gets outputs under batch_outputs/<sanitized_stem>/ with INDEX.txt there.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "examples"
PALETTE = REPO_ROOT / "data" / "riso-palette-six.json"
BATCH_ROOT = REPO_ROOT / "batch_outputs"

IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
)


def safe_stem_dir(name: str) -> str:
    """Filesystem-safe directory name from input basename without extension."""
    s = re.sub(r'[<>:"/\\|?*]', "_", name)
    s = s.strip(" .")
    return s or "output"


def list_input_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return out


def build_jobs(input_path: Path, out_root: Path) -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    dithers = ("none", "ordered", "floyd")
    ink_modes = ("nearest", "assign", "family")

    for k in (3, 4, 5):
        base = [
            sys.executable,
            "-m",
            "photo_riso",
            "-i",
            str(input_path),
            "--k",
            str(k),
            "--seed",
            "42",
            "--sample-size",
            "120000",
        ]
        for im in ink_modes:
            jobs.append(
                (
                    f"k{k}_binary_dither-none_ink-{im}",
                    [
                        *base,
                        "--mask-mode",
                        "binary",
                        "--dither",
                        "none",
                        "--ink-palette",
                        str(PALETTE),
                        "--ink-map-mode",
                        im,
                        "-o",
                    ],
                )
            )
        for d in dithers:
            for im in ink_modes:
                jobs.append(
                    (
                        f"k{k}_tonal-softmax_dither-{d}_ink-{im}",
                        [
                            *base,
                            "--mask-mode",
                            "tonal-softmax",
                            "--dither",
                            d,
                            "--ink-palette",
                            str(PALETTE),
                            "--ink-map-mode",
                            im,
                            "-o",
                        ],
                    )
                )
                jobs.append(
                    (
                        f"k{k}_tonal-assigned_dither-{d}_ink-{im}",
                        [
                            *base,
                            "--mask-mode",
                            "tonal-assigned",
                            "--dither",
                            d,
                            "--ink-palette",
                            str(PALETTE),
                            "--ink-map-mode",
                            im,
                            "-o",
                        ],
                    )
                )

    resolved: list[tuple[str, list[str]]] = []
    for name, cmd in jobs:
        if cmd[-1] != "-o":
            raise RuntimeError(name)
        out = out_root / name
        resolved.append((name, [*cmd, str(out)]))
    return resolved


def quote_cmd(cmd: list[str]) -> str:
    parts: list[str] = []
    for c in cmd:
        if " " in c or '"' in c:
            parts.append('"' + c.replace('"', '\\"') + '"')
        else:
            parts.append(c)
    return " ".join(parts)


def run_batch_for_image(input_path: Path) -> int:
    stem_dir = safe_stem_dir(input_path.stem)
    out_root = BATCH_ROOT / stem_dir
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(input_path, out_root)
    index_lines: list[str] = []

    print(f"=== {input_path.name} -> {out_root.relative_to(REPO_ROOT)} ({len(jobs)} jobs) ===")

    for name, cmd in jobs:
        out_dir = Path(cmd[-1])
        out_dir.mkdir(parents=True, exist_ok=True)
        line = quote_cmd(cmd)
        (out_dir / "command.txt").write_text(line + "\n", encoding="utf-8")
        index_lines.append(f"{name}\t{line}")

        print("RUN:", name)
        r = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if r.returncode != 0:
            print(f"FAILED: {name} (exit {r.returncode})", file=sys.stderr)
            return r.returncode

    (out_root / "INDEX.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Done. {len(jobs)} runs under {out_root}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory of images to process (default: {DEFAULT_INPUT_DIR})",
    )
    args = p.parse_args(argv)

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not PALETTE.is_file():
        print(f"Missing palette: {PALETTE}", file=sys.stderr)
        return 1

    images = list_input_images(input_dir)
    if not images:
        print(
            f"No image files found in {input_dir} "
            f"(supported: {', '.join(sorted(IMAGE_SUFFIXES))})",
            file=sys.stderr,
        )
        return 1

    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    jobs_per_image = len(build_jobs(images[0], BATCH_ROOT / "_size_probe"))
    total_jobs = 0
    for path in images:
        rc = run_batch_for_image(path)
        if rc != 0:
            return rc
        total_jobs += jobs_per_image

    print(f"All images OK. {len(images)} file(s), {total_jobs} jobs total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
