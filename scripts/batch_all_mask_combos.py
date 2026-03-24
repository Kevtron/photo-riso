#!/usr/bin/env python3
"""
Run photo-riso across k in {3,4,5} and all combinations of:
- mask-mode: binary, tonal-softmax, tonal-assigned
- dither: none, ordered, floyd (meaningful for tonal only; still recorded)
- ink: none, or palette with map-mode nearest / assign / family

Writes command.txt into each output directory.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / "examples" / "input.png"
PALETTE = REPO_ROOT / "data" / "riso-palette-six.json"
OUT_ROOT = REPO_ROOT / "batch_outputs"


def build_jobs() -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    dithers = ("none", "ordered", "floyd")
    ink_modes = ("nearest", "assign", "family")

    for k in (3, 4, 5):
        # --- No ink palette ---
        base = [
            sys.executable,
            "-m",
            "photo_riso",
            "-i",
            str(INPUT),
            "--k",
            str(k),
            "--seed",
            "42",
            "--sample-size",
            "120000",
        ]
        jobs.append(
            (
                f"k{k}_binary_dither-none_no-ink",
                [*base, "--mask-mode", "binary", "--dither", "none", "-o"],
            )
        )
        for d in dithers:
            jobs.append(
                (
                    f"k{k}_tonal-softmax_dither-{d}_no-ink",
                    [
                        *base,
                        "--mask-mode",
                        "tonal-softmax",
                        "--dither",
                        d,
                        "-o",
                    ],
                )
            )
            jobs.append(
                (
                    f"k{k}_tonal-assigned_dither-{d}_no-ink",
                    [
                        *base,
                        "--mask-mode",
                        "tonal-assigned",
                        "--dither",
                        d,
                        "-o",
                    ],
                )
            )

        # --- With ink palette ---
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

    # Append output dir placeholder — filled in run()
    resolved: list[tuple[str, list[str]]] = []
    for name, cmd in jobs:
        if cmd[-1] != "-o":
            raise RuntimeError(name)
        out = OUT_ROOT / name
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


def main() -> int:
    if not INPUT.is_file():
        print(f"Missing input image: {INPUT}", file=sys.stderr)
        return 1
    if not PALETTE.is_file():
        print(f"Missing palette: {PALETTE}", file=sys.stderr)
        return 1

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    index_lines: list[str] = []

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

    (OUT_ROOT / "INDEX.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Done. {len(jobs)} runs under {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
