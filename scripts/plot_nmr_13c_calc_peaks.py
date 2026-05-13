#!/usr/bin/env python3
"""
Stick plot: grouped calculated 13C δ (from nmr_13c_calc) vs literature δ.

Runs nmr_13c_calc with an optional third argument to emit JSON, or reads
--calc-json from a prior run.

Main molecule JSON may include:
  "experimental_13c_shift_ppm": [ ... ]
one value per carbon in the same order as nmr_13c_calc (increasing atom index,
same as "carbons" in the emitted JSON).

Use the same nmr_13c_calc binary that works in **this** shell (e.g. a Docker
build under ./build/ vs a different host build under ./build-local/ are not
interchangeable if shared libraries differ).

By default the horizontal axis **zooms to the data** (plus padding) so nearby
peaks (e.g. CH₂ vs CH₃ a few tenths of a ppm apart) stay visible. Use
--full-ppm-scale for a fixed 0–220 ppm window (then --ppm-min / --ppm-max apply).

Examples:

  # A — Let the script run the calculator (binary must load here):
  python3 scripts/plot_nmr_13c_calc_peaks.py \\
    sample_input/propane.json sample_input/methane.json \\
    --nmr-13c-calc ./build/nmr_13c_calc \\
    --out student_output/nmr_13c_propane_peaks.png

  # B — You already ran: nmr_13c_calc main ref out.json  (skip binary from Python):
  python3 scripts/plot_nmr_13c_calc_peaks.py \\
    sample_input/propane.json sample_input/methane.json \\
    --calc-json student_output/propane_13c_calc.json \\
    --out student_output/nmr_13c_propane_peaks.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from matplotlib_defaults import (
    LEGEND_FONT_SIZE,
    SMALL_ANNOTATION_FONT_SIZE,
    apply_large_plot_text,
)


def plot_stick_overlay_grouped(
    ax,
    pred: list[float],
    exp_pts: list[float],
    *,
    stick_height: float,
    linewidth: float,
    text_sep_ppm: float,
) -> None:
    n_peaks = max(len(pred), 1)
    h_scale = stick_height / float(n_peaks)
    for mp, me in zip(pred, exp_pts):
        h = h_scale
        ax.plot(
            [mp, mp],
            [0.0, h],
            color="black",
            linewidth=linewidth,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            [me, me],
            [0.0, h],
            color="red",
            linewidth=linewidth,
            alpha=0.9,
            solid_capstyle="butt",
            zorder=3,
        )
        yb = h + 0.035
        yr = h + 0.035
        if abs(mp - me) < text_sep_ppm:
            yr = yb + 0.055
        ax.text(
            mp,
            yb,
            f"{mp:.1f}",
            ha="center",
            va="bottom",
            fontsize=SMALL_ANNOTATION_FONT_SIZE,
            color="black",
            zorder=5,
        )
        ax.text(
            me,
            yr,
            f"{me:.1f}",
            ha="center",
            va="bottom",
            fontsize=SMALL_ANNOTATION_FONT_SIZE,
            color="#c00",
            zorder=5,
        )
    ax.plot([], [], color="black", linewidth=linewidth, label="Calculated (grouped)")
    ax.plot(
        [],
        [],
        color="red",
        linewidth=linewidth,
        alpha=0.9,
        label="Reference (exp.)",
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("main_json", type=Path, help="Main molecule config (same as nmr_13c_calc).")
    ap.add_argument("ref_json", type=Path, help="Reference molecule config (e.g. methane).")
    ap.add_argument(
        "--nmr-13c-calc",
        type=Path,
        default=Path("build/nmr_13c_calc"),
        help="Path to nmr_13c_calc executable (default ./build/nmr_13c_calc).",
    )
    ap.add_argument(
        "--calc-json",
        type=Path,
        default=None,
        help="Use existing JSON from a prior: nmr_13c_calc main ref this.json",
    )
    ap.add_argument(
        "--stick-lw",
        type=float,
        default=2.0,
        help="Stick linewidth.",
    )
    ap.add_argument(
        "--text-sep-ppm",
        type=float,
        default=2.0,
        help="Stagger labels when |calc−exp| < this (ppm).",
    )
    ap.add_argument(
        "--full-ppm-scale",
        action="store_true",
        help="Use fixed axis from --ppm-min to --ppm-max (default 0–220). "
        "Omit this flag to auto-zoom to calculated + experimental peaks.",
    )
    ap.add_argument(
        "--ppm-min",
        type=float,
        default=0.0,
        help="With --full-ppm-scale: low edge of δ axis (default 0).",
    )
    ap.add_argument(
        "--ppm-max",
        type=float,
        default=220.0,
        help="With --full-ppm-scale: high edge (default 220).",
    )
    ap.add_argument(
        "--ppm-pad",
        type=float,
        default=4.0,
        help="When auto-zooming: padding (ppm) beyond min/max δ (default 4).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("student_output/nmr_13c_calc_peaks.png"),
        help="Output PNG path.",
    )
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    args = ap.parse_args()

    if args.full_ppm_scale and args.ppm_min >= args.ppm_max:
        print("--ppm-min must be less than --ppm-max.", file=sys.stderr)
        return 1

    main_cfg = json.loads(args.main_json.read_text())
    exp_key = "experimental_13c_shift_ppm"
    if exp_key not in main_cfg:
        print(
            f"Main JSON must include '{exp_key}' (array, one δ per carbon in "
            "calculation order).",
            file=sys.stderr,
        )
        return 1
    exp_arr = main_cfg[exp_key]
    if not isinstance(exp_arr, list):
        print(f"{exp_key} must be a JSON array.", file=sys.stderr)
        return 1
    exp_pts_all = [float(x) for x in exp_arr]

    tmp_path: Path | None = None
    if args.calc_json is not None:
        calc_path = args.calc_json
        if not calc_path.is_file():
            print(f"Not a file: {calc_path}", file=sys.stderr)
            return 1
    else:
        exe = args.nmr_13c_calc
        if not exe.is_file() or not os_access_ok(exe):
            print(
                f"Missing or non-executable: {exe}\n"
                "Build nmr_13c_calc or pass --calc-json / --nmr-13c-calc.",
                file=sys.stderr,
            )
            return 1
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            subprocess.run(
                [
                    str(exe),
                    str(args.main_json.resolve()),
                    str(args.ref_json.resolve()),
                    str(tmp_path.resolve()),
                ],
                check=True,
            )
            calc_path = tmp_path
        except subprocess.CalledProcessError as e:
            print(f"nmr_13c_calc failed: {e}", file=sys.stderr)
            if e.returncode == 127:
                print(
                    "Exit 127 often means wrong binary for this environment "
                    "(e.g. host build-local inside Docker) or missing shared "
                    "libraries. Use --nmr-13c-calc pointing to the same build "
                    "that works when you run nmr_13c_calc by hand, or pass "
                    "--calc-json path/to.json if you already wrote JSON with "
                    "a successful run.",
                    file=sys.stderr,
                )
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            return 1

    data = json.loads(calc_path.read_text())
    if tmp_path is not None:
        tmp_path.unlink(missing_ok=True)

    carbons = data.get("carbons") or []
    n_c = len(carbons)
    if n_c != len(exp_pts_all):
        print(
            f"Length mismatch: {exp_key} has {len(exp_pts_all)} values but "
            f"calculation has {n_c} carbons.",
            file=sys.stderr,
        )
        return 1

    groups = data.get("groups") or []
    if not groups:
        print("Calc JSON has no 'groups' entries.", file=sys.stderr)
        return 1

    pred_peaks: list[float] = []
    exp_peaks: list[float] = []
    for g in groups:
        pred_peaks.append(float(g["delta_avg_ppm"]))
        slots = [int(i) for i in g["carbon_slot_indices"]]
        exp_peaks.append(sum(exp_pts_all[i] for i in slots) / float(len(slots)))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib:  pip install matplotlib", file=sys.stderr)
        return 1

    apply_large_plot_text(plt)

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 2.8))
    plot_stick_overlay_grouped(
        ax,
        pred_peaks,
        exp_peaks,
        stick_height=1.0,
        linewidth=args.stick_lw,
        text_sep_ppm=args.text_sep_ppm,
    )
    title = f"13C NMR — {data.get('molecule_label', args.main_json.stem)}"
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=LEGEND_FONT_SIZE)
    if args.full_ppm_scale:
        ax.set_xlim(args.ppm_max, args.ppm_min)
    else:
        pts = pred_peaks + exp_peaks + exp_pts_all
        span = max(pts) - min(pts)
        pad = max(args.ppm_pad, 0.08 * span if span > 0 else args.ppm_pad)
        x_lo = min(pts) - pad
        x_hi = max(pts) + pad
        ax.set_xlim(x_hi, x_lo)
    ax.set_ylim(0.0, 1.22)
    ax.set_xlabel("δ (ppm, literature scale in JSON)")
    ax.set_ylabel("Relative intensity")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out.resolve()}")
    return 0


def os_access_ok(path: Path) -> bool:
    import os

    return os.path.isfile(path) and os.access(path, os.X_OK)


if __name__ == "__main__":
    raise SystemExit(main())
