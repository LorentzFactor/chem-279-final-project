#!/usr/bin/env python3
"""
Overlay predicted vs training-reference ¹H chemical shifts as stick spectra.

Reads:
  - Training list JSON (same schema as nmr_1h_training_export): per-molecule
    experimental_shift_ppm (reference / literature δ used for fitting).
  - Feature CSV from nmr_1h_training_export: sigma_para_ppm per hydrogen row
    (same row order as the export).

Predicted δ uses the empirical linear map (same as nmr_1h_calc when not in
relative mode):  delta_pred = beta0 + beta1 * sigma_para_ppm

Stick mode (default): hydrogens are clustered in sorted predicted-delta order
when neighbor gaps are <= --group-tol ppm (same idea as nmr_1h_calc grouped
summary). One black stick per cluster at mean(calculated δ) and one red stick
at mean(training δ) for those same hydrogens. Stick height is proportional to
the number of hydrogens in the cluster (normalized so the largest multiplet
reaches full scale).

Use --per-proton-sticks for one line per hydrogen (no averaging).

The horizontal axis defaults to 0–10 ppm (typical ¹H window); override with
--ppm-min / --ppm-max.

By default one subplot is produced for every molecule in the training JSON
that has experimental_shift_ppm and matching CSV rows. Pass --molecule LABEL
to plot only that molecule (single spectrum).

Optional --lorentzian restores the previous broadened (pseudo-)spectrum view.

Requires: matplotlib  (pip install matplotlib)

Example — one molecule:
  python3 scripts/plot_nmr_training_spectrum.py \\
    sample_input/nmr_1h_training_export.json \\
    student_output/nmr_1h_training_features.csv \\
    --calibration-json sample_input/methane.json \\
    --molecule n-butane \\
    --out student_output/nmr_n_butane.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def lorentzian(x: float, x0: float, fwhm: float) -> float:
    """Unit-height Lorentzian; FWHM in same units as x (ppm)."""
    g = 0.5 * fwhm
    if g <= 0.0:
        return 1.0 if abs(x - x0) < 1e-12 else 0.0
    t = (x - x0) / g
    return 1.0 / (1.0 + t * t)


def spectrum_sum(x_grid: list[float], centers: list[float], fwhm: float) -> list[float]:
    out = [0.0] * len(x_grid)
    if not centers:
        return out
    for xc in centers:
        for i, x in enumerate(x_grid):
            out[i] += lorentzian(x, xc, fwhm)
    m = max(out) or 1.0
    return [v / m for v in out]


def cluster_indices_by_delta(values: list[float], tol_ppm: float) -> list[list[int]]:
    """1D single-linkage clusters on sorted values (indices into values)."""
    n = len(values)
    if n == 0:
        return []
    if tol_ppm <= 0.0:
        return [[i] for i in range(n)]
    order = sorted(range(n), key=lambda i: values[i])
    groups: list[list[int]] = []
    for k in range(n):
        i = order[k]
        if k == 0:
            groups.append([i])
            continue
        j = order[k - 1]
        if values[i] - values[j] <= tol_ppm:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def cluster_mean(vals: list[float], cluster: list[int]) -> float:
    return sum(vals[i] for i in cluster) / float(len(cluster))


def plot_stick_overlay_grouped(
    ax,
    pred: list[float],
    exp_pts: list[float],
    *,
    group_tol_ppm: float,
    stick_height: float,
    linewidth: float,
) -> None:
    """One black / red stick per cluster; height scales with cluster size (n_H)."""
    clusters = cluster_indices_by_delta(pred, group_tol_ppm)
    if not clusters:
        return
    max_n = max(len(cl) for cl in clusters) or 1
    for cl in clusters:
        n_h = len(cl)
        h = stick_height * (float(n_h) / float(max_n))
        mp = cluster_mean(pred, cl)
        me = cluster_mean(exp_pts, cl)
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
        if abs(mp - me) < 0.12:
            yr = yb + 0.055
        ax.text(
            mp,
            yb,
            f"{mp:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            zorder=5,
        )
        ax.text(
            me,
            yr,
            f"{me:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#c00",
            zorder=5,
        )
    ax.plot(
        [],
        [],
        color="black",
        linewidth=linewidth,
        label="Calculated",
    )
    ax.plot(
        [],
        [],
        color="red",
        linewidth=linewidth,
        alpha=0.9,
        label="Actual",
    )


def plot_stick_overlay(
    ax,
    pred: list[float],
    exp_pts: list[float],
    *,
    stick_height: float,
    linewidth: float,
) -> None:
    """Vertical sticks at each δ (one line per hydrogen)."""
    for p in pred:
        ax.plot(
            [p, p],
            [0.0, stick_height],
            color="black",
            linewidth=linewidth,
            solid_capstyle="butt",
            zorder=2,
        )
    for e in exp_pts:
        ax.plot(
            [e, e],
            [0.0, stick_height],
            color="red",
            linewidth=linewidth,
            alpha=0.9,
            solid_capstyle="butt",
            zorder=3,
        )
    for p, e in zip(pred, exp_pts):
        yp = stick_height + 0.03
        yr = yp + (0.052 if abs(p - e) < 0.08 else 0.0)
        ax.text(
            p,
            yp,
            f"{p:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
            color="black",
            zorder=5,
        )
        ax.text(
            e,
            yr,
            f"{e:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
            color="#c00",
            zorder=5,
        )
    # Legend proxies (avoid one entry per stick)
    ax.plot([], [], color="black", linewidth=linewidth, label="Calculated")
    ax.plot(
        [],
        [],
        color="red",
        linewidth=linewidth,
        alpha=0.9,
        label="Actual",
    )


def load_csv_by_molecule(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    by_mol: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            lab = (row.get("molecule") or "").strip()
            if lab:
                by_mol[lab].append(row)
    for lab in by_mol:
        by_mol[lab].sort(
            key=lambda r: (
                int(r["h_rank"]) if (r.get("h_rank") or "").strip().isdigit() else 0
            )
        )
    return dict(by_mol)


def load_calibration(args: argparse.Namespace) -> tuple[float, float]:
    if args.beta0 is not None and args.beta1 is not None:
        return float(args.beta0), float(args.beta1)
    if args.calibration_json is None:
        raise SystemExit(
            "Provide --calibration-json (with shift_calibration_beta0/1) "
            "or both --beta0 and --beta1."
        )
    cfg = json.loads(Path(args.calibration_json).read_text())
    try:
        return float(cfg["shift_calibration_beta0"]), float(
            cfg["shift_calibration_beta1"]
        )
    except KeyError as e:
        raise SystemExit(f"Missing key in calibration JSON: {e}") from e


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "training_json",
        type=Path,
        help="Training list JSON (experimental_shift_ppm per molecule).",
    )
    ap.add_argument(
        "features_csv",
        type=Path,
        help="CSV written by nmr_1h_training_export.",
    )
    ap.add_argument(
        "--calibration-json",
        type=Path,
        default=None,
        help="Molecule JSON containing shift_calibration_beta0/beta1.",
    )
    ap.add_argument("--beta0", type=float, default=None, help="Override intercept.")
    ap.add_argument("--beta1", type=float, default=None, help="Override slope.")
    ap.add_argument(
        "--lorentzian",
        action="store_true",
        help="Draw broadened Lorentzian sum instead of stick lines.",
    )
    ap.add_argument(
        "--fwhm",
        type=float,
        default=0.05,
        help="Lorentzian FWHM in ppm (only with --lorentzian; default 0.05).",
    )
    ap.add_argument(
        "--stick-lw",
        type=float,
        default=2.0,
        help="Stick linewidth (default 2.0).",
    )
    ap.add_argument(
        "--group-tol",
        type=float,
        default=0.12,
        help=(
            "Stick mode: merge neighbors in sorted predicted δ when gap <= this "
            "(ppm); default 0.12 (matches nmr_1h_calc default). "
            "Use <=0 for one stick per hydrogen."
        ),
    )
    ap.add_argument(
        "--per-proton-sticks",
        action="store_true",
        help="Stick mode: draw one stick per hydrogen (ignore --group-tol).",
    )
    ap.add_argument(
        "--ppm-min",
        type=float,
        default=0.0,
        help="Low-ppm edge of δ axis (default 0).",
    )
    ap.add_argument(
        "--ppm-max",
        type=float,
        default=10.0,
        help="High-ppm edge of δ axis (default 10).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("student_output/nmr_training_spectrum.png"),
        help="Output image path (default student_output/nmr_training_spectrum.png).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI for --out.",
    )
    ap.add_argument(
        "--molecule",
        metavar="LABEL",
        default=None,
        help=(
            "Plot only this molecule's spectrum (training JSON label, e.g. "
            "n-butane, propane). Omit to plot every plottable training molecule."
        ),
    )
    args = ap.parse_args()
    if args.ppm_min >= args.ppm_max:
        print("--ppm-min must be less than --ppm-max.", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib:  pip install matplotlib", file=sys.stderr)
        return 1

    beta0, beta1 = load_calibration(args)
    root = json.loads(args.training_json.read_text())
    csv_by_label = load_csv_by_molecule(args.features_csv)

    exp_by_label: dict[str, list[float]] = {}
    order: list[str] = []
    for mol in root.get("molecules", []):
        if "label" not in mol:
            continue
        label = str(mol["label"])
        order.append(label)
        if "experimental_shift_ppm" not in mol:
            continue
        arr = mol["experimental_shift_ppm"]
        if isinstance(arr, list):
            exp_by_label[label] = [float(v) for v in arr]

    labels_ok: list[str] = []
    for label in order:
        if label not in exp_by_label or label not in csv_by_label:
            continue
        rows = csv_by_label[label]
        exp_list = exp_by_label[label]
        if len(rows) != len(exp_list):
            print(
                f"Warning: {label}: CSV rows ({len(rows)}) != "
                f"experimental_shift_ppm length ({len(exp_list)}); skip.",
                file=sys.stderr,
            )
            continue
        for i, (row, y_json) in enumerate(zip(rows, exp_list)):
            de = (row.get("delta_exp_ppm") or "").strip()
            if de:
                y_csv = float(de)
                if abs(y_csv - y_json) > 1e-4:
                    print(
                        f"Warning: {label} h_rank={row.get('h_rank')}: "
                        f"CSV delta_exp_ppm={y_csv} vs JSON {y_json}",
                        file=sys.stderr,
                    )
        labels_ok.append(label)

    if not labels_ok:
        print(
            "No molecules to plot: need experimental_shift_ppm in JSON, "
            "matching CSV rows, and same hydrogen count.",
            file=sys.stderr,
        )
        return 1

    if args.molecule is not None:
        want = args.molecule.strip()
        if want not in labels_ok:
            avail = ", ".join(sorted(labels_ok))
            print(
                f"Unknown or unplottable --molecule {want!r}. "
                f"Plottable labels: {avail}",
                file=sys.stderr,
            )
            return 1
        labels_ok = [want]

    n = len(labels_ok)
    fig, axes = plt.subplots(n, 1, figsize=(9.0, 2.2 * n), squeeze=False)
    ax_flat = axes.ravel()

    for ax, label in zip(ax_flat, labels_ok):
        rows = csv_by_label[label]
        exp_pts = exp_by_label[label]
        pred: list[float] = []
        for row in rows:
            sp = (row.get("sigma_para_ppm") or "").strip()
            if not sp:
                print(f"Warning: {label} row missing sigma_para_ppm", file=sys.stderr)
                pred.append(float("nan"))
            else:
                sigma_para = float(sp)
                pred.append(beta0 + beta1 * sigma_para)
        if any(math.isnan(p) for p in pred):
            continue

        ppm_lo = args.ppm_min
        ppm_hi = args.ppm_max

        if args.lorentzian:
            npt = 800
            step = (ppm_hi - ppm_lo) / (npt - 1)
            x_grid = [ppm_lo + i * step for i in range(npt)]
            y_black = spectrum_sum(x_grid, pred, args.fwhm)
            y_red = spectrum_sum(x_grid, exp_pts, args.fwhm)
            ax.plot(x_grid, y_black, color="black", lw=1.4, label="Calculated")
            ax.plot(
                x_grid,
                y_red,
                color="red",
                lw=1.4,
                alpha=0.85,
                label="Actual",
            )
            ax.set_ylabel("Relative intensity")
        else:
            if args.per_proton_sticks:
                plot_stick_overlay(
                    ax,
                    pred,
                    exp_pts,
                    stick_height=1.0,
                    linewidth=args.stick_lw,
                )
            else:
                plot_stick_overlay_grouped(
                    ax,
                    pred,
                    exp_pts,
                    group_tol_ppm=args.group_tol,
                    stick_height=1.0,
                    linewidth=args.stick_lw,
                )
            ax.set_ylabel("Relative intensity")

        title = f"1H NMR - {label}"
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(ppm_hi, ppm_lo)
        if args.lorentzian:
            ax.set_ylim(0.0, 1.05)
        else:
            ax.set_ylim(0.0, 1.22)
        ax.set_xlabel("δ (ppm)")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
