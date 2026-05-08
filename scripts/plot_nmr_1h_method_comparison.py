#!/usr/bin/env python3
"""Overlay CNDO and INDO 1H stick spectra on the same chart.

The script fits one global linear calibration per method directly from the two
feature CSVs, then plots grouped stick spectra for each molecule in the
training list JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def ols_1d(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        raise ValueError("Need at least 2 points for OLS.")
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if abs(den) < 1e-30:
        raise ValueError("Degenerate x spread; cannot fit.")
    beta1 = (n * sxy - sx * sy) / den
    beta0 = (sy - beta1 * sx) / n
    return beta0, beta1


def load_csv_by_molecule(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    by_mol: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("molecule") or "").strip()
            if label:
                by_mol[label].append(row)
    for label, rows in by_mol.items():
        rows.sort(key=lambda row: int((row.get("h_rank") or "0").strip() or "0"))
        by_mol[label] = rows
    return dict(by_mol)


def fit_from_csv(csv_path: Path) -> tuple[float, float]:
    xs: list[float] = []
    ys: list[float] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sigma_para = (row.get("sigma_para_ppm") or "").strip()
            delta_exp = (row.get("delta_exp_ppm") or "").strip()
            if not sigma_para or not delta_exp:
                continue
            xs.append(float(sigma_para))
            ys.append(float(delta_exp))
    return ols_1d(xs, ys)


def cluster_indices_by_delta(values: list[float], tol_ppm: float) -> list[list[int]]:
    if not values:
        return []
    if tol_ppm <= 0.0:
        return [[i] for i in range(len(values))]
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    groups: list[list[int]] = [[order[0]]]
    for idx in order[1:]:
        prev = groups[-1][-1]
        if values[idx] - values[prev] <= tol_ppm:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def cluster_mean(values: list[float], cluster: list[int]) -> float:
    return sum(values[idx] for idx in cluster) / float(len(cluster))


def plot_grouped_series(
    ax,
    values: list[float],
    *,
    group_tol: float,
    height: float,
    linewidth: float,
    color: str,
    label: str,
) -> None:
    clusters = cluster_indices_by_delta(values, group_tol)
    if not clusters:
        return
    for cluster in clusters:
        center = cluster_mean(values, cluster)
        scaled_height = height * float(len(cluster))
        ax.plot(
            [center, center],
            [0.0, scaled_height],
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
        )
    ax.plot([], [], color=color, linewidth=linewidth, label=label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_json", type=Path)
    parser.add_argument("cndo_csv", type=Path)
    parser.add_argument("indo_csv", type=Path)
    parser.add_argument("--molecule", default=None)
    parser.add_argument("--group-tol", type=float, default=0.12)
    parser.add_argument("--stick-lw", type=float, default=2.0)
    parser.add_argument("--ppm-min", type=float, default=0.0)
    parser.add_argument("--ppm-max", type=float, default=10.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("student_output/nmr_1h_method_comparison.png"),
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if args.ppm_min >= args.ppm_max:
        print("--ppm-min must be less than --ppm-max.", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    beta0_cndo, beta1_cndo = fit_from_csv(args.cndo_csv)
    beta0_indo, beta1_indo = fit_from_csv(args.indo_csv)

    root = json.loads(args.training_json.read_text())
    cndo_by_label = load_csv_by_molecule(args.cndo_csv)
    indo_by_label = load_csv_by_molecule(args.indo_csv)

    labels: list[str] = []
    exp_by_label: dict[str, list[float]] = {}
    for mol in root.get("molecules", []):
        label = str(mol.get("label", "")).strip()
        exp = mol.get("experimental_shift_ppm")
        if not label or not isinstance(exp, list):
            continue
        labels.append(label)
        exp_by_label[label] = [float(value) for value in exp]

    plot_labels: list[str] = []
    for label in labels:
        if label not in cndo_by_label or label not in indo_by_label:
            continue
        exp = exp_by_label[label]
        if len(exp) != len(cndo_by_label[label]) or len(exp) != len(indo_by_label[label]):
            print(
                f"Warning: skip {label} because hydrogen counts do not match across inputs.",
                file=sys.stderr,
            )
            continue
        plot_labels.append(label)

    if args.molecule is not None:
        if args.molecule not in plot_labels:
            available = ", ".join(sorted(plot_labels))
            print(f"Unknown --molecule {args.molecule!r}. Available: {available}", file=sys.stderr)
            return 1
        plot_labels = [args.molecule]

    if not plot_labels:
        print("No comparable molecules found.", file=sys.stderr)
        return 1

    max_h_count = 1
    for label in plot_labels:
        counts = [
            len(cluster)
            for values in (exp_by_label[label],)
            for cluster in cluster_indices_by_delta(values, args.group_tol)
        ]
        for rows, beta0, beta1 in (
            (cndo_by_label[label], beta0_cndo, beta1_cndo),
            (indo_by_label[label], beta0_indo, beta1_indo),
        ):
            pred = [beta0 + beta1 * float(row["sigma_para_ppm"]) for row in rows]
            counts.extend(len(cluster) for cluster in cluster_indices_by_delta(pred, args.group_tol))
        if counts:
            max_h_count = max(max_h_count, max(counts))

    fig, axes = plt.subplots(len(plot_labels), 1, figsize=(9.0, 2.4 * len(plot_labels)), squeeze=False)

    for ax, label in zip(axes.ravel(), plot_labels):
        exp = exp_by_label[label]
        cndo_rows = cndo_by_label[label]
        indo_rows = indo_by_label[label]
        cndo_pred = [beta0_cndo + beta1_cndo * float(row["sigma_para_ppm"]) for row in cndo_rows]
        indo_pred = [beta0_indo + beta1_indo * float(row["sigma_para_ppm"]) for row in indo_rows]

        plot_grouped_series(
            ax,
            exp,
            group_tol=args.group_tol,
            height=1.0,
            linewidth=args.stick_lw,
            color="#c00000",
            label="Experimental",
        )
        plot_grouped_series(
            ax,
            cndo_pred,
            group_tol=args.group_tol,
            height=0.86,
            linewidth=args.stick_lw,
            color="black",
            label="CNDO",
        )
        plot_grouped_series(
            ax,
            indo_pred,
            group_tol=args.group_tol,
            height=0.72,
            linewidth=args.stick_lw,
            color="#1f77b4",
            label="INDO",
        )

        ax.set_title(f"1H NMR comparison - {label}")
        ax.set_ylabel("Hydrogen count")
        ax.set_xlabel("delta (ppm)")
        ax.set_xlim(args.ppm_max, args.ppm_min)
        ax.set_ylim(0.0, max_h_count + 0.25)
        ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"CNDO fit: beta0={beta0_cndo:.6f}, beta1={beta1_cndo:.6f}")
    print(f"INDO fit: beta0={beta0_indo:.6f}, beta1={beta1_indo:.6f}")
    print(f"Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())