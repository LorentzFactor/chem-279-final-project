#!/usr/bin/env python3
"""Overlay CNDO and INDO 13C grouped stick plots on the same chart."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from matplotlib_defaults import LEGEND_FONT_SIZE, apply_large_plot_text


def cluster_indices_by_delta(values: list[float], tol_ppm: float) -> list[list[int]]:
    if not values:
        return []
    if tol_ppm <= 0.0:
        return [[idx] for idx in range(len(values))]
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    groups: list[list[int]] = [[order[0]]]
    for idx in order[1:]:
        prev = groups[-1][-1]
        if values[idx] - values[prev] <= tol_ppm:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def grouped_series_from_calc(data: dict) -> tuple[list[float], list[int]]:
    groups = data.get("groups") or []
    if groups:
        centers = [float(group["delta_avg_ppm"]) for group in groups]
        weights = [int(group.get("n_carbons", 1)) for group in groups]
        return centers, weights
    carbons = data.get("carbons") or []
    centers = [float(row["delta_ppm"]) for row in carbons]
    weights = [1] * len(centers)
    return centers, weights


def grouped_series_from_experimental(values: list[float], tol_ppm: float) -> tuple[list[float], list[int]]:
    groups = cluster_indices_by_delta(values, tol_ppm)
    centers = [sum(values[idx] for idx in group) / float(len(group)) for group in groups]
    weights = [len(group) for group in groups]
    return centers, weights


def plot_series(ax, centers: list[float], weights: list[int], *, color: str, label: str, height: float, linewidth: float) -> None:
    if not centers:
        return
    max_weight = max(weights) or 1
    for center, weight in zip(centers, weights):
        scaled_height = height * (float(weight) / float(max_weight))
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
    parser.add_argument("main_json", type=Path)
    parser.add_argument("cndo_calc_json", type=Path)
    parser.add_argument("indo_calc_json", type=Path)
    parser.add_argument("--stick-lw", type=float, default=2.0)
    parser.add_argument("--ppm-pad", type=float, default=4.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("student_output/nmr_13c_method_comparison.png"),
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    apply_large_plot_text(plt)

    main_cfg = json.loads(args.main_json.read_text())
    cndo = json.loads(args.cndo_calc_json.read_text())
    indo = json.loads(args.indo_calc_json.read_text())

    cndo_centers, cndo_weights = grouped_series_from_calc(cndo)
    indo_centers, indo_weights = grouped_series_from_calc(indo)
    tol = float(main_cfg.get("shift_grouping_tol_ppm", cndo.get("shift_grouping_tol_ppm", 0.2)))

    exp = main_cfg.get("experimental_13c_shift_ppm")
    exp_centers: list[float] = []
    exp_weights: list[int] = []
    if isinstance(exp, list) and exp:
        exp_values = [float(value) for value in exp]
        exp_centers, exp_weights = grouped_series_from_experimental(exp_values, tol)

    all_centers = cndo_centers + indo_centers + exp_centers
    if not all_centers:
        print("No peaks to plot.", file=sys.stderr)
        return 1

    try:
        label = str(main_cfg["label"])
    except KeyError:
        label = str(cndo.get("molecule_label") or args.main_json.stem)

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 2.8))
    if exp_centers:
        plot_series(
            ax,
            exp_centers,
            exp_weights,
            color="#c00000",
            label="Experimental",
            height=1.0,
            linewidth=args.stick_lw,
        )
    plot_series(
        ax,
        cndo_centers,
        cndo_weights,
        color="black",
        label="CNDO",
        height=0.86,
        linewidth=args.stick_lw,
    )
    plot_series(
        ax,
        indo_centers,
        indo_weights,
        color="#1f77b4",
        label="INDO",
        height=0.72,
        linewidth=args.stick_lw,
    )

    ax.set_title(f"13C NMR comparison - {label}")
    ax.set_ylabel("Relative intensity")
    ax.set_xlabel("delta (ppm)")
    ax.set_xlim(max(all_centers) + args.ppm_pad, min(all_centers) - args.ppm_pad)
    ax.set_ylim(0.0, 1.08)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=LEGEND_FONT_SIZE)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())