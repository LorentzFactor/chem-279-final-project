#!/usr/bin/env python3
"""Generate one combined figure for CNDO vs INDO 1H and 13C stick plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from matplotlib_defaults import (
    FIGURE_TITLE_SIZE,
    LEGEND_FONT_SIZE,
    apply_large_plot_text,
)


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


def cluster_mean(values: list[float], cluster: list[int]) -> float:
    return sum(values[idx] for idx in cluster) / float(len(cluster))


def grouped_centers_and_weights(values: list[float], tol_ppm: float) -> tuple[list[float], list[int]]:
    clusters = cluster_indices_by_delta(values, tol_ppm)
    centers = [cluster_mean(values, cluster) for cluster in clusters]
    weights = [len(cluster) for cluster in clusters]
    return centers, weights


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


def compute_global_limits(
    rows: list[dict],
    key_groups: list[str],
    *,
    pad: float,
    default_limits: tuple[float, float],
) -> tuple[float, float]:
    min_center = None
    max_center = None
    for row in rows:
        centers: list[float] = []
        for key in key_groups:
            centers.extend(row.get(key, []))
        if not centers:
            continue
        row_min = min(centers)
        row_max = max(centers)
        min_center = row_min if min_center is None else min(min_center, row_min)
        max_center = row_max if max_center is None else max(max_center, row_max)
    if min_center is None or max_center is None:
        return default_limits
    return max_center + pad, min_center - pad


def plot_series(
    ax,
    centers: list[float],
    weights: list[int],
    *,
    color: str,
    label: str,
    height: float,
    linewidth: float,
    normalize: bool = True,
) -> None:
    if not centers:
        return
    max_weight = max(weights) or 1
    for center, weight in zip(centers, weights):
        if normalize:
            scaled_height = height * (float(weight) / float(max_weight))
        else:
            scaled_height = height * float(weight)
        ax.plot(
            [center, center],
            [0.0, scaled_height],
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
        )
    ax.plot([], [], color=color, linewidth=linewidth, label=label)


def load_1h_rows(training_json: Path, cndo_csv: Path, indo_csv: Path) -> tuple[list[dict], tuple[float, float], tuple[float, float]]:
    beta_cndo = fit_from_csv(cndo_csv)
    beta_indo = fit_from_csv(indo_csv)
    root = json.loads(training_json.read_text())
    cndo_by_label = load_csv_by_molecule(cndo_csv)
    indo_by_label = load_csv_by_molecule(indo_csv)

    rows: list[dict] = []
    for mol in root.get("molecules", []):
        label = str(mol.get("label", "")).strip()
        exp = mol.get("experimental_shift_ppm")
        if not label or not isinstance(exp, list):
            continue
        if label not in cndo_by_label or label not in indo_by_label:
            continue
        exp_pts = [float(value) for value in exp]
        cndo_rows = cndo_by_label[label]
        indo_rows = indo_by_label[label]
        if len(exp_pts) != len(cndo_rows) or len(exp_pts) != len(indo_rows):
            continue
        rows.append(
            {
                "label": label,
                "exp": exp_pts,
                "cndo_pred": [beta_cndo[0] + beta_cndo[1] * float(row["sigma_para_ppm"]) for row in cndo_rows],
                "indo_pred": [beta_indo[0] + beta_indo[1] * float(row["sigma_para_ppm"]) for row in indo_rows],
            }
        )
    return rows, beta_cndo, beta_indo


def load_13c_rows(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    rows: list[dict] = []
    for entry in manifest.get("entries", []):
        main_cfg = json.loads(Path(entry["main_json"]).read_text())
        cndo = json.loads(Path(entry["cndo_calc_json"]).read_text())
        indo = json.loads(Path(entry["indo_calc_json"]).read_text())
        label = str(main_cfg.get("label") or cndo.get("molecule_label") or Path(entry["main_json"]).stem)
        tol = float(main_cfg.get("shift_grouping_tol_ppm", cndo.get("shift_grouping_tol_ppm", 0.2)))
        exp = main_cfg.get("experimental_13c_shift_ppm")
        exp_centers: list[float] = []
        exp_weights: list[int] = []
        if isinstance(exp, list) and exp:
            exp_centers, exp_weights = grouped_centers_and_weights([float(value) for value in exp], tol)
        cndo_centers, cndo_weights = grouped_series_from_calc(cndo)
        indo_centers, indo_weights = grouped_series_from_calc(indo)
        rows.append(
            {
                "label": label,
                "exp_centers": exp_centers,
                "exp_weights": exp_weights,
                "cndo_centers": cndo_centers,
                "cndo_weights": cndo_weights,
                "indo_centers": indo_centers,
                "indo_weights": indo_weights,
            }
        )
    return rows


def style_axis(ax, *, title: str, x_max: float, x_min: float, y_max: float, y_label: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("delta (ppm)")
    ax.set_xlim(x_max, x_min)
    ax.set_ylim(0.0, y_max)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=LEGEND_FONT_SIZE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_json", type=Path)
    parser.add_argument("cndo_csv", type=Path)
    parser.add_argument("indo_csv", type=Path)
    parser.add_argument("carbon_manifest_json", type=Path)
    parser.add_argument("--group-tol-1h", type=float, default=0.12)
    parser.add_argument("--ppm-min-1h", type=float, default=None)
    parser.add_argument("--ppm-max-1h", type=float, default=None)
    parser.add_argument("--ppm-pad-1h", type=float, default=0.35)
    parser.add_argument("--ppm-pad-13c", type=float, default=4.0)
    parser.add_argument("--ppm-min-13c", type=float, default=None)
    parser.add_argument("--ppm-max-13c", type=float, default=None)
    parser.add_argument("--stick-lw", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("student_output/nmr_method_comparison_overview.png"),
    )
    args = parser.parse_args()

    if (
        args.ppm_min_1h is not None
        and args.ppm_max_1h is not None
        and args.ppm_min_1h >= args.ppm_max_1h
    ):
        print("Invalid 1H ppm range.", file=sys.stderr)
        return 1
    if (
        args.ppm_min_13c is not None
        and args.ppm_max_13c is not None
        and args.ppm_min_13c >= args.ppm_max_13c
    ):
        print("Invalid 13C ppm range.", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    apply_large_plot_text(plt)

    h_rows, beta_cndo, beta_indo = load_1h_rows(args.training_json, args.cndo_csv, args.indo_csv)
    c_rows = load_13c_rows(args.carbon_manifest_json)
    nrows = max(len(h_rows), len(c_rows))
    if nrows == 0:
        print("No comparable spectra found.", file=sys.stderr)
        return 1

    if args.ppm_min_1h is not None and args.ppm_max_1h is not None:
        h1_xlim = (args.ppm_max_1h, args.ppm_min_1h)
    else:
        h1_xlim = compute_global_limits(
            h_rows,
            ["exp", "cndo_pred", "indo_pred"],
            pad=args.ppm_pad_1h,
            default_limits=(10.0, 0.0),
        )

    max_h1_weight = 1
    for row in h_rows:
        exp_centers, exp_weights = grouped_centers_and_weights(row["exp"], args.group_tol_1h)
        cndo_centers, cndo_weights = grouped_centers_and_weights(row["cndo_pred"], args.group_tol_1h)
        indo_centers, indo_weights = grouped_centers_and_weights(row["indo_pred"], args.group_tol_1h)
        row_weights = exp_weights + cndo_weights + indo_weights
        if row_weights:
            max_h1_weight = max(max_h1_weight, max(row_weights))

    if args.ppm_min_13c is not None and args.ppm_max_13c is not None:
        c13_xlim = (args.ppm_max_13c, args.ppm_min_13c)
    else:
        c13_xlim = compute_global_limits(
            c_rows,
            ["exp_centers", "cndo_centers", "indo_centers"],
            pad=args.ppm_pad_13c,
            default_limits=(220.0, 0.0),
        )

    fig, axes = plt.subplots(nrows, 2, figsize=(15.0, 2.6 * nrows), squeeze=False)

    for row_idx in range(nrows):
        ax_h = axes[row_idx][0]
        ax_c = axes[row_idx][1]

        if row_idx < len(h_rows):
            row = h_rows[row_idx]
            exp_centers, exp_weights = grouped_centers_and_weights(row["exp"], args.group_tol_1h)
            cndo_centers, cndo_weights = grouped_centers_and_weights(row["cndo_pred"], args.group_tol_1h)
            indo_centers, indo_weights = grouped_centers_and_weights(row["indo_pred"], args.group_tol_1h)
            h1_centers = exp_centers + cndo_centers + indo_centers
            plot_series(ax_h, exp_centers, exp_weights, color="#c00000", label="Experimental", height=1.0, linewidth=args.stick_lw, normalize=False)
            plot_series(ax_h, cndo_centers, cndo_weights, color="black", label="CNDO", height=1.0, linewidth=args.stick_lw, normalize=False)
            plot_series(ax_h, indo_centers, indo_weights, color="#1f77b4", label="INDO", height=1.0, linewidth=args.stick_lw, normalize=False)
            style_axis(
                ax_h,
                title=f"1H - {row['label']}",
                x_max=h1_xlim[0],
                x_min=h1_xlim[1],
                y_max=max_h1_weight + 0.25,
                y_label="Hydrogen count",
            )
        else:
            ax_h.axis("off")

        if row_idx < len(c_rows):
            row = c_rows[row_idx]
            row_centers = row["exp_centers"] + row["cndo_centers"] + row["indo_centers"]
            if not row_centers:
                ax_c.axis("off")
            else:
                if row["exp_centers"]:
                    plot_series(ax_c, row["exp_centers"], row["exp_weights"], color="#c00000", label="Experimental", height=1.0, linewidth=args.stick_lw)
                plot_series(ax_c, row["cndo_centers"], row["cndo_weights"], color="black", label="CNDO", height=0.86, linewidth=args.stick_lw)
                plot_series(ax_c, row["indo_centers"], row["indo_weights"], color="#1f77b4", label="INDO", height=0.72, linewidth=args.stick_lw)
                style_axis(
                    ax_c,
                    title=f"13C - {row['label']}",
                    x_max=c13_xlim[0],
                    x_min=c13_xlim[1],
                    y_max=1.08,
                    y_label="Relative intensity",
                )
        else:
            ax_c.axis("off")

    fig.suptitle(
        "CNDO vs INDO NMR Comparison\n"
        f"1H fits: CNDO beta0={beta_cndo[0]:.3f}, beta1={beta_cndo[1]:.3f}; "
        f"INDO beta0={beta_indo[0]:.3f}, beta1={beta_indo[1]:.3f}",
        fontsize=FIGURE_TITLE_SIZE,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())