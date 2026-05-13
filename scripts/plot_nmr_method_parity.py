#!/usr/bin/env python3
"""Generate parity plots for 1H/13C and CNDO/INDO."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from matplotlib_defaults import FIGURE_TITLE_SIZE, LEGEND_FONT_SIZE, apply_large_plot_text


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


def correlation_coefficient(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    syy = sum(y * y for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    num = n * sxy - sx * sy
    den_x = n * sxx - sx * sx
    den_y = n * syy - sy * sy
    if den_x <= 0.0 or den_y <= 0.0:
        return float("nan")
    return num / math.sqrt(den_x * den_y)


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


def build_1h_points(training_json: Path, csv_path: Path) -> tuple[list[dict], tuple[float, float]]:
    root = json.loads(training_json.read_text())
    by_molecule = load_csv_by_molecule(csv_path)

    fit_x: list[float] = []
    fit_y: list[float] = []
    for rows in by_molecule.values():
        for row in rows:
            sigma_para = (row.get("sigma_para_ppm") or "").strip()
            delta_exp = (row.get("delta_exp_ppm") or "").strip()
            if not sigma_para or not delta_exp:
                continue
            fit_x.append(float(sigma_para))
            fit_y.append(float(delta_exp))
    beta0, beta1 = ols_1d(fit_x, fit_y)

    points: list[dict] = []
    for mol in root.get("molecules", []):
        label = str(mol.get("label", "")).strip()
        exp = mol.get("experimental_shift_ppm")
        rows = by_molecule.get(label)
        if not label or not isinstance(exp, list) or rows is None:
            continue
        if len(exp) != len(rows):
            continue
        for row, exp_val in zip(rows, exp):
            sigma_para = float(row["sigma_para_ppm"])
            pred = beta0 + beta1 * sigma_para
            points.append({"label": label, "exp": float(exp_val), "pred": pred})
    return points, (beta0, beta1)


def build_13c_points(manifest_path: Path, method_key: str) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    points: list[dict] = []
    json_key = f"{method_key}_calc_json"
    for entry in manifest.get("entries", []):
        main_cfg = json.loads(Path(entry["main_json"]).read_text())
        exp = main_cfg.get("experimental_13c_shift_ppm")
        if not isinstance(exp, list):
            continue
        calc = json.loads(Path(entry[json_key]).read_text())
        carbons = calc.get("carbons") or []
        if len(carbons) != len(exp):
            continue
        label = str(main_cfg.get("label") or calc.get("molecule_label") or Path(entry["main_json"]).stem)
        for calc_row, exp_val in zip(carbons, exp):
            points.append({"label": label, "exp": float(exp_val), "pred": float(calc_row["delta_ppm"])})
    return points


def correlation_from_points(points: list[dict]) -> float:
    if not points:
        return float("nan")
    return correlation_coefficient(
        [pt["exp"] for pt in points],
        [pt["pred"] for pt in points],
    )


def draw_parity(ax, points: list[dict], *, title: str, palette: dict[str, str]) -> None:
    if not points:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    labels_in_order: list[str] = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        label = point["label"]
        if label not in grouped:
            labels_in_order.append(label)
        grouped[label].append(point)

    min_val = min(min(pt["exp"], pt["pred"]) for pt in points)
    max_val = max(max(pt["exp"], pt["pred"]) for pt in points)
    pad = max(0.25, 0.05 * (max_val - min_val if max_val > min_val else 1.0))
    lo = min_val - pad
    hi = max_val + pad

    for label in labels_in_order:
        xs = [pt["exp"] for pt in grouped[label]]
        ys = [pt["pred"] for pt in grouped[label]]
        ax.scatter(xs, ys, s=28, alpha=0.85, label=label, color=palette[label], edgecolors="none")

    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("Experimental delta (ppm)")
    ax.set_ylabel("Predicted delta (ppm)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=LEGEND_FONT_SIZE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_json", type=Path)
    parser.add_argument("cndo_csv", type=Path)
    parser.add_argument("indo_csv", type=Path)
    parser.add_argument("carbon_manifest_json", type=Path)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("student_output/nmr_method_parity_plots.png"),
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    apply_large_plot_text(plt)

    h1_cndo, h1_cndo_fit = build_1h_points(args.training_json, args.cndo_csv)
    h1_indo, h1_indo_fit = build_1h_points(args.training_json, args.indo_csv)
    c13_cndo = build_13c_points(args.carbon_manifest_json, "cndo")
    c13_indo = build_13c_points(args.carbon_manifest_json, "indo")

    molecule_labels = []
    for points in (h1_cndo, h1_indo, c13_cndo, c13_indo):
        for pt in points:
            if pt["label"] not in molecule_labels:
                molecule_labels.append(pt["label"])
    cmap = plt.get_cmap("tab10")
    palette = {label: cmap(i % 10) for i, label in enumerate(molecule_labels)}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    draw_parity(
        axes[0][0],
        h1_cndo,
        title=f"1H CNDO parity (r={correlation_from_points(h1_cndo):.3f})",
        palette=palette,
    )
    draw_parity(
        axes[0][1],
        h1_indo,
        title=f"1H INDO parity (r={correlation_from_points(h1_indo):.3f})",
        palette=palette,
    )
    draw_parity(
        axes[1][0],
        c13_cndo,
        title=f"13C CNDO parity (r={correlation_from_points(c13_cndo):.3f})",
        palette=palette,
    )
    draw_parity(
        axes[1][1],
        c13_indo,
        title=f"13C INDO parity (r={correlation_from_points(c13_indo):.3f})",
        palette=palette,
    )

    h1_cndo_r = correlation_from_points(h1_cndo)
    h1_indo_r = correlation_from_points(h1_indo)
    c13_cndo_r = correlation_from_points(c13_cndo)
    c13_indo_r = correlation_from_points(c13_indo)
    fig.suptitle(
        "NMR Parity Plots\n"
        f"1H CNDO r={h1_cndo_r:.3f}, "
        f"1H INDO r={h1_indo_r:.3f}, "
        f"13C CNDO r={c13_cndo_r:.3f}, "
        f"13C INDO r={c13_indo_r:.3f}",
        fontsize=FIGURE_TITLE_SIZE,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"1H CNDO correlation: {h1_cndo_r:.6f}")
    print(f"1H INDO correlation: {h1_indo_r:.6f}")
    print(f"13C CNDO correlation: {c13_cndo_r:.6f}")
    print(f"13C INDO correlation: {c13_indo_r:.6f}")
    print(f"Wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())