#!/usr/bin/env python3
"""Run 13C CNDO/INDO Takaishi comparison analysis for overlap molecules.

This script:
- finds molecules present in both sample_input JSON configs and Takaishi CSV,
- runs nmr_13c_calc for CNDO and INDO,
- appends one JSON object per run into a single combined JSONL file,
- builds comparison tables against Takaishi and experimental references,
- fits INDO predictions to experimental via linear regression,
- writes four required parity plots under student_output/takaishi.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from matplotlib_defaults import (
    ANNOTATION_FONT_SIZE,
    LEGEND_FONT_SIZE,
    apply_large_plot_text,
)


def normalize_name(value: str) -> str:
    lowered = value.strip().lower().replace("_", "-")
    return "".join(ch for ch in lowered if ch.isalnum())


def label_aliases(value: str) -> set[str]:
    base = value.strip()
    aliases = {normalize_name(base)}
    lowered = base.lower()
    for suffix in ("_pcs2", "-pcs2", " pcs2"):
        if lowered.endswith(suffix):
            trimmed = base[: -len(suffix)].strip(" _-")
            if trimmed:
                aliases.add(normalize_name(trimmed))
    if lowered.endswith("pcs2"):
        trimmed = base[:-4].strip(" _-")
        if trimmed:
            aliases.add(normalize_name(trimmed))
    return {alias for alias in aliases if alias}


@dataclass
class CompoundSeries:
    compound: str
    takaishi: list[float]
    experimental: list[float]


@dataclass
class CalcSeries:
    compound: str
    method: str
    predicted_abs: list[float]


def load_sample_configs(sample_dir: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for cfg_path in sorted(sample_dir.glob("*.json")):
        if cfg_path.name == "nmr_1h_training_export.json":
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        labels = {cfg_path.stem}
        label = cfg.get("label")
        if isinstance(label, str) and label.strip():
            labels.add(label)

        for raw in labels:
            key = normalize_name(raw)
            if key and key not in lookup:
                lookup[key] = cfg_path
    return lookup


def load_takaishi_csv(csv_path: Path) -> dict[str, CompoundSeries]:
    grouped: dict[str, list[tuple[float, float]]] = {}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            compound = (row.get("Compound") or "").strip().strip('"')
            chem = (row.get("Chem Shift (ppm)") or "").strip()
            err = (row.get("Error (Chem Shift)") or "").strip()
            if not compound or not chem or not err:
                continue

            try:
                takaishi = float(chem)
                experimental = float(chem) - float(err)
            except ValueError:
                continue

            key = normalize_name(compound)
            grouped.setdefault(key, []).append((takaishi, experimental))

    out: dict[str, CompoundSeries] = {}
    for key, pairs in grouped.items():
        # Keep pairings intact and sort by decreasing ppm for deterministic rank pairing.
        sorted_pairs = sorted(pairs, key=lambda pair: pair[0], reverse=True)
        out[key] = CompoundSeries(
            compound=key,
            takaishi=[pair[0] for pair in sorted_pairs],
            experimental=[pair[1] for pair in sorted_pairs],
        )
    return out


def run_calc(
    exe: Path,
    method_flag: str,
    molecule_json: Path,
    reference_json: Path,
    combined_jsonl: Path,
) -> None:
    cmd = [
        str(exe),
        method_flag,
        str(molecule_json),
        str(reference_json),
        "--jsonl-out",
        str(combined_jsonl),
    ]
    subprocess.run(cmd, check=True)


def read_grouped_delta(payload: dict[str, Any]) -> list[float]:
    groups = payload.get("groups") or []
    if groups:
        return [float(group["delta_avg_ppm"]) for group in groups]

    carbons = payload.get("carbons") or []
    return [float(row["delta_ppm"]) for row in carbons]


def convert_delta_to_absolute(delta_values: list[float], methane_shift_ppm: float) -> list[float]:
    return [methane_shift_ppm - delta for delta in delta_values]


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def build_row_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        method = str(row.get("method", "")).strip().lower()
        label = str(row.get("molecule_label", "")).strip()
        if not method or not label:
            continue
        for alias in label_aliases(label):
            lookup[(method, alias)] = row
    return lookup


def compute_rmse_mae(y_true: list[float], y_pred: list[float]) -> tuple[float, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.size == 0:
        return float("nan"), float("nan")
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    return rmse, mae


def correlation(xs: list[float], ys: list[float]) -> float:
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if x_arr.size < 2 or y_arr.size < 2:
        return float("nan")
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    try:
        result = pearsonr(x_arr, y_arr)
    except Exception:
        return float("nan")
    r_raw = result[0] if isinstance(result, tuple) else result.statistic
    return float(np.asarray(r_raw).item())


def make_parity_plot(
    out_path: Path,
    x_values: list[float],
    y_values: list[float],
    labels: list[str],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> dict[str, float]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for parity plotting") from exc

    apply_large_plot_text(plt)

    if not x_values or len(x_values) != len(y_values):
        raise ValueError("Parity plot requires non-empty, equal-length x/y vectors.")

    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)

    cmap = plt.get_cmap("tab10")
    palette = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.8))

    for label in unique_labels:
        xs = [x for x, l in zip(x_values, labels) if l == label]
        ys = [y for y, l in zip(y_values, labels) if l == label]
        if len(xs) >= 2:
            ax.plot(xs, ys, linestyle="--", linewidth=1.0, alpha=0.6, color=palette[label])
        ax.scatter(xs, ys, s=28, alpha=0.85, label=label, color=palette[label], edgecolors="none")

    lo = min(min(x_values), min(y_values))
    hi = max(max(x_values), max(y_values))
    pad = max(0.5, 0.06 * (hi - lo if hi > lo else 1.0))
    lo -= pad
    hi += pad

    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.1, linestyle="--", label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    r_val = correlation(x_values, y_values)
    rmse, mae = compute_rmse_mae(x_values, y_values)

    ax.set_title(title)
    metrics_text = f"r = {r_val:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}"
    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=ANNOTATION_FONT_SIZE,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.5"},
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    handles, legend_labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, label="Dashed: same-molecule peak connection"))
    legend_labels.append("Dashed: same-molecule peak connection")
    ax.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(1, 1), fontsize=LEGEND_FONT_SIZE)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"r": r_val, "rmse": rmse, "mae": mae, "n": float(len(x_values))}


def make_parity_plot_grid(
    out_path: Path,
    plot_data: list[tuple[list[float], list[float], list[str], str, str, str]],
) -> dict[str, dict[str, float]]:
    """Create a 2x2 grid of parity plots with a shared legend.

    Args:
        out_path: Output path for the grid figure
        plot_data: List of 6-tuples (x_values, y_values, labels, title, x_label, y_label)

    Returns:
        Dictionary mapping plot names to their metrics
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for parity plotting") from exc

    apply_large_plot_text(plt)

    if len(plot_data) != 4:
        raise ValueError("Grid plot requires exactly 4 plots")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes_flat = axes.flatten()

    all_metrics: dict[str, dict[str, float]] = {}
    all_label_sets: list[list[str]] = []
    plot_names = ["takaishi_vs_experimental", "cndo_vs_experimental", "cndo_vs_takaishi", "indo_fit_vs_experimental"]

    for idx, (x_values, y_values, labels, title, x_label, y_label) in enumerate(plot_data):
        if not x_values or len(x_values) != len(y_values):
            raise ValueError(f"Plot {idx} requires non-empty, equal-length x/y vectors.")

        ax = axes_flat[idx]

        unique_labels: list[str] = []
        for label in labels:
            if label not in unique_labels:
                unique_labels.append(label)
        all_label_sets.append(unique_labels)

        cmap = plt.get_cmap("tab10")
        palette = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}

        for label in unique_labels:
            xs = [x for x, l in zip(x_values, labels) if l == label]
            ys = [y for y, l in zip(y_values, labels) if l == label]
            if len(xs) >= 2:
                ax.plot(xs, ys, linestyle="--", linewidth=0.8, alpha=0.4, color=palette[label], zorder=1)
            ax.scatter(xs, ys, s=50, alpha=0.75, label=label, color=palette[label], edgecolors="black", linewidth=0.4, zorder=2)

        lo = min(min(x_values), min(y_values))
        hi = max(max(x_values), max(y_values))
        pad = max(0.5, 0.08 * (hi - lo if hi > lo else 1.0))
        lo -= pad
        hi += pad

        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.4, linestyle="--", alpha=0.7, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        # Add subtle grid
        ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
        ax.set_axisbelow(True)

        r_val = correlation(x_values, y_values)
        rmse, mae = compute_rmse_mae(x_values, y_values)

        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        metrics_text = f"r = {r_val:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}"
        ax.text(
            0.04,
            0.96,
            metrics_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "white", "alpha": 0.9, "edgecolor": "gray", "linewidth": 0.8},
        )
        ax.set_xlabel(x_label, fontsize=11, fontweight="normal")
        ax.set_ylabel(y_label, fontsize=11, fontweight="normal")
        ax.tick_params(labelsize=9)

        all_metrics[plot_names[idx]] = {"r": r_val, "rmse": rmse, "mae": mae, "n": float(len(x_values))}

    # Create shared legend with all unique molecules
    all_unique_labels: list[str] = []
    for label_set in all_label_sets:
        for label in label_set:
            if label not in all_unique_labels:
                all_unique_labels.append(label)

    cmap = plt.get_cmap("tab10")
    handles: list[Line2D] = []
    legend_labels: list[str] = []

    for i, label in enumerate(all_unique_labels):
        color = cmap(i % 10)
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=7, markeredgecolor="black", markeredgewidth=0.5, label=label))
        legend_labels.append(label)

    handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, alpha=0.7, label="y = x"))
    legend_labels.append("y = x")
    handles.append(
        Line2D([0], [0], color="gray", linestyle="--", linewidth=0.8, alpha=0.4, label="Peak connection")
    )
    legend_labels.append("Peak connection")

    fig.legend(
        handles,
        legend_labels,
        loc="center left",
        ncol=1,
        fontsize=10,
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    fig.suptitle("13C NMR Parity Analysis: Takaishi Reference vs Computational Methods", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 0.92, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    return all_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nmr-13c-calc",
        type=Path,
        default=Path("build/nmr_13c_calc"),
        help="Path to nmr_13c_calc executable.",
    )
    parser.add_argument(
        "--sample-input",
        type=Path,
        default=Path("sample_input"),
        help="Directory with molecule JSON configs.",
    )
    parser.add_argument(
        "--takaishi-csv",
        type=Path,
        default=Path("takaishi_data/table1_paraffins_nmr.csv"),
        help="Takaishi reference CSV.",
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=Path("sample_input/methane.json"),
        help="Reference config passed to nmr_13c_calc.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("student_output/takaishi"),
        help="Output directory for analysis artifacts.",
    )
    args = parser.parse_args()

    if not args.nmr_13c_calc.is_file():
        print(f"Missing executable: {args.nmr_13c_calc}", file=sys.stderr)
        return 1
    if not args.sample_input.is_dir():
        print(f"Missing sample input directory: {args.sample_input}", file=sys.stderr)
        return 1
    if not args.takaishi_csv.is_file():
        print(f"Missing CSV: {args.takaishi_csv}", file=sys.stderr)
        return 1
    if not args.reference_json.is_file():
        print(f"Missing reference JSON: {args.reference_json}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale_json in args.out_dir.glob("*_13c_calc_*.json"):
        stale_json.unlink(missing_ok=True)

    sample_lookup = load_sample_configs(args.sample_input)
    csv_lookup = load_takaishi_csv(args.takaishi_csv)

    overlap_keys = sorted(set(sample_lookup.keys()) & set(csv_lookup.keys()))
    if not overlap_keys:
        print("No overlap compounds found between sample_input and Takaishi CSV.", file=sys.stderr)
        return 1

    methane_key = normalize_name("methane")
    if methane_key not in csv_lookup or not csv_lookup[methane_key].takaishi:
        print("Takaishi CSV must include methane Chem Shift for absolute conversion.", file=sys.stderr)
        return 1
    methane_shift = float(csv_lookup[methane_key].takaishi[0])

    combined_jsonl = args.out_dir / "combined_results.jsonl"
    combined_jsonl.write_text("", encoding="utf-8")

    cndo_series: dict[str, CalcSeries] = {}
    indo_series: dict[str, CalcSeries] = {}
    warnings: list[str] = []

    for key in overlap_keys:
        molecule_json = sample_lookup[key]

        print(f"Running CNDO/INDO for {molecule_json.stem} ...")
        run_calc(args.nmr_13c_calc, "--cndo", molecule_json, args.reference_json, combined_jsonl)
        run_calc(args.nmr_13c_calc, "--indo", molecule_json, args.reference_json, combined_jsonl)

    jsonl_rows = load_jsonl_rows(combined_jsonl)
    row_lookup = build_row_lookup(jsonl_rows)

    for key in overlap_keys:
        cndo_row = row_lookup.get(("cndo", key))
        indo_row = row_lookup.get(("indo", key))
        if cndo_row is None or indo_row is None:
            print(f"Missing JSONL rows for {key} (cndo/indo).", file=sys.stderr)
            return 1

        cndo_delta = read_grouped_delta(cndo_row)
        indo_delta = read_grouped_delta(indo_row)

        cndo_abs = sorted(convert_delta_to_absolute(cndo_delta, methane_shift), reverse=True)
        indo_abs = sorted(convert_delta_to_absolute(indo_delta, methane_shift), reverse=True)

        cndo_series[key] = CalcSeries(compound=key, method="cndo", predicted_abs=cndo_abs)
        indo_series[key] = CalcSeries(compound=key, method="indo", predicted_abs=indo_abs)

    comparison_rows: list[dict[str, Any]] = []

    tak_vs_exp_x: list[float] = []
    tak_vs_exp_y: list[float] = []
    tak_vs_exp_labels: list[str] = []

    cndo_vs_exp_x: list[float] = []
    cndo_vs_exp_y: list[float] = []
    cndo_vs_exp_labels: list[str] = []

    cndo_vs_tak_x: list[float] = []
    cndo_vs_tak_y: list[float] = []
    cndo_vs_tak_labels: list[str] = []

    indo_vs_exp_x_raw: list[float] = []
    indo_vs_exp_y: list[float] = []
    indo_vs_exp_labels: list[str] = []

    for key in overlap_keys:
        refs = csv_lookup[key]
        cndo = cndo_series[key]
        indo = indo_series[key]

        n_ref = min(len(refs.takaishi), len(refs.experimental))
        n_cmp = min(n_ref, len(cndo.predicted_abs), len(indo.predicted_abs))
        if n_cmp <= 0:
            warnings.append(f"No comparable peaks for {key}")
            continue
        if len(refs.takaishi) != len(cndo.predicted_abs) or len(refs.takaishi) != len(indo.predicted_abs):
            warnings.append(
                f"Peak-count mismatch for {key}: csv={len(refs.takaishi)} cndo={len(cndo.predicted_abs)} indo={len(indo.predicted_abs)}; truncating to {n_cmp}."
            )

        for idx in range(n_cmp):
            takaishi_ppm = refs.takaishi[idx]
            experimental_ppm = refs.experimental[idx]
            cndo_ppm = cndo.predicted_abs[idx]
            indo_ppm = indo.predicted_abs[idx]

            comparison_rows.append(
                {
                    "compound": key,
                    "peak_rank": idx + 1,
                    "takaishi_ppm": takaishi_ppm,
                    "experimental_ppm": experimental_ppm,
                    "cndo_predicted_ppm": cndo_ppm,
                    "indo_predicted_ppm_raw": indo_ppm,
                }
            )

            tak_vs_exp_x.append(experimental_ppm)
            tak_vs_exp_y.append(takaishi_ppm)
            tak_vs_exp_labels.append(key)

            cndo_vs_exp_x.append(experimental_ppm)
            cndo_vs_exp_y.append(cndo_ppm)
            cndo_vs_exp_labels.append(key)

            cndo_vs_tak_x.append(takaishi_ppm)
            cndo_vs_tak_y.append(cndo_ppm)
            cndo_vs_tak_labels.append(key)

            indo_vs_exp_x_raw.append(indo_ppm)
            indo_vs_exp_y.append(experimental_ppm)
            indo_vs_exp_labels.append(key)

    if len(indo_vs_exp_x_raw) < 2:
        print("Not enough points to fit INDO linear regression.", file=sys.stderr)
        return 1

    x_fit = np.asarray(indo_vs_exp_x_raw, dtype=float).reshape(-1, 1)
    y_fit = np.asarray(indo_vs_exp_y, dtype=float)
    reg = LinearRegression()
    reg.fit(x_fit, y_fit)
    beta0 = float(reg.intercept_)
    beta1 = float(reg.coef_[0])
    indo_vs_exp_y_fit = reg.predict(x_fit).tolist()

    for row in comparison_rows:
        row["indo_predicted_ppm_fit"] = beta0 + beta1 * float(row["indo_predicted_ppm_raw"])

    csv_out = args.out_dir / "comparisons.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "compound",
            "peak_rank",
            "takaishi_ppm",
            "experimental_ppm",
            "cndo_predicted_ppm",
            "indo_predicted_ppm_raw",
            "indo_predicted_ppm_fit",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    grid_metrics = make_parity_plot_grid(
        args.out_dir / "parity_plots_grid.png",
        [
            (
                tak_vs_exp_x,
                tak_vs_exp_y,
                tak_vs_exp_labels,
                "13C NMR Takaishi vs Experimental",
                "Experimental (ppm)",
                "Takaishi (ppm)",
            ),
            (
                cndo_vs_exp_x,
                cndo_vs_exp_y,
                cndo_vs_exp_labels,
                "13C NMR CNDO/2 Predictions vs Experimental",
                "Experimental (ppm)",
                "CNDO/2 Predicted (ppm)",
            ),
            (
                cndo_vs_tak_x,
                cndo_vs_tak_y,
                cndo_vs_tak_labels,
                "13C NMR CNDO/2 Predictions vs Takaishi",
                "Takaishi (ppm)",
                "CNDO/2 Predicted (ppm)",
            ),
            (
                indo_vs_exp_y,
                indo_vs_exp_y_fit,
                indo_vs_exp_labels,
                "13C NMR INDO (Linear Fit) vs Experimental",
                "Experimental (ppm)",
                "INDO Fitted (ppm)",
            ),
        ],
    )

    metrics_out = args.out_dir / "indo_linear_fit.json"
    fit_rmse, fit_mae = compute_rmse_mae(indo_vs_exp_y, indo_vs_exp_y_fit)
    metrics_payload = {
        "model": "experimental_ppm = beta0 + beta1 * indo_predicted_ppm_raw",
        "beta0": beta0,
        "beta1": beta1,
        "n_points": len(indo_vs_exp_x_raw),
        "fit_r": correlation(indo_vs_exp_y, indo_vs_exp_y_fit),
        "fit_rmse": fit_rmse,
        "fit_mae": fit_mae,
        "plot_metrics": grid_metrics,
        "molecules_analyzed": overlap_keys,
        "warnings": warnings,
    }
    metrics_out.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {combined_jsonl}")
    print(f"Wrote {csv_out}")
    print(f"Wrote {metrics_out}")
    print(f"Wrote {args.out_dir / 'parity_plots_grid.png'}")

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
