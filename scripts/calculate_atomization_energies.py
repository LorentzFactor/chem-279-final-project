#!/usr/bin/env python3
"""Calculate CNDO/2 and INDO atomization energies for referenced molecules.

Scans sample input JSON files, keeps only those with
reference_values.atomization_energy.units == "kJ/mol", runs molecule_energy_calc
in JSON mode for both methods, and prints a comparison table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from matplotlib_defaults import (
    ANNOTATION_FONT_SIZE,
    FIGURE_TITLE_SIZE,
    SMALL_ANNOTATION_FONT_SIZE,
    apply_large_plot_text,
)


METHODS = (
    ("cndo", "CNDO/2"),
    ("indo", "INDO"),
)


@dataclass(frozen=True)
class ReferenceCase:
    config_path: Path
    label: str
    reference_kj_per_mol: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "sample_input",
        help="Directory containing molecule JSON inputs",
    )
    parser.add_argument(
        "--calculator",
        type=Path,
        default=repo_root / "build" / "molecule_energy_calc",
        help="Path to the molecule_energy_calc executable",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=repo_root / "student_output" / "atomization_energies.csv",
        help="Path for CSV export of per-molecule atomization energy comparisons",
    )
    parser.add_argument(
        "--indo-vs-cndo2-out",
        type=Path,
        default=repo_root / "student_output" / "atomization_energy_parity_INDO_CNDO2.png",
        help="Path for INDO vs CNDO/2 (no reference) parity plot image",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Saved plot resolution",
    )
    parser.add_argument(
        "--fail-on-run-error",
        action="store_true",
        help="Exit immediately if one calculator run fails",
    )
    return parser.parse_args()


def load_reference_case(config_path: Path) -> ReferenceCase | None:
    with config_path.open() as handle:
        config = json.load(handle)

    atomization = (
        config.get("reference_values", {})
        .get("atomization_energy")
    )
    if not isinstance(atomization, dict):
        return None

    if atomization.get("units") != "kJ/mol":
        return None

    value = atomization.get("value")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None

    label = config.get("label") or config_path.stem
    return ReferenceCase(
        config_path=config_path,
        label=str(label),
        reference_kj_per_mol=float(value),
    )


def discover_reference_cases(input_dir: Path) -> list[ReferenceCase]:
    cases: list[ReferenceCase] = []
    for config_path in sorted(input_dir.glob("*.json")):
        case = load_reference_case(config_path)
        if case is not None:
            cases.append(case)
    return cases


def run_calculator(calculator: Path, repo_root: Path, method_flag: str, config_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(calculator), method_flag, "--json", str(config_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def format_number(value: float) -> str:
    return f"{value:12.3f}"


def print_results_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'Molecule':<18} {'Method':<8} {'Reference':>12} {'Calculated':>12}"
        f" {'Error':>12} {'Abs Error':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label']:<18} {row['method']:<8}"
            f" {format_number(row['reference_kj_per_mol'])}"
            f" {format_number(row['calculated_kj_per_mol'])}"
            f" {format_number(row['error_kj_per_mol'])}"
            f" {format_number(row['abs_error_kj_per_mol'])}"
        )


def print_summary(rows: list[dict[str, Any]]) -> None:
    print()
    for _, method_name in METHODS:
        method_rows = [row for row in rows if row["method"] == method_name]
        if not method_rows:
            continue
        mae = sum(row["abs_error_kj_per_mol"] for row in method_rows) / len(method_rows)
        mse = sum(row["error_kj_per_mol"] ** 2 for row in method_rows) / len(method_rows)
        rmse = math.sqrt(mse)
        rho = pearson_rho(method_rows)
        print(
            f"{method_name}: n={len(method_rows)}, "
            f"MAE={mae:.3f} kJ/mol, RMSE={rmse:.3f} kJ/mol, rho={rho:.4f}"
        )


def pearson_rho(method_rows: list[dict[str, Any]]) -> float:
    if len(method_rows) < 2:
        return float("nan")

    xs = [row["reference_kj_per_mol"] for row in method_rows]
    ys = [row["calculated_kj_per_mol"] for row in method_rows]

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    if var_x <= 0.0 or var_y <= 0.0:
        return float("nan")

    return cov / math.sqrt(var_x * var_y)


def write_csv(rows: list[dict[str, Any]], csv_out: Path) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "method",
        "reference_kj_per_mol",
        "calculated_kj_per_mol",
        "error_kj_per_mol",
        "abs_error_kj_per_mol",
    ]
    with csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})



def _jitter(vals: list[float], scale: float, seed: int = 0) -> list[float]:
    import numpy as np
    rng = np.random.default_rng(seed)
    return (np.asarray(vals, dtype=float) + rng.normal(0.0, scale, len(vals))).tolist()


_METHOD_COLORS = {
    "CNDO/2": "#1f77b4",
    "INDO": "#d62728",
}



def write_parity_plot_indo_vs_cndo2(rows: list[dict[str, Any]], plot_out: Path, dpi: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for parity plot export") from exc

    apply_large_plot_text(plt)

    cndo_map = {row["label"]: row["calculated_kj_per_mol"] for row in rows if row["method"] == "CNDO/2"}
    indo_map = {row["label"]: row["calculated_kj_per_mol"] for row in rows if row["method"] == "INDO"}
    paired_labels = sorted(set(cndo_map) & set(indo_map))

    fig, ax = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    ax.set_xlabel("CNDO/2 atomization energy (kJ/mol)")
    ax.set_ylabel("INDO atomization energy (kJ/mol)")

    if not paired_labels:
        ax.text(0.5, 0.5, "No paired data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(plot_out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    xs = [cndo_map[lbl] for lbl in paired_labels]
    ys = [indo_map[lbl] for lbl in paired_labels]

    all_vals = xs + ys
    jitter_scale = 0.015 * (max(all_vals) - min(all_vals)) if all_vals else 0.0
    jxs = xs
    jys = _jitter(ys, jitter_scale, seed=1)

    ax.scatter(jxs, jys, s=55, alpha=0.9, color=_METHOD_COLORS["CNDO/2"], edgecolors="none", zorder=3)

    all_vals = xs + ys
    span = max(all_vals) - min(all_vals)
    pad = max(5.0, 0.08 * span if span > 0.0 else 10.0)
    lo, hi = min(all_vals) - pad, max(all_vals) + pad

    _label_nudge = {
        "isobutane": (6, -4),
        "2,3-dimethylbutane": (-105, 2),
        "ethane": (10, -4),
        "HF": (-8, 7),
        "H2": (4, -6),
        "methane": (4, 3),
    }
    for lbl, x_val, y_val in zip(paired_labels, jxs, jys):
        ox, oy = _label_nudge.get(lbl, (4, 4))
        ax.annotate(
            lbl,
            xy=(x_val, y_val),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=SMALL_ANNOTATION_FONT_SIZE,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.9),
        )
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    diffs = [y - x for x, y in zip(xs, ys)]
    abs_diffs = [abs(d) for d in diffs]
    mae = sum(abs_diffs) / len(abs_diffs)
    pseudo_rows = [{"reference_kj_per_mol": x, "calculated_kj_per_mol": y} for x, y in zip(xs, ys)]
    rho = pearson_rho(pseudo_rows)  # type: ignore[arg-type]
    ax.text(
        0.03, 0.97,
        f"MAE={mae:.1f} kJ/mol\nrho={rho:.4f}",
        ha="left", va="top",
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    fig.suptitle("Atomization Energy: INDO vs CNDO/2", fontsize=FIGURE_TITLE_SIZE)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir.resolve()
    calculator = args.calculator.resolve()
    csv_out = args.csv_out.resolve()
    indo_vs_cndo2_out = args.indo_vs_cndo2_out.resolve()

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1
    if not calculator.is_file():
        print(f"Calculator executable not found: {calculator}", file=sys.stderr)
        return 1

    cases = discover_reference_cases(input_dir)
    if not cases:
        print("No molecule configs with atomization_energy reference values in kJ/mol were found.")
        return 0

    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for case in cases:
        config_arg = case.config_path.relative_to(repo_root)
        for method_flag, method_name in METHODS:
            try:
                result = run_calculator(
                    calculator=calculator,
                    repo_root=repo_root,
                    method_flag=f"--{method_flag}",
                    config_path=config_arg,
                )
            except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
                failures.append(f"{case.label} [{method_name}]: {exc}")
                if args.fail_on_run_error:
                    print(failures[-1], file=sys.stderr)
                    return 1
                continue

            calculated = result["atomization_energy_kj_per_mol"]
            error = calculated - case.reference_kj_per_mol
            rows.append(
                {
                    "label": case.label,
                    "method": method_name,
                    "reference_kj_per_mol": case.reference_kj_per_mol,
                    "calculated_kj_per_mol": calculated,
                    "error_kj_per_mol": error,
                    "abs_error_kj_per_mol": abs(error),
                }
            )

    if rows:
        print_results_table(rows)
        print_summary(rows)
        write_csv(rows, csv_out)
        print(f"Wrote CSV: {csv_out}")
        try:
            write_parity_plot_indo_vs_cndo2(rows, indo_vs_cndo2_out, args.dpi)
            print(f"Wrote INDO vs CNDO/2 plot: {indo_vs_cndo2_out}")
        except RuntimeError as exc:
            print(f"Could not write plot: {exc}", file=sys.stderr)
            return 1

    if failures:
        print(file=sys.stderr)
        print("Run failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())