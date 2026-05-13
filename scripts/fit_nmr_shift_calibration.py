#!/usr/bin/env python3
"""
Fit δ ≈ β0 + β1 * sigma_para_ppm from CSV produced by nmr_1h_training_export.

Usage:
  python3 scripts/fit_nmr_shift_calibration.py student_output/nmr_1h_training_features.csv

Requires column delta_exp_ppm to be non-empty for all rows you want in the fit
(fill from experiment / literature, same order as hydrogens in the export).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def ols_1d(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (beta0, beta1, rmse) for y ≈ beta0 + beta1 * x."""
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
    sse = sum((y - (beta0 + beta1 * x)) ** 2 for x, y in zip(xs, ys))
    rmse = math.sqrt(sse / n)
    return beta0, beta1, rmse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="CSV from nmr_1h_training_export")
    args = ap.parse_args()

    xs: list[float] = []
    ys: list[float] = []
    rows_used: list[str] = []

    with args.csv.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            dcell = (row.get("delta_exp_ppm") or "").strip()
            if not dcell:
                continue
            try:
                y = float(dcell)
                x = float(row["sigma_para_ppm"])
            except (KeyError, ValueError) as e:
                print(f"Skip row (bad data): {row} ({e})", file=sys.stderr)
                continue
            xs.append(x)
            ys.append(y)
            rows_used.append(
                f"{row.get('molecule','?')} atom={row.get('atom_index','?')}"
            )

    if len(xs) < 2:
        print(
            "Need at least 2 rows with numeric delta_exp_ppm. "
            "Fill the column in the CSV (from experiment) and retry.",
            file=sys.stderr,
        )
        return 1

    b0, b1, rmse = ols_1d(xs, ys)
    print(f"Fitted on n={len(xs)} hydrogens.")
    print(f"  beta0 (intercept) = {b0:.6f}")
    print(f"  beta1 (slope)     = {b1:.6f}")
    print(f"  RMSE (ppm)        = {rmse:.6f}")
    print("Prediction: delta_pred_ppm = beta0 + beta1 * sigma_para_ppm")
    print()
    print("To use in nmr_1h_calc, add to the **main** molecule JSON (or reference):")
    print(f'  "shift_calibration_beta0": {b0},')
    print(f'  "shift_calibration_beta1": {b1}')
    print("(omit both to keep uncalibrated δ = σ_ref,avg − σ_total.)")
    print('Optional: "shift_calibration_relative": true  →  ')
    print("  δ = beta1 * (sigma_para_main − mean(sigma_para_ref)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
