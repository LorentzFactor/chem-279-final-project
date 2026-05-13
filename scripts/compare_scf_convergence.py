#!/usr/bin/env python3
"""Compare SCF convergence for fixed-point and DIIS across sample inputs.

Runs molecule_energy_calc in JSON mode for both iteration methods for each
molecule config in sample_input and reports convergence outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

ITERATION_METHODS: tuple[tuple[str, str], ...] = (
    ("--fixed-point", "fixed-point"),
    ("--diis", "diis"),
)


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
        help="Path to molecule_energy_calc executable",
    )
    parser.add_argument(
        "--method",
        choices=["cndo", "indo"],
        default="cndo",
        help="Electronic structure method flag passed to calculator",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=repo_root / "student_output" / "scf_convergence_comparison.csv",
        help="Path for CSV export of convergence comparison results",
    )
    return parser.parse_args()


def discover_configs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def run_calculation(
    calculator: Path,
    repo_root: Path,
    method_flag: str,
    iteration_flag: str,
    config_path: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        [str(calculator), method_flag, iteration_flag, "--json", str(config_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def as_display(value: Any) -> str:
    return "" if value is None else str(value)


def write_csv(rows: list[dict[str, Any]], csv_out: Path) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "molecule",
        "method",
        "iteration_method",
        "status",
        "scf_iterations",
        "total_energy_ev",
        "error",
    ]
    with csv_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: blank_if_none(row.get(name)) for name in fieldnames})


def print_markdown_summary(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["molecule"], {})[row["iteration_method"]] = row

    print("| Molecule | Fixed-point status | Fixed-point iters | DIIS status | DIIS iters |")
    print("| --- | --- | ---: | --- | ---: |")

    for molecule in sorted(grouped):
        fixed_row = grouped[molecule].get("fixed-point", {})
        diis_row = grouped[molecule].get("diis", {})

        fixed_status = fixed_row.get("status", "missing")
        fixed_iters = as_display(fixed_row.get("scf_iterations", ""))
        diis_status = diis_row.get("status", "missing")
        diis_iters = as_display(diis_row.get("scf_iterations", ""))

        print(
            f"| {molecule} | {fixed_status} | {fixed_iters} | {diis_status} | {diis_iters} |"
        )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    method_flag = f"--{args.method}"

    if not args.calculator.exists():
        raise FileNotFoundError(f"Calculator executable not found: {args.calculator}")

    configs = discover_configs(args.input_dir)
    rows: list[dict[str, Any]] = []

    for config_path in configs:
        molecule = config_path.stem
        for iteration_flag, iteration_name in ITERATION_METHODS:
            try:
                result = run_calculation(
                    args.calculator.resolve(),
                    repo_root,
                    method_flag,
                    iteration_flag,
                    config_path.resolve(),
                )
                rows.append(
                    {
                        "molecule": molecule,
                        "method": result.get("method", args.method.upper()),
                        "iteration_method": iteration_name,
                        "status": "converged",
                        "scf_iterations": result.get("scf_iterations"),
                        "total_energy_ev": result.get("total_energy_ev"),
                        "error": "",
                    }
                )
            except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
                error_message = (
                    exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
                )
                rows.append(
                    {
                        "molecule": molecule,
                        "method": args.method.upper(),
                        "iteration_method": iteration_name,
                        "status": "failed",
                        "scf_iterations": None,
                        "total_energy_ev": None,
                        "error": error_message,
                    }
                )

    write_csv(rows, args.csv_out)
    print_markdown_summary(rows)
    print(f"\nWrote CSV: {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
