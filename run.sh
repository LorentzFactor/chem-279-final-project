#!/usr/bin/env bash
# run.sh — install Python dependencies, build C++ executables, generate all plots.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# 1. Python virtual environment and package installation
# ---------------------------------------------------------------------------
VENV="$ROOT/.venv"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "=== Creating Python virtual environment ==="
  python3 -m venv "$VENV"
fi

echo "=== Installing Python packages ==="
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet \
  "matplotlib>=3.7" \
  numpy \
  scipy \
  scikit-learn

PYTHON_BIN="$VENV/bin/python"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# 2. Build C++ executables
# ---------------------------------------------------------------------------
echo "=== Building executables ==="
"$ROOT/build.sh"

# ---------------------------------------------------------------------------
# 3. Prepare output directory
# ---------------------------------------------------------------------------
mkdir -p "$ROOT/student_output"

# ---------------------------------------------------------------------------
# 4. Atomization energy parity plot
# ---------------------------------------------------------------------------
echo "=== Generating atomization_energy_parity.png ==="
"$PYTHON_BIN" "$ROOT/scripts/calculate_atomization_energies.py"

# ---------------------------------------------------------------------------
# 5. Takaishi 13C analysis plots  (student_output/takaishi/)
# ---------------------------------------------------------------------------
echo "=== Generating Takaishi 13C analysis plots ==="
PYTHON_BIN="$PYTHON_BIN" "$ROOT/scripts/run_takaishi_13c_analysis.sh"

# ---------------------------------------------------------------------------
# 6. NMR method parity plots  (student_output/nmr_method_parity_plots.png)
# ---------------------------------------------------------------------------
echo "=== Generating NMR method parity plots ==="
PYTHON_BIN="$PYTHON_BIN" "$ROOT/scripts/generate_all_nmr_method_comparison_plots.sh"

echo ""
echo "All plots generated:"
echo "  student_output/atomization_energy_parity.png"
echo "  student_output/takaishi/*.png"
echo "  student_output/nmr_method_parity_plots.png"
