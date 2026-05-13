#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
CALC="${NMR_13C_CALC:-$BUILD_DIR/nmr_13c_calc}"
OUT_DIR="${OUT_DIR:-$ROOT/student_output/takaishi}"

if [[ ! -x "$CALC" ]]; then
  echo "Missing executable: $CALC" >&2
  echo "Build first (e.g. run ./build.sh) or set NMR_13C_CALC." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi

"$PYTHON_BIN" "$ROOT/scripts/analyze_takaishi_13c.py" \
  --nmr-13c-calc "$CALC" \
  --sample-input "$ROOT/sample_input" \
  --takaishi-csv "$ROOT/takaishi_data/table1_paraffins_nmr.csv" \
  --reference-json "$ROOT/sample_input/methane.json" \
  --out-dir "$OUT_DIR"

echo "Done. Artifacts are in $OUT_DIR"
