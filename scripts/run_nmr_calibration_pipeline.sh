#!/usr/bin/env bash
# Rebuild (optional), export training CSV, then fit shift calibration (OLS).
# Usage: from repo root:  ./scripts/run_nmr_calibration_pipeline.sh
#        or:              ./scripts/run_nmr_calibration_pipeline.sh path/to/training_list.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
TRAIN_JSON="${1:-$ROOT/sample_input/nmr_1h_training_export.json}"

EXPORT_BIN="$BUILD_DIR/nmr_1h_training_export"
if [[ ! -x "$EXPORT_BIN" ]]; then
  echo "Missing executable: $EXPORT_BIN" >&2
  echo "Run:  ./build.sh   (or cmake --build \"$BUILD_DIR\")" >&2
  exit 1
fi

"$EXPORT_BIN" "$TRAIN_JSON"
python3 "$ROOT/scripts/fit_nmr_shift_calibration.py" "$ROOT/student_output/nmr_1h_training_features.csv"
