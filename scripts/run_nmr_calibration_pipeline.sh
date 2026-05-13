#!/usr/bin/env bash
# Rebuild (optional), export training CSV, then fit shift calibration (OLS).
# Usage: from repo root:  ./scripts/run_nmr_calibration_pipeline.sh
#        or:              ./scripts/run_nmr_calibration_pipeline.sh path/to/training_list.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
METHOD_INPUT="${1:-}"
TRAIN_JSON="${2:-$ROOT/sample_input/nmr_1h_training_export.json}"

case "$METHOD_INPUT" in
  --cndo|cndo)
    METHOD_FLAG="--cndo"
    ;;
  --indo|indo)
    METHOD_FLAG="--indo"
    ;;
  --mindo|mindo)
    METHOD_FLAG="--mindo"
    ;;
  *)
    echo "First argument must be --cndo, --indo, or --mindo." >&2
    exit 1
    ;;
esac

EXPORT_BIN="$BUILD_DIR/nmr_1h_training_export"
if [[ ! -x "$EXPORT_BIN" ]]; then
  echo "Missing executable: $EXPORT_BIN" >&2
  echo "Run:  ./build.sh   (or cmake --build \"$BUILD_DIR\")" >&2
  exit 1
fi

OUTPUT_CSV="$(python3 - "$TRAIN_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as handle:
    data = json.load(handle)

print(data["output_csv"])
PY
)"

"$EXPORT_BIN" "$METHOD_FLAG" "$TRAIN_JSON"
python3 "$ROOT/scripts/fit_nmr_shift_calibration.py" "$ROOT/$OUTPUT_CSV"
