#!/usr/bin/env bash
# Generate one combined overlay figure with CNDO and INDO on the same chart.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p student_output

BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
H_EXPORT="$BUILD_DIR/nmr_1h_training_export"
C_CALC="$BUILD_DIR/nmr_13c_calc"
OVERVIEW_PLOT="$ROOT/scripts/plot_nmr_method_comparison_overview.py"
PARITY_PLOT="$ROOT/scripts/plot_nmr_method_parity.py"
TRAIN_JSON="${1:-$ROOT/sample_input/nmr_1h_training_export.json}"
REF_JSON="$ROOT/sample_input/methane.json"
C_CSV="$ROOT/student_output/nmr_1h_training_features_cndo.csv"
I_CSV="$ROOT/student_output/nmr_1h_training_features_indo.csv"
OUT_PNG="${2:-$ROOT/student_output/nmr_method_comparison_overview.png}"
PARITY_OUT="${3:-$ROOT/student_output/nmr_method_parity_plots.png}"

for required in "$H_EXPORT" "$C_CALC" "$OVERVIEW_PLOT" "$PARITY_PLOT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
CARBON_MANIFEST="$tmpdir/nmr_13c_method_comparison_manifest.json"

if [[ ! -f "$TRAIN_JSON" ]]; then
  echo "Missing training JSON: $TRAIN_JSON" >&2
  exit 1
fi

rewrite_output_csv() {
  local src_json="$1"
  local dst_json="$2"
  local output_csv="$3"
  python3 - "$src_json" "$dst_json" "$output_csv" <<'PY'
import json
import sys

src, dst, out_csv = sys.argv[1:4]
with open(src, encoding='utf-8') as handle:
    data = json.load(handle)
data['output_csv'] = out_csv
with open(dst, 'w', encoding='utf-8') as handle:
    json.dump(data, handle, indent=2)
    handle.write('\n')
PY
}

echo "Using BUILD_DIR=$BUILD_DIR"
rewrite_output_csv "$TRAIN_JSON" "$tmpdir/nmr_1h_training_export_cndo.json" "student_output/nmr_1h_training_features_cndo.csv"
rewrite_output_csv "$TRAIN_JSON" "$tmpdir/nmr_1h_training_export_indo.json" "student_output/nmr_1h_training_features_indo.csv"

echo "=== 1H: export CNDO and INDO training CSVs ==="
"$H_EXPORT" --cndo "$tmpdir/nmr_1h_training_export_cndo.json"
"$H_EXPORT" --indo "$tmpdir/nmr_1h_training_export_indo.json"

echo "=== 13C: calculate CNDO and INDO comparison inputs ==="
entries_json="[]"

# Dynamically discover all molecules with experimental_13c_shift_ppm in sample_input/
while IFS= read -r main_json; do
  [[ -z "$main_json" ]] && continue
  
  # Extract label from filename (e.g., "propane.json" -> "propane")
  label=$(basename "$main_json" .json)
  
  cndo_json="$ROOT/student_output/${label}_13c_calc_cndo.json"
  indo_json="$ROOT/student_output/${label}_13c_calc_indo.json"

  echo "  $label"
  "$C_CALC" --cndo "$main_json" "$REF_JSON" "$cndo_json"
  "$C_CALC" --indo "$main_json" "$REF_JSON" "$indo_json"
  entries_json="$entries_json"$'\n'"$main_json|$cndo_json|$indo_json"
done < <(python3 - "$ROOT/sample_input" <<'PY'
import json
from pathlib import Path
import sys

sample_dir = Path(sys.argv[1])
for json_file in sorted(sample_dir.glob("*.json")):
    try:
        data = json.loads(json_file.read_text())
        if "experimental_13c_shift_ppm" in data:
            print(str(json_file))
    except (json.JSONDecodeError, IOError):
        pass
PY
)

ENTRIES_JSON="$entries_json" python3 - "$CARBON_MANIFEST" <<'PY'
import json
import os
import sys

lines = os.environ.get("ENTRIES_JSON", "").splitlines()
entries = []
for line in lines:
  line = line.strip()
  if not line or "|" not in line:
    continue
  main_json, cndo_json, indo_json = line.split("|", 2)
  entries.append(
    {
      "main_json": main_json,
      "cndo_calc_json": cndo_json,
      "indo_calc_json": indo_json,
    }
  )

with open(sys.argv[1], "w", encoding="utf-8") as handle:
  json.dump({"entries": entries}, handle, indent=2)
  handle.write("\n")
PY

echo "=== Overview plot ==="
ENTRIES_JSON="$entries_json" python3 "$OVERVIEW_PLOT" "$TRAIN_JSON" "$C_CSV" "$I_CSV" "$CARBON_MANIFEST" --out "$OUT_PNG"

echo "=== Parity plots ==="
python3 "$PARITY_PLOT" "$TRAIN_JSON" "$C_CSV" "$I_CSV" "$CARBON_MANIFEST" --out "$PARITY_OUT"

echo "Done. Wrote $OUT_PNG and $PARITY_OUT"