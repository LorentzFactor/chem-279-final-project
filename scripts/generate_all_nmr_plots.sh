#!/usr/bin/env bash
# Regenerate 1H training CSV + per-molecule 1H/13C stick plots into student_output/.
# From repo root:  ./scripts/generate_all_nmr_plots.sh
# Optional:  BUILD_DIR=/path/to/build  (defaults to $ROOT/build)
# Requires: built nmr_1h_training_export, nmr_13c_calc; python3 + matplotlib.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p student_output

BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
echo "Using BUILD_DIR=$BUILD_DIR"
H_EXPORT="$BUILD_DIR/nmr_1h_training_export"
H_CALC_PLOT="$ROOT/scripts/plot_nmr_training_spectrum.py"
C_CALC="$BUILD_DIR/nmr_13c_calc"
C_PLOT="$ROOT/scripts/plot_nmr_13c_calc_peaks.py"
COMPARE_PLOTS="$ROOT/scripts/generate_all_nmr_method_comparison_plots.sh"
TRAIN_JSON="$ROOT/sample_input/nmr_1h_training_export.json"
CAL_JSON="$ROOT/sample_input/methane.json"
REF_JSON="$ROOT/sample_input/methane.json"

for bin in "$H_EXPORT" "$C_CALC"; do
  if [[ ! -x "$bin" ]]; then
    echo "Missing executable: $bin" >&2
    echo "Configure and build, e.g.:  cmake -S . -B build && cmake --build build -j4" >&2
    exit 1
  fi
done

echo "=== 1H: export training CSV ==="
"$H_EXPORT" "$TRAIN_JSON"

echo "=== 1H: per-molecule plots ==="
while IFS= read -r label; do
  [[ -z "$label" ]] && continue
  out="$ROOT/student_output/nmr_1h_${label}.png"
  echo "  $label -> $out"
  python3 "$H_CALC_PLOT" "$TRAIN_JSON" "$ROOT/student_output/nmr_1h_training_features.csv" \
    --calibration-json "$CAL_JSON" --molecule "$label" --out "$out"
done <<'LABELS'
methane
ethane
propane
isobutane
n-butane
n-pentane
LABELS

main_json_for_label() {
  case "$1" in
    methane) echo "$ROOT/sample_input/methane.json" ;;
    ethane) echo "$ROOT/sample_input/ethane.json" ;;
    propane) echo "$ROOT/sample_input/propane.json" ;;
    isobutane) echo "$ROOT/sample_input/isobutane.json" ;;
    n-butane) echo "$ROOT/sample_input/n_butane.json" ;;
    n-pentane) echo "$ROOT/sample_input/n_pentane.json" ;;
    *) echo "" ;;
  esac
}

echo "=== 13C: calc JSON + plots ==="
while IFS= read -r label; do
  [[ -z "$label" ]] && continue
  mj="$(main_json_for_label "$label")"
  if [[ -z "$mj" || ! -f "$mj" ]]; then
    echo "  skip $label (no main JSON mapping)" >&2
    continue
  fi
  jout="$ROOT/student_output/${label}_13c_calc.json"
  png="$ROOT/student_output/nmr_13c_${label}.png"
  echo "  $label: $C_CALC -> $jout -> $png"
  "$C_CALC" "$mj" "$REF_JSON" "$jout"
  python3 "$C_PLOT" "$mj" "$REF_JSON" --calc-json "$jout" --out "$png"
done <<'LABELS'
methane
ethane
propane
isobutane
n-butane
n-pentane
LABELS

if [[ -f "$COMPARE_PLOTS" ]]; then
  echo "=== CNDO vs INDO comparison plots ==="
  BUILD_DIR="$BUILD_DIR" "$COMPARE_PLOTS"
fi

echo "Done. Outputs under student_output/"
