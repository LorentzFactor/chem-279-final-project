#!/bin/sh
# Remove the standard build dir, common alternate build dirs, and generated output.
set -eu
ROOT="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$ROOT/build" "$ROOT/build2" "$ROOT/build_local" "$ROOT/build_fresh" \
  "$ROOT/student_output"
