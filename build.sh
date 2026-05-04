#!/bin/sh
# Single out-of-tree build directory: ./build (relative to this repo).
set -eu
ROOT="$(cd "$(dirname "$0")" && pwd)"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Debug
cmake --build "$ROOT/build" -j4
