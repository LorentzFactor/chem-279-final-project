#!/bin/sh
# Interactive shell in the course dev container, with extra Python deps for
# scripts/plot_nmr_training_spectrum.py (matplotlib). Rebuilds the local image
# when this file or Dockerfile.interactive / python/requirements-viz.txt change
# (docker layer cache keeps it fast after the first build).
#
# Optional: CHEM279_INTERACTIVE_IMAGE=mytag  to override the local image name.
# Refresh the upstream base on rebuild:  docker build --pull ...

set -eu
ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${CHEM279_INTERACTIVE_IMAGE:-chem279-interactive:latest}"

docker build --pull -t "$IMAGE" -f "$ROOT/Dockerfile.interactive" "$ROOT"

docker run \
  --rm \
  --interactive \
  --tty \
  --volume "$ROOT:/work" \
  --workdir /work \
  "$IMAGE"
