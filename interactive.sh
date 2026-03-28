#!/bin/sh

docker run \
  --pull=always \
  --rm \
  --interactive \
  --tty \
  --volume "$(pwd):/work" \
  ghcr.io/berkeley-chem-179-279/dev:latest