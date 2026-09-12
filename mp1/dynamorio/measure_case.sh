#!/bin/sh
# Shared entry point; the public wrappers select a workload.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$script_dir/.venv/bin/python" ]; then
  echo "Build first: sh $script_dir/build.sh" >&2
  exit 1
fi
exec "$script_dir/.venv/bin/python" "$script_dir/measure.py" "$@"
