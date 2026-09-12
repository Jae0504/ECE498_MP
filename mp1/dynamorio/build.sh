#!/bin/sh
# Local, pinned dependencies + Release builds + correctness gate; no sudo.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/build_local.py" "$@"
