#!/bin/sh
# Full fresh-clone workflow. For incremental cases use measure_*.sh directly.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
sh "$script_dir/build.sh"
exec sh "$script_dir/measure_all.sh" "$@"
