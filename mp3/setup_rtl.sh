#!/bin/sh
set -eu
MP3_RTL_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "${1:-}" = "--skip-install" ]; then
    if [ ! -x "$MP3_RTL_ROOT/.venv/bin/python" ]; then
        echo "No prepared environment. Run sh mp3/setup_rtl.sh first." >&2
        exit 1
    fi
else
    sh "$MP3_RTL_ROOT/setup.sh"
fi
"$MP3_RTL_ROOT/.venv/bin/python" "$MP3_RTL_ROOT/tools/prepare_sources.py" --only miniLLAMA
"$MP3_RTL_ROOT/.venv/bin/python" "$MP3_RTL_ROOT/tools/setup_rtl.py" "$@"
