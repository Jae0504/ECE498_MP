#!/bin/sh
# Run from any directory. Installs only into mp3/.venv and mp3/.deps.
set -eu

MP3_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MP3_BOOTSTRAP_PYTHON=${MP3_PYTHON:-python3}

"$MP3_BOOTSTRAP_PYTHON" -c 'import sys; assert (3, 9) <= sys.version_info[:2] <= (3, 12), "Use Python 3.9–3.12 (set MP3_PYTHON if needed)"'
if ! command -v git >/dev/null 2>&1; then
    echo "Git is required to initialize the miniLLAMA submodule." >&2
    exit 1
fi

"$MP3_BOOTSTRAP_PYTHON" "$MP3_ROOT/tools/prepare_sources.py" --only miniLLAMA
"$MP3_BOOTSTRAP_PYTHON" -m venv "$MP3_ROOT/.venv"
MP3_ENV_PYTHON="$MP3_ROOT/.venv/bin/python"
# Older bundled pip versions can reject dependency metadata on the CPU wheel index.
"$MP3_ENV_PYTHON" -m pip install --upgrade pip
"$MP3_ENV_PYTHON" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
"$MP3_ENV_PYTHON" -m pip install -r "$MP3_ROOT/requirements_rtl.txt"
"$MP3_ENV_PYTHON" -m pytest -q "$MP3_ROOT/tests"
echo "MP3/MP4 base Python environment PASS. Required RTL tool setup:"
echo "  sh mp3/setup_rtl.sh"
